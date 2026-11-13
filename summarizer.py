"""
Bot Bot — Session Summarizer (Part 8)

Summarizes a completed conversation session into 3-5 sentences using
the local OpenVINO NPU model (via npu_server.py), and saves it via
memory.py. These summaries let Bot Bot recall things from past sessions
beyond just the last 10 messages of the current one.

No Ollama anywhere in this stack — everything routes through
npu_server.py, which is the OpenVINO NPU inference server.
"""

import re

import httpx

from memory import get_all_turns, save_session_summary, session_has_summary

NPU_SERVER_URL = "http://localhost:11435/api/chat"
MODEL_NAME = "botbot"

# Observed real-world generation time for a single summarization call on
# this NPU hardware can exceed 120s (confirmed via npu_server.py's own
# log: "Generated 512 chars in 130.23s"), which was silently killing
# otherwise-successful calls at the old 120s httpx timeout — the call
# wasn't hung or failing, it just needed a few more seconds than the
# timeout allowed. 300s gives real headroom above the slowest observed
# call rather than trimming it to the bare minimum.
NPU_CALL_TIMEOUT_SECONDS = 300

SUMMARY_INSTRUCTION = """
Summarize the following conversation in 3-5 sentences. Focus on key
topics discussed, emotions expressed, goals or plans mentioned, and
anything that would be useful to remember for a future conversation.
Write the summary in third person, factually, as private notes for
future reference — not as a message to the user. Do not include
greetings or filler, just the substance.

IMPORTANT — source of truth: only record facts, goals, plans, and
emotions that the User actually stated. "Bot Bot" lines are the
assistant's own responses, not verified information. Do not record a
claim, recollection, or agreement made by Bot Bot (e.g. "I remember
that", "that's right", "yeah, it's been a while") as an established
fact unless the User's own words in this conversation confirm it. If
Bot Bot said something that the User did not actually state or
confirm, leave it out of the summary entirely rather than repeating it
as fact.
"""

# _consolidate_segments() used to reuse SUMMARY_INSTRUCTION, which is written
# for a raw "User:/Bot Bot:" transcript and doesn't describe its input
# (already-written summary segments).
CONSOLIDATE_SEGMENTS_INSTRUCTION = """
You will be given several partial summaries ("Segments") of ONE long
conversation, in time order. Merge them into a single summary of 3-6
sentences. Write in third person, factually, as private notes for future
reference, not as a message to the user. Keep important facts, emotions, goals,
and plans, and prefer later segments if two segments conflict. Only use what
the segments actually say. Do not add, guess, or infer anything new.
"""

ROLLUP_SUMMARY_INSTRUCTION = """
You are maintaining a rolling summary of an ongoing conversation. You will be
given the existing summary so far (if any) and a new batch of turns that
happened after it. Fold the new turns into the existing summary, producing a
single updated summary in 3-6 sentences. Preserve important facts, emotions,
goals, and plans from the existing summary that are still relevant — do not
drop them just because the new batch doesn't repeat them. Write in third
person, factually, as private notes for future reference — not as a message
to the user. Do not include greetings or filler, just the substance.

IMPORTANT — source of truth: only record facts, goals, plans, and
emotions that the User actually stated, in this batch or in the
existing summary. "Bot Bot" lines are the assistant's own responses,
not verified information. Do not record a claim, recollection, or
agreement made by Bot Bot (e.g. "I remember that", "that's right",
"yeah, it's been a while") as an established fact unless the User's
own words confirm it. If Bot Bot said something that the User did not
actually state or confirm, leave it out of the summary — including
dropping it from the existing summary if it was mistakenly captured
there before, rather than carrying it forward again.
"""


# Hard cap on how many chars of raw conversation go into a single NPU
# call from this file. NPU's compiled ceiling is 6144 tokens; this stays
# well under that even at a pessimistic ~2 chars/token, leaving headroom
# for SUMMARY_INSTRUCTION/ROLLUP_SUMMARY_INSTRUCTION + the existing-summary
# block + generation itself. Confirmed necessary by a real crash: a full,
# uncapped session sent as one prompt hit ~38,895 tokens against the 6144
# ceiling (see PROJECT_STATE.txt) — get_all_turns() has no built-in limit,
# so this file must enforce one itself rather than assuming callers do.
MAX_CHARS_PER_SUMMARY_BATCH = 6000

# Cap on how large the running fold summary is allowed to get inside
# summarize_session()'s multi-batch loop before that loop freezes it off
# as a completed segment and starts folding the remaining batches into a
# fresh "" instead of continuing to grow it — the same chunk-boundary
# pattern memory.py's maybe_rollup_session() uses for rolling_summaries,
# applied here to summarize_session()'s in-memory fold, which threaded one
# unbounded growing `summary` string through the whole loop before this.
# Same 6000 figure as MAX_CHARS_PER_SUMMARY_BATCH on purpose — both are
# bounding "how much of one side goes into a single NPU call" against the
# same NPU_HARD_TOKEN_LIMIT, just for the existing-summary side here
# rather than the new-turns side.
SEGMENT_MAX_CHARS = 6000

# Matches long runs of base64-alphabet characters with no whitespace —
# the signature of an embedded attachment payload riding inside a turn's
# text content. These compress to zero useful summary signal but can,
# by themselves, blow a single turn past MAX_CHARS_PER_SUMMARY_BATCH,
# which chunking-by-turn can never fix (the turn is bigger than the
# whole batch budget before it's even grouped with anything else).
_BASE64_BLOB_RE = re.compile(r"[A-Za-z0-9+/=]{200,}")

# Reserved space per turn for the "Role: " prefix, the newline joining
# turns, and the truncation marker itself, so a capped turn's rendered
# line still comfortably fits inside MAX_CHARS_PER_SUMMARY_BATCH on its
# own, with margin to spare.
_PER_TURN_OVERHEAD_CHARS = 100


def _sanitize_turn_content(content: str, max_chars: int) -> str:
    """
    Makes a single turn's content safe to send to the NPU on its own
    within max_chars, in two passes:
      1. Redact any embedded base64-like blob (attachment payload) down
         to a short placeholder noting how much was removed.
      2. If content is still too long after that (e.g. a genuinely long
         stretch of real natural-language text), hard-truncate with a
         marker rather than letting it through oversized.
    This is a safety net for the pathological single-turn case only;
    ordinary turns pass through unchanged.
    """
    def _redact(match: "re.Match") -> str:
        return f"[attachment data omitted, {len(match.group(0))} chars]"

    sanitized = _BASE64_BLOB_RE.sub(_redact, content)

    budget = max_chars - _PER_TURN_OVERHEAD_CHARS
    if len(sanitized) > budget:
        sanitized = sanitized[:budget] + "... [truncated]"

    return sanitized


def _chunk_turns_by_chars(turns: list, max_chars: int) -> list[list]:
    """
    Splits a flat list of turn dicts into consecutive chunks, each kept
    under max_chars of rendered "Role: content" text. Never splits a
    single turn across chunks.

    Each turn's content is first run through _sanitize_turn_content(),
    so a lone turn that would otherwise exceed max_chars by itself (most
    commonly a base64 attachment payload) is redacted/truncated down to
    fit rather than being sent to the NPU oversized — that was a known
    gap: chunking-by-turn gave such a turn its own chunk, but that chunk
    could still exceed the NPU's prompt-length ceiling on its own.
    Turns are copied, not mutated in place, so callers' original turn
    dicts are left untouched.
    """
    chunks = []
    current: list = []
    current_len = 0
    for t in turns:
        content = _sanitize_turn_content(t["content"], max_chars)
        sanitized_t = {**t, "content": content}
        line_len = len(content) + 20  # rough allowance for "Role: " prefix + newline
        if current and current_len + line_len > max_chars:
            chunks.append(current)
            current = []
            current_len = 0
        current.append(sanitized_t)
        current_len += line_len
    if current:
        chunks.append(current)
    return chunks


def _is_corrupted_or_glitched_turn(content: str) -> bool:
    """Detects corrupted/glitched assistant turns (comma stutters, loops, hallucinated poetry)."""
    if not content:
        return False
    lower = content.lower()
    glitched_phrases = [
        "still here. not fixed",
        "(quiet)",
        "(whisper)",
        "in the ripples",
        "in the wind",
        "you're not, you're not",
        "i'm, i'm, i'm",
        "yes, yes, yes",
    ]
    if any(p in lower for p in glitched_phrases):
        return True
    if re.search(r'(\b[\w\'\s]{1,20}\b)(?:,\s*\1){2,}', lower):
        return True
    return False


async def summarize_session(
    session_id: str,
    force: bool = False,
    npu_lock=None,
    known_people: set[str] | None = None,
    is_guest: bool = False,
) -> str | None:
    """
    Generates and saves a summary for the given session by sending its
    turn history to npu_server.py in size-capped batches, folded together
    incrementally via the same rollup logic used by summarize_turns_batch()
    — rather than concatenating the entire session into a single prompt,
    which has no upper bound and can (and did) exceed the NPU's hard
    MAX_PROMPT_LEN=6144 ceiling on longer sessions. Skips sessions that
    already have a summary unless force=True. Returns the summary text, or
    None if there was nothing to summarize.

    The multi-batch fold itself is size-bounded too: the running summary
    threaded through the loop is capped at SEGMENT_MAX_CHARS, so a very
    long session doesn't just move the same unbounded-growth problem from
    a single oversized prompt into a series of ever-larger ones. Once the
    running summary crosses that cap, it's frozen off as a completed
    segment and folding continues into a fresh "" — ordinary sessions stay
    well under one segment and never notice this; only unusually long ones
    produce more than one, in which case _consolidate_segments() folds
    them into the single final summary that gets saved.

    npu_lock - optional; when provided, serializes every NPU call this
               function makes against other callers (main.py's live /chat
               path, the follow-up loop, and summarize_turns_batch()) via
               the same NPUThrottle instance, same reasoning as
               summarize_turns_batch()'s existing npu_lock parameter. If
               not provided, falls back to no locking (previous behavior)
               rather than raising, so existing callers that don't pass it
               don't break.
    known_people - optional (Step 2); rag.py's get_known_people() set,
               passed straight through to memory.save_session_summary()
               so the saved summary gets person/activity tagged for
               get_recent_summaries()'s query-aware retrieval. Not
               imported directly here — same reasoning as npu_lock: the
               caller (main.py) already has rag.py imported, and this
               file stays free of a memory.py<->rag.py import cycle.
               Omitting it (or passing None/empty) is fine: the summary
               still saves, just without person tags until a later
               backfill_summary_tags() call fixes it up.
    """
    if not force and session_has_summary(session_id):
        return None

    turns = get_all_turns(session_id)
    if not turns:
        return None
    turns = [t for t in turns if not (t.get("role") == "assistant" and _is_corrupted_or_glitched_turn(t.get("content", "")))]
    if not turns:
        return None

    batches = _chunk_turns_by_chars(turns, MAX_CHARS_PER_SUMMARY_BATCH)

    if len(batches) == 1:
        # Common case (short/normal sessions): one call, same prompt shape
        # as before this fix.
        conversation_text = "\n".join(
            f"{'User' if t['role'] == 'user' else 'Bot Bot'}: {t['content']}"
            for t in batches[0]
        )
        messages = [
            {"role": "system", "content": SUMMARY_INSTRUCTION},
            {"role": "user", "content": conversation_text}
        ]
        summary = await _call_npu(messages, npu_lock)
    else:
        # Long session: fold batches incrementally using the same
        # rolling-summary approach as summarize_turns_batch(), so no
        # single NPU call ever sees more than MAX_CHARS_PER_SUMMARY_BATCH
        # of raw conversation regardless of total session length. The
        # running `summary` itself is also size-bounded (SEGMENT_MAX_CHARS)
        # rather than left to grow across the whole loop unchecked — once
        # it crosses that cap it's frozen off as a segment and folding
        # continues fresh, mirroring memory.py's chunk-boundary logic.
        segments: list[str] = []
        summary = ""
        for batch in batches:
            if len(summary) >= SEGMENT_MAX_CHARS:
                segments.append(summary)
                summary = ""
            summary = await summarize_turns_batch(summary, batch, npu_lock)
        if summary:
            segments.append(summary)

        summary = segments[0] if len(segments) == 1 else await _consolidate_segments(segments, npu_lock)

    start_ts = turns[0].get("timestamp") if turns and isinstance(turns[0], dict) else None
    end_ts = turns[-1].get("timestamp") if turns and isinstance(turns[-1], dict) else None
    if start_ts and end_ts:
        start_date = str(start_ts)[:10]
        end_date = str(end_ts)[:10]
        date_tag = f"[{start_date}]" if start_date == end_date else f"[{start_date} to {end_date}]"
        if not summary.startswith("["):
            summary = f"{date_tag} {summary}"

    save_session_summary(session_id, summary, known_people=known_people, is_guest=is_guest)
    return summary


def _combine_segments_for_consolidation(segments: list[str], max_chars: int) -> str:
    """
    Joins summary segments into one block for the final consolidation
    call, hard-truncating with a marker if the joined text would exceed
    max_chars — same defensive-truncation style as _sanitize_turn_content(),
    applied here to whole segments instead of raw turn content. This only
    ever fires for unusually long sessions that produced more than one
    SEGMENT_MAX_CHARS-sized segment; ordinary sessions never reach it.
    """
    combined = "\n\n".join(f"Segment {i + 1}: {s}" for i, s in enumerate(segments))
    if len(combined) > max_chars:
        # Keep the TAIL: the most recent segments matter most. The old code kept
        # the head (dropping the newest content) while labelling the cut
        # "[earlier segments truncated]".
        combined = "[earlier segments truncated] ..." + combined[-max_chars:]
    return combined


async def _consolidate_segments(segments: list[str], npu_lock=None) -> str:
    """
    Folds multiple size-bounded summary segments (produced when
    summarize_session()'s multi-batch loop crossed SEGMENT_MAX_CHARS more
    than once) into the single final session summary via one additional
    NPU call. Only reached for unusually long sessions — the common case
    (a single segment) returns straight from summarize_session() without
    ever calling this.
    """
    combined = _combine_segments_for_consolidation(segments, MAX_CHARS_PER_SUMMARY_BATCH)
    messages = [
        {"role": "system", "content": CONSOLIDATE_SEGMENTS_INSTRUCTION},
        {"role": "user", "content": combined},
    ]
    return await _call_npu(messages, npu_lock)


async def _call_npu(messages: list, npu_lock=None) -> str:
    """Shared single-call helper — with or without lock protection."""
    async def _do_call() -> str:
        async with httpx.AsyncClient(timeout=NPU_CALL_TIMEOUT_SECONDS) as client:
            response = await client.post(
                NPU_SERVER_URL,
                json={
                    "model": MODEL_NAME,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_predict": 150,
                        "temperature": 0.3,
                        "top_p": 0.85,
                        "repeat_penalty": 1.05,
                    },
                }
            )
            response.raise_for_status()
            data = response.json()
            content = data["message"]["content"].strip()
            if len(content) > 500:
                truncated = content[:500]
                last_period = max(truncated.rfind(". "), truncated.rfind(".\n"))
                if last_period > 200:
                    content = truncated[:last_period + 1].strip()
                else:
                    content = truncated.rstrip() + "..."
            return content

    if npu_lock is not None:
        async with npu_lock:
            return await _do_call()
    return await _do_call()


async def summarize_turns_batch(existing_summary: str, turns: list, npu_lock) -> str:
    """
    Folds a batch of turns into an existing rolling summary and returns the
    updated summary text. Used by memory.maybe_rollup_session() for
    mid-session (as opposed to end-of-session) summarization.

    existing_summary - the rolling summary so far, or "" if this is the
                        first rollup for the session.
    turns             - list of turn dicts (same shape as get_all_turns())
                         representing the new batch to fold in.
    npu_lock          - passed in rather than imported, both to avoid a
                         circular import (memory.py -> summarizer.py ->
                         main.py) and, more importantly, because concurrent
                         calls into npu_server.py are exactly what this lock
                         exists to prevent.

    Raises on HTTP/parsing failure rather than swallowing it.
    """
    if not turns:
        return existing_summary

    turns = [t for t in turns if not (t.get("role") == "assistant" and _is_corrupted_or_glitched_turn(t.get("content", "")))]
    if not turns:
        return existing_summary

    # Safety net, not the normal path: maybe_rollup_session() should
    # always pass a bounded batch (ROLLUP_TRIGGER_TURNS turns), but if a
    # future caller or config change ever passes something larger than
    # this file's per-call budget, fold it in sub-batches internally
    # rather than risk repeating the exact failure that hit
    # summarize_session() (a single oversized NPU call).
    sub_batches = _chunk_turns_by_chars(turns, MAX_CHARS_PER_SUMMARY_BATCH)

    summary = existing_summary
    for sub_batch in sub_batches:
        conversation_text = "\n".join(
            f"{'User' if t['role'] == 'user' else 'Bot Bot'}: {t['content']}"
            for t in sub_batch
        )

        existing_block = (
            f"Existing summary so far:\n{summary}\n\n"
            if summary
            else "Existing summary so far: (none yet, this is the first rollup)\n\n"
        )

        messages = [
            {"role": "system", "content": ROLLUP_SUMMARY_INSTRUCTION},
            {
                "role": "user",
                "content": f"{existing_block}New turns to fold in:\n{conversation_text}",
            },
        ]

        summary = await _call_npu(messages, npu_lock)

    return summary