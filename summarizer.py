"""
Bot Bot — Session Summarizer (Part 8)

Summarizes a completed conversation session into 3-5 sentences using
the local OpenVINO NPU model (via npu_server.py), and saves it via
memory.py. These summaries let Bot Bot recall things from past sessions
beyond just the last 10 messages of the current one.

No Ollama anywhere in this stack — everything routes through
npu_server.py, which is the OpenVINO NPU inference server.
"""

import httpx

from memory import get_all_turns, save_session_summary, session_has_summary

NPU_SERVER_URL = "http://localhost:11435/api/chat"
MODEL_NAME = "botbot"

SUMMARY_INSTRUCTION = """
Summarize the following conversation in 3-5 sentences. Focus on key
topics discussed, emotions expressed, goals or plans mentioned, and
anything that would be useful to remember for a future conversation.
Write the summary in third person, factually, as private notes for
future reference — not as a message to the user. Do not include
greetings or filler, just the substance.
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


def _chunk_turns_by_chars(turns: list, max_chars: int) -> list[list]:
    """
    Splits a flat list of turn dicts into consecutive chunks, each kept
    under max_chars of rendered "Role: content" text. Never splits a
    single turn across chunks; a lone turn larger than max_chars still
    gets its own chunk rather than being dropped or truncated, since
    losing content silently would defeat the point of summarizing it.
    """
    chunks = []
    current: list = []
    current_len = 0
    for t in turns:
        line_len = len(t["content"]) + 20  # rough allowance for "Role: " prefix + newline
        if current and current_len + line_len > max_chars:
            chunks.append(current)
            current = []
            current_len = 0
        current.append(t)
        current_len += line_len
    if current:
        chunks.append(current)
    return chunks


async def summarize_session(session_id: str, force: bool = False, npu_lock=None) -> str | None:
    """
    Generates and saves a summary for the given session by sending its
    turn history to npu_server.py in size-capped batches, folded together
    incrementally via the same rollup logic used by summarize_turns_batch()
    — rather than concatenating the entire session into a single prompt,
    which has no upper bound and can (and did) exceed the NPU's hard
    MAX_PROMPT_LEN=6144 ceiling on longer sessions. Skips sessions that
    already have a summary unless force=True. Returns the summary text, or
    None if there was nothing to summarize.

    npu_lock - optional; when provided, serializes every NPU call this
               function makes against other callers (main.py's live /chat
               path, the follow-up loop, and summarize_turns_batch()) via
               the same NPUThrottle instance, same reasoning as
               summarize_turns_batch()'s existing npu_lock parameter. If
               not provided, falls back to no locking (previous behavior)
               rather than raising, so existing callers that don't pass it
               don't break.
    """
    if not force and session_has_summary(session_id):
        return None

    turns = get_all_turns(session_id)
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
        # of raw conversation regardless of total session length.
        summary = ""
        for batch in batches:
            summary = await summarize_turns_batch(summary, batch, npu_lock)

    save_session_summary(session_id, summary)
    return summary


async def _call_npu(messages: list, npu_lock=None) -> str:
    """Shared single-call helper — with or without lock protection."""
    async def _do_call() -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                NPU_SERVER_URL,
                json={
                    "model": MODEL_NAME,
                    "messages": messages,
                    "stream": False
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"].strip()

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