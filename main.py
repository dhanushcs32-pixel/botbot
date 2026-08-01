"""
Bot Bot — FastAPI Backend
Connects to the local OpenVINO NPU inference server (npu_server.py) running
the 'botbot' model. No Ollama anywhere in this stack.
"""

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, Depends, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from collections import defaultdict
import asyncio
import time
import re
import logging
from logging.handlers import RotatingFileHandler
import httpx

from mode_router import detect_mode, get_system_prompt, get_retrieval_policy
from memory import save_turn, get_recent_history, get_recent_summaries, get_all_session_ids, get_summary_for_session, get_important_links, maybe_rollup_session, is_test_session
from rag import search_personal_data, get_style_profile, keyword_search, search_reference_data, is_trivial_message
from semantic_style_check import check_semantic_violations
from summarizer import summarize_session, summarize_turns_batch
from media import process_uploaded_file, decode_app_attachment
from secret_key import API_KEY
from followup import check_and_send_followups

# --- Persistent debug logging --------------------------------------------
# The [DEBUG prompt sizes] and [style_check] lines used to only go to
# stdout/PowerShell scrollback — fine until the terminal closes or scrolls
# past its buffer, at which point that diagnostic history is just gone
# (this happened — see PROJECT_STATE.txt). Mirrors the same messages to a
# rotating file on disk instead, so past sessions stay inspectable.
# 5 x 5MB files (~2-3 weeks of typical debug volume) before oldest rotates
# out; adjust maxBytes/backupCount if that's not enough runway.
debug_logger = logging.getLogger("botbot.debug")
debug_logger.setLevel(logging.INFO)
_file_handler = RotatingFileHandler(
    "botbot_debug.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
debug_logger.addHandler(_file_handler)
debug_logger.propagate = False  # avoid double-printing via root logger


def debug_log(msg: str):
    """Prints to console (unchanged behavior) AND persists to
    botbot_debug.log so terminal-buffer loss can't erase it again."""
    print(msg)
    debug_logger.info(msg)

# --- Style safety net ---------------------------------------------------
# get_system_prompt()/my_style_template.txt already instruct the model not
# to use emojis or AI-tell stock phrases, and that instruction is placed
# last in the prompt (see mode_router.py) for maximum positional weight on
# this small quantized model. This filter is a backstop, not the primary
# fix — it guarantees the two most visible symptoms (emojis, stock AI
# phrasing) never reach the user even on the turns where the model doesn't
# fully follow the prompt instruction.
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols & pictographs, supplemental symbols
    "\U00002600-\U000027BF"  # misc symbols, dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicator (flags)
    "\U00002190-\U000021FF"  # arrows (catches some emoji-adjacent symbols)
    "]+",
    flags=re.UNICODE,
)

AI_TELL_PHRASES = [
    "i understand", "great question", "certainly!", "i'd be happy to",
    "feel free to", "let me know if you have any other questions",
    "i hope this helps", "as an ai", "it's important to note that",
    "sure thing!", "got it!", "no problem at all", "you're welcome!",
    "don't hesitate to reach out", "totally okay", "totally normal",
    "good luck with", "remember, it's okay",
]

# Option 2 (validate-and-retry): which violations are worth paying for a
# full NPU regeneration versus which are already safely handled by
# scrub_reply()'s deterministic regex fixes.
#
# emoji / ai_tell_phrase: scrub_reply already removes these cleanly and
# safely (a stray emoji or a stock phrase can just be deleted without
# damaging the surrounding sentence) — not worth a ~25-100s+ regeneration.
#
# bullet_list / header / bold_pseudo_header / template_divider: these are
# STRUCTURAL — the underlying answer is shaped like a list/template, and
# stripping the markdown characters (scrub_reply's current fix) leaves the
# same list-shaped structure behind with the formatting just uglier, not
# actually fixed. Confirmed via real logs (PROJECT_STATE.txt) that these
# specifically correlate with advice-style prompts ("how do i", "tips on")
# regardless of prompt size — a genuine instruction-following gap, which a
# regenerate-with-correction pass can actually fix, unlike post-hoc regex.
#
# bracket_placeholder / salutation_closing intentionally excluded — rare,
# log_only edge cases where a regex fix already carries lower risk and a
# full retry isn't clearly justified yet.
RETRY_TRIGGER_VIOLATIONS = {"bullet_list", "header", "bold_pseudo_header", "template_divider"}

# Hard cap on regenerations per reply. 1 retry = at most 2 total NPU calls
# for a violating turn. Kept low deliberately: each retry costs a full
# generation (observed 22-112s on this hardware), and retry isn't
# guaranteed to succeed — after MAX_STYLE_RETRIES failed attempts, fall
# back to scrub_reply()'s regex-cleaned version of the last attempt rather
# than retrying indefinitely.
MAX_STYLE_RETRIES = 1

# --- Structural violation rule table --------------------------------------
# Single source of truth for both detection (check_style_violations) and
# fixing (scrub_reply) — replaces separate hardcoded patterns that
# previously had to be edited in two places and could drift out of sync
# (this happened once: BOLD_LABEL_PATTERN existed in detection before its
# matching fix line was added). Add a rule here and it's automatically
# detected AND handled per its action, no other code changes needed.
#
# action values:
#   "strip"       - remove the matched text entirely (e.g. a bullet marker)
#   "replace"     - re.sub with `repl` (e.g. "**Label:**" -> "Label: ")
#   "remove_line" - delete the whole matched line (e.g. a "---" divider —
#                   pure noise, no content to preserve)
#   "log_only"    - flagged in violations/logs but NOT auto-fixed, where a
#                   safe deterministic fix risks mangled/nonsensical text
#                   (e.g. stripping "[Recipient's Name]" out of "Dear
#                   [Recipient's Name]," would leave "Dear ," behind).
STRUCTURE_RULES = [
    {
        "name": "bullet_list",
        "pattern": re.compile(r"^\s{0,3}(?:[-*]\s+|\d+\.\s+)", re.MULTILINE),
        "action": "strip",
    },
    {
        "name": "header",
        "pattern": re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE),
        "action": "strip",
    },
    {
        "name": "bold_pseudo_header",
        # bold markdown used as a label, e.g. "**Research Thoroughly:**
        # Make sure..." — model reaches for this on "tips"-style advice
        # even when told not to use bullets/headers, since it's neither.
        # Trailing `:?\s*` handles both "**Label:**" and "**Label**:"
        # forms so neither leaves a stray/doubled colon behind.
        "pattern": re.compile(r"\*{1,2}([^*\n]{2,40}?):?\*{1,2}\s*:?\s*"),
        "action": "replace",
        "repl": r"\1: ",
    },
    {
        "name": "template_divider",
        # a line that's ONLY dashes/equals/underscores/asterisks (3+) —
        # markdown horizontal rule / doc-template section break. Pure
        # formatting noise, safe to delete outright.
        "pattern": re.compile(r"^[ \t]*[-=*_]{3,}[ \t]*\n?", re.MULTILINE),
        "action": "remove_line",
    },
    {
        "name": "bracket_placeholder",
        # unfilled template slots like "[Your Name]", "[Recipient]" — a
        # strong signal the model has slipped into generic-template/
        # ghostwriting mode rather than actually texting as itself.
        "pattern": re.compile(r"\[[A-Z][a-zA-Z' \-]{1,30}\]"),
        "action": "log_only",
    },
    {
        "name": "salutation_closing",
        # letter/email-style sign-offs: "Best regards,", "Sincerely,",
        # etc., optionally followed by a bracketed or bare name.
        "pattern": re.compile(
            r"(?im)^\s*(best regards|best|sincerely|warm(?:ly)? regards|"
            r"kind regards|regards|cheers)\s*,\s*$"
        ),
        "action": "log_only",
    },
]


def check_style_violations(text: str) -> list[str]:
    """
    Checkpoint run on the raw model output before any fixing happens.
    Returns a list of violation labels found, purely for logging/
    visibility — this is what lets you see in the server logs whether
    the model is actually drifting from style over time, rather than
    silently trusting that scrub_reply() catching things means
    everything's fine upstream. Structural/template-harness rules are
    driven by STRUCTURE_RULES; emoji and AI-tell phrases stay as their
    own explicit checks since they use different matching logic
    (character-class scan vs. substring list).
    """
    violations = []
    if EMOJI_PATTERN.search(text):
        violations.append("emoji")
    lowered = text.lower()
    if any(p in lowered for p in AI_TELL_PHRASES):
        violations.append("ai_tell_phrase")
    for rule in STRUCTURE_RULES:
        if rule["pattern"].search(text):
            violations.append(rule["name"])
    return violations


def scrub_reply(text: str, session_id: str = "") -> str:
    """
    Style enforcement checkpoint run on every reply right before it's
    saved to history or sent to the client.
      1. check_style_violations() flags what's wrong (logged).
      2. Deterministic fixes applied per STRUCTURE_RULES action
         (strip/replace/remove_line auto-fixed; log_only left as-is).
      3. Tier 2 semantic style check (log-only) — catches AI-esque
         paraphrases exact-match AI_TELL_PHRASES can't, via embedding
         similarity. See semantic_style_check.py for full rationale.
         Does NOT alter the returned text, findings are logged only,
         pending threshold tuning against real traffic.
    Anything NOT fixable this way (e.g. a reply that's grammatically
    correct/formal throughout with no single fixable trigger) won't be
    caught here — that's a prompt/model-side problem, not something a
    post-hoc filter can safely rewrite without risking mangled text.
    """
    if not text:
        return text

    violations = check_style_violations(text)
    if violations:
        print(f"[style_check] session '{session_id}': violations found — {violations}")
        debug_logger.info(f"[style_check] session '{session_id}': violations found — {violations}")

    text = EMOJI_PATTERN.sub("", text)

    lowered = text.lower()
    for phrase in AI_TELL_PHRASES:
        if phrase in lowered:
            # [^.\n!?] (not just [^.!?]) — critical for this style profile:
            # texting-voice replies are often fragmented lines with no
            # period/exclamation/question mark, so without the \n boundary
            # too, one flagged phrase could greedily eat everything back to
            # the start of the whole reply (confirmed by testing — a phrase
            # on line 3 wiped lines 1-2 entirely). Newline is now also a
            # hard boundary, so removal stays confined to the line/sentence
            # the phrase actually appeared in.
            text = re.sub(
                rf"[^.\n!?]*{re.escape(phrase)}[^.\n!?]*[.!?\n]?",
                "",
                text,
                flags=re.IGNORECASE,
            )
            lowered = text.lower()

    for rule in STRUCTURE_RULES:
        if rule["action"] == "strip" or rule["action"] == "remove_line":
            text = rule["pattern"].sub("", text)
        elif rule["action"] == "replace":
            text = rule["pattern"].sub(rule["repl"], text)
        # "log_only" rules intentionally do nothing here — already
        # recorded in the violations log above.

    text = re.sub(r"[ \t]{2,}", " ", text).strip()

    # --- Tier 2: semantic style check (log-only) --------------------
    # Runs AFTER the deterministic fixes above, so it's scoring the
    # already-cleaned text. Deliberately does NOT alter `text` — see
    # module docstring in semantic_style_check.py and PROJECT_STATE.txt
    # for why this stays log-only until thresholds are tuned against
    # real traffic.
    try:
        semantic_findings = check_semantic_violations(text, session_id=session_id)
        if semantic_findings:
            summary = [f"{f['category']}:{f['score']}" for f in semantic_findings]
            print(f"[semantic_style_check] session '{session_id}': {summary}")
            debug_logger.info(
                f"[semantic_style_check] session '{session_id}': {semantic_findings}"
            )
    except Exception as e:
        # Never let a diagnostic-only feature break the actual reply path.
        print(f"[semantic_style_check] error, skipping: {e}")

    return text

app = FastAPI(title="Bot Bot Backend")

# Allows the browser-based test UI (chat.html, opened via file://) and the
# future Android app to call this API. Safe here since this server only
# ever runs on the local network, never exposed to the internet.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NPU_SERVER_URL = "http://127.0.0.1:11435/api/chat"

# Option 2 correction addendum, appended to the system prompt ONLY on a
# retry attempt (never saved, never part of the normal prompt). Kept short
# and deliberately does NOT quote the previous bad reply back — including
# the full violating text would grow the prompt by up to ~num_predict
# tokens right when we're already near the trim budget on some turns, and
# risks the model anchoring on / partially copying its own bad structure.
# Naming the violation types by label is enough context for the model to
# self-correct without re-reading what it did wrong.
def _build_retry_addendum(violations: list[str]) -> str:
    readable = {
        "bullet_list": "bullet points or numbered lists",
        "header": "markdown headers (#, ##, etc.)",
        "bold_pseudo_header": "bold text used as a label/header",
        "template_divider": "horizontal rule / section divider lines",
    }
    named = [readable.get(v, v) for v in violations if v in readable]
    violation_text = ", ".join(named) if named else "forbidden formatting"
    return (
        f"\n\nCRITICAL — your last answer used {violation_text}. Do not do "
        "that. Answer the same question again, but write it as plain, "
        "natural conversational prose — like texting a close friend, not "
        "writing a guide or document. No lists, no headers, no bold labels, "
        "no dividers. Just talk normally, in flowing sentences.\n"
    )


async def _call_npu_once(messages: list) -> str:
    """
    Single NPU generation call — shared by the first attempt and any
    style-violation retries in generate_reply(), so the payload shape /
    options / error handling only exist in one place. Returns the raw
    (unscrubbed) reply text. Raises HTTPException(503) on backend failure,
    same behavior as the original inline call this replaces.
    """
    try:
        async with npu_lock:
            async with httpx.AsyncClient(timeout=240) as client:
                response = await client.post(
                    NPU_SERVER_URL,
                    json={
                        "model": MODEL_NAME,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "num_predict": 300,
                            "temperature": 0.85,
                            "top_p": 0.9,
                            "repeat_penalty": 1.15
                        }
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"]
    except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.TransportError) as e:
        print(f"[generate_reply] npu_server call failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="The model backend is temporarily unavailable or overloaded. Please try again in a moment."
        )
MODEL_NAME = "botbot"

# Serializes every npu_server.py call across the whole app — the live
# /chat path AND the background proactive follow-up loop (followup.py)
# both go through this. Root cause confirmed: this machine has genuinely
# tight RAM (~15.4GB) for an 8B model + full stack, and concurrent
# npu_server.py requests (e.g. a live chat landing at the same moment the
# follow-up loop is generating for several sessions) were pushing
# memory pressure past the point of a crash.
#
# A plain asyncio.Lock() was tried first (one generation at a time)
# but crashes persisted even fully serialized — back-to-back calls
# were landing before the previous call's memory was actually released
# back to the OS. NPUThrottle below adds an enforced cooldown on
# top of the lock: after a call finishes, the NEXT caller must wait at
# least NPU_MIN_GAP_SECONDS before it's allowed to start, giving
# Windows time to actually reclaim memory. Used identically to a plain
# lock (`async with npu_lock:`), so no other call site needs to
# change.
NPU_MIN_GAP_SECONDS = 5.0


class NPUThrottle:
    def __init__(self, min_gap_seconds: float = NPU_MIN_GAP_SECONDS):
        self._lock = asyncio.Lock()
        self._min_gap = min_gap_seconds
        self._last_release: float | None = None

    async def __aenter__(self):
        await self._lock.acquire()
        if self._last_release is not None:
            elapsed = time.monotonic() - self._last_release
            remaining = self._min_gap - elapsed
            if remaining > 0:
                print(f"[npu_throttle] cooling down {remaining:.1f}s before next call")
                await asyncio.sleep(remaining)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._last_release = time.monotonic()
        self._lock.release()
        return False


npu_lock = NPUThrottle()

# Live WebSocket connections, keyed by session_id, and messages queued
# for delivery to a session that isn't currently connected.
active_connections: Dict[str, WebSocket] = {}
pending_messages: Dict[str, List[str]] = defaultdict(list)

FOLLOWUP_CHECK_INTERVAL_SECONDS = 30 * 60  # check for follow-ups every 30 minutes


async def push_message_to_session(session_id: str, message: str):
    """
    Delivers a proactive message to a session — live over its
    WebSocket if currently connected, otherwise queued for delivery
    the next time that session connects.
    """
    ws = active_connections.get(session_id)
    if ws:
        try:
            await ws.send_json({"type": "message", "reply": message, "mode": "proactive"})
            return
        except Exception:
            pass  # connection may have just dropped — fall through to queueing
    pending_messages[session_id].append(message)


async def _followup_loop():
    """Background task: periodically checks whether any session warrants a proactive follow-up."""
    while True:
        await asyncio.sleep(FOLLOWUP_CHECK_INTERVAL_SECONDS)
        try:
            await check_and_send_followups(push_message_to_session, npu_lock)
        except Exception as e:
            print(f"[followup loop] error: {e}")


@app.on_event("startup")
async def start_followup_loop():
    asyncio.create_task(_followup_loop())


def detect_mentioned_session(message: str, known_sessions: list[str], current_session_id: str) -> str | None:
    """
    Checks if the user's message explicitly names a past session by its
    session_id (e.g. "what did we talk about in therapytest"). Skips the
    current session itself (nothing to "look up" there — it's just the
    live conversation), and requires a reasonably specific id (4+ chars)
    to avoid accidental matches on very short/common ids.
    """
    msg = message.lower()
    for sid in known_sessions:
        if sid == current_session_id:
            continue
        if len(sid) >= 4 and sid.lower() in msg:
            return sid
    return None


async def _background_rollup(session_id: str):
    """
    Fire-and-forget wrapper around memory.maybe_rollup_session(), scheduled
    via asyncio.create_task() right after save_turn() in generate_reply() —
    same pattern already used for the follow-up loop. Runs after the reply
    has already been returned to the caller, so a slow or failed rollup
    never delays or breaks the live /chat response.

    maybe_rollup_session() itself is a cheap no-op unless the session has
    crossed ROLLUP_TRIGGER_TURNS un-folded turns (see memory.py), so most
    calls here return immediately without touching npu_server.py at all.
    Uses the same module-scope npu_lock as the live /chat path and the
    follow-up loop, so a rollup call never races a live generation.
    """
    try:
        await maybe_rollup_session(session_id, summarize_turns_batch, npu_lock)
    except Exception as e:
        print(f"[background_rollup] error for session '{session_id}': {e}")


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    # Matches the Android app's payload shape: a list of dicts with at
    # least "name", "type", and "base64" fields. Optional — chat.html
    # and plain PowerShell requests won't send this at all.
    attachments: Optional[List[Dict[str, Any]]] = None


class ChatResponse(BaseModel):
    reply: str
    mode: str


class SummarizeRequest(BaseModel):
    session_id: str
    force: bool = False


class SearchRequest(BaseModel):
    query: str
    max_results: int = 15
    context_lines: int = 8


async def verify_api_key(x_api_key: str = Header(None)):
    """
    Guards every sensitive endpoint. The caller must send the shared
    secret in an "X-API-Key" header. This keeps the API usable only by
    devices that actually have the key (you) — important since this
    server is reachable over Tailscale/LAN and has no other auth layer.
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/")
async def root():
    return {"status": "Bot Bot backend is running"}


@app.get("/api/health")
async def health():
    """
    Health check endpoint used by the Android app's 'Test Connection'
    feature. Keep this lightweight — no DB or model calls, just confirms
    the server process is alive and reachable.
    """
    return {"status": "ok"}


async def generate_reply(session_id: str, prompt_message: str, history_message: str = None) -> tuple[str, str]:
    """
    Core reply pipeline — mode detection, RAG, history, style, and the
    npu_server.py call. Shared by both /chat (plain text) and /chat/upload
    (file/image attachments), so both paths behave identically.

    prompt_message   - the full text actually sent to the model this
                        turn (may include extracted file/image content)
    history_message  - what gets SAVED to conversation history instead,
                        if different from prompt_message. Keeps large
                        extracted file/OCR text out of future prompts —
                        otherwise every subsequent message in the
                        session would keep re-paying for that content
                        via get_recent_history(). Defaults to
                        prompt_message when not given (plain text case).
    """
    if history_message is None:
        history_message = prompt_message

    # Part 4 — behavioral mode detection
    mode = detect_mode(prompt_message)

    # Strict, explicit retrieval partitioning — see RETRIEVAL_POLICY in
    # mode_router.py for the full reasoning. Each mode states exactly
    # what it's allowed to retrieve and how much; nothing here silently
    # cascades into a more expensive path just because a keyword matched.
    policy = get_retrieval_policy(mode)

    # Part 6/7 — retrieve relevant personal context via RAG
    # Skip entirely for trivial messages (hi, ok, lol) — no useful
    # context to find, and saves real time on the most common messages.
    # Even when not trivial, top_k is mode-gated: casual/conversational
    # messages get a real "bare minimum" pull (policy["personal_top_k"]
    # is small), supportive/guidance get the fuller tier — see policy
    # comments in mode_router.py for why this is safe even at the
    # fuller tier now.
    personal_context = ""
    reference_context = ""
    if not is_trivial_message(prompt_message):
        personal_context = search_personal_data(
            prompt_message,
            top_k=policy["personal_top_k"],
            max_result_chars=policy["personal_max_chars"]
        )
        if policy["reference"]:
            reference_context = search_reference_data(prompt_message, top_k=policy["reference_top_k"])

    # --- DIRECT FILE INJECTION: Important links.txt ---
    important_links = get_important_links()
    if important_links:
        if personal_context:
            personal_context = f"{personal_context}\n\n{important_links}"
        else:
            personal_context = important_links
    # ----------------------------------------------------

    # Part 8 — pull recent summaries of OTHER past sessions
    # limit is also mode-gated now (policy["summaries_limit"]) — casual
    # chat doesn't need the full ambient summary list every turn.
    #
    # Test-session isolation: if THIS session is a test session (e.g.
    # "testtest"), skip pulling other sessions' summaries entirely — a
    # test sandbox shouldn't see real conversation history at all, not
    # just avoid leaking its own content elsewhere. get_recent_summaries()
    # already excludes test sessions from the pool other (real) sessions
    # see; this is the other direction of the same isolation guarantee.
    if is_test_session(session_id):
        session_summaries_text = ""
    else:
        summaries = get_recent_summaries(exclude_session_id=session_id, limit=policy["summaries_limit"])
        session_summaries_text = "\n".join(f"- {s}" for s in summaries) if summaries else ""

    # If the user explicitly named a past session by its session_id
    # (e.g. "what did we talk about in therapytest"), fetch that
    # session's real content directly rather than relying on it to
    # show up in the general ambient summaries pool above.
    #
    # Test-session isolation: known_sessions is filtered to exclude test
    # sessions before matching, so no session — test or real — can ever
    # trigger a lookup into a test session's actual content just by
    # typing its name (e.g. "testtest") in a message. Without this,
    # detect_mentioned_session() would happily fetch and inject a test
    # session's real turns into a completely unrelated conversation,
    # which is a live leak, not just a theoretical one.
    referenced_session_context = ""
    if is_test_session(session_id):
        # Full isolation: a test session also can't look up any OTHER
        # session by name — keeps it a clean, self-contained sandbox in
        # both directions rather than only protecting its own privacy.
        mentioned_session = None
    else:
        known_sessions = [sid for sid in get_all_session_ids() if not is_test_session(sid)]
        mentioned_session = detect_mentioned_session(prompt_message, known_sessions, session_id)
    if mentioned_session:
        summary = get_summary_for_session(mentioned_session)
        if summary:
            referenced_session_context = f"Session '{mentioned_session}': {summary}"
        else:
            turns = get_recent_history(mentioned_session, limit=10)
            if turns:
                turns_text = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
                referenced_session_context = (
                    f"Session '{mentioned_session}' (not yet summarized, recent turns):\n{turns_text}"
                )

    # Part 13 — user's style profile, always injected in full
    style_profile = get_style_profile()

    system_prompt = get_system_prompt(
        mode,
        personal_context=personal_context,
        style_profile=style_profile,
        session_summaries=session_summaries_text,
        reference_context=reference_context,
        referenced_session_context=referenced_session_context
    )

    # Part 5 — pull recent conversation history for THIS session
    # No longer capped at a fixed turn count — get_recent_history() now
    # caps by token budget (max_tokens, default 2200) instead, so this
    # naturally shrinks/grows the prompt to leave headroom for the rest
    # of the system prompt (RAG chunks, summaries, style profile).
    history = get_recent_history(session_id)

    # --- TEMPORARY DEBUG LOGGING ---
    # Prints the character length of every component that goes into the
    # prompt, so we can see exactly which piece is unexpectedly large on
    # any given turn instead of guessing from npu_server.py's token count
    # alone.
    # ~4 chars/token is a rough estimate, consistent with the heuristic
    # used in memory.py's get_recent_history().
    # ~4 chars/token was the original estimate, but a real npu_server.py
    # failure confirmed this significantly UNDERCOUNTS against Qwen's
    # actual tokenizer: a prompt estimated here at ~2736 tokens was
    # actually 4519 tokens once really tokenized (~1.65x higher) —
    # likely chat-template overhead (role markers/special tokens per
    # turn) that a flat chars/4 count on raw content doesn't capture.
    # Using ~2.5 chars/token as a more honest (still approximate, but
    # far less dangerously optimistic) estimate for prompt-budget/
    # debug-logging purposes going forward.
    CHARS_PER_TOKEN_ESTIMATE = 2.5

    def _est_tokens(s: str) -> int:
        return int(len(s) / CHARS_PER_TOKEN_ESTIMATE)

    history_chars = sum(len(h["content"]) for h in history)
    debug_log(
        "[DEBUG prompt sizes] "
        f"personal_context={len(personal_context)} chars (~{_est_tokens(personal_context)} tok) | "
        f"reference_context={len(reference_context)} chars (~{_est_tokens(reference_context)} tok) | "
        f"session_summaries_text={len(session_summaries_text)} chars (~{_est_tokens(session_summaries_text)} tok) | "
        f"referenced_session_context={len(referenced_session_context)} chars (~{_est_tokens(referenced_session_context)} tok) | "
        f"style_profile={len(style_profile)} chars (~{_est_tokens(style_profile)} tok) | "
        f"history={history_chars} chars (~{int(history_chars / CHARS_PER_TOKEN_ESTIMATE)} tok, {len(history)} turns) | "
        f"system_prompt_total={len(system_prompt)} chars (~{_est_tokens(system_prompt)} tok) | "
        f"prompt_message={len(prompt_message)} chars (~{_est_tokens(prompt_message)} tok)"
    )
    # --- END TEMPORARY DEBUG LOGGING ---

    # --- HARD TRIM-TO-FIT SAFETY NET ---
    # The estimate above is just for debugging/visibility — it has now been
    # wrong twice in production, in two different directions of severity
    # (observed ~1.16x and ~1.65x undercounts vs. the real tokenizer), so it
    # cannot be trusted to gate whether a request is safe to send. This is
    # confirmed by a real crash: a prompt estimated here at ~5678 tokens was
    # actually 6590 once tokenized by npu_server.py, over its hard
    # MAX_PROMPT_LEN ceiling of 6144 — GPU fallback failed first (OpenCL
    # ProgramBuilder error on this integrated-GPU hardware), then NPU itself
    # rejected the oversized prompt outright, surfacing as an unhandled 500.
    #
    # Rather than trying to out-guess the tokenizer, this hard-caps the
    # actual character budget using a deliberately pessimistic chars/token
    # ratio (worse than the worst undercount seen so far) and physically
    # trims content until the prompt is guaranteed to fit.
    #
    # FIX 3 — reordered from the original least-to-most-damaging list.
    # The original order trimmed conversation history FIRST, via an
    # unbounded while-loop that could empty it completely — meaning the
    # thing most valuable to the user (what was actually just said) was
    # the first thing sacrificed, while static instructional padding
    # (reference/technique context, other-session summaries) and RAG
    # personal_context — which the user never sees as "missing" the way
    # they'd notice the bot forgetting the last few turns — survived
    # longer. New order, least valuable to lose first:
    #   1. reference/technique context + other-session summaries (static
    #      padding, mode-gated, least conversationally relevant)
    #   2. personal RAG context (search_personal_data results — relevant,
    #      but a re-query next turn can recover it; recent history can't
    #      be recovered once dropped)
    #   3. conversation history — LAST resort, and only down to a
    #      protected floor (HISTORY_PROTECTED_FLOOR_TURNS) rather than to
    #      zero. Only if the prompt still doesn't fit with nothing left
    #      but the floor does it dip below the floor, as a genuine last
    #      resort to avoid a hard 500.
    NPU_HARD_TOKEN_LIMIT = 6144
    HARD_CAP_SAFETY_MARGIN_TOKENS = 600
    HARD_CAP_CHARS_PER_TOKEN = 1.5  # pessimistic on purpose
    HARD_CAP_TARGET_CHARS = int(
        (NPU_HARD_TOKEN_LIMIT - HARD_CAP_SAFETY_MARGIN_TOKENS) * HARD_CAP_CHARS_PER_TOKEN
    )
    # Minimum number of history dict entries (user+assistant messages, not
    # "turns" in the save_turn sense) to preserve before history trimming
    # is allowed to proceed past this floor. 4 entries = last ~2 exchanges.
    HISTORY_PROTECTED_FLOOR_TURNS = 4

    def _total_prompt_chars(sp: str, hist: list, pm: str) -> int:
        return len(sp) + sum(len(h["content"]) for h in hist) + len(pm)

    total_chars = _total_prompt_chars(system_prompt, history, prompt_message)
    trimmed = False

    # Step 1: reference/technique context + other-session summaries first —
    # static padding, least damaging to lose. Trims reference_context down
    # to nothing first, then session_summaries_text if still over budget.
    if total_chars > HARD_CAP_TARGET_CHARS and (reference_context or session_summaries_text):
        overflow = total_chars - HARD_CAP_TARGET_CHARS
        if reference_context:
            keep = max(0, len(reference_context) - overflow)
            overflow -= (len(reference_context) - keep)
            reference_context = reference_context[:keep]
        if overflow > 0 and session_summaries_text:
            session_summaries_text = session_summaries_text[: max(0, len(session_summaries_text) - overflow)]
        system_prompt = get_system_prompt(
            mode,
            personal_context=personal_context,
            style_profile=style_profile,
            session_summaries=session_summaries_text,
            reference_context=reference_context,
            referenced_session_context=referenced_session_context
        )
        trimmed = True
        total_chars = _total_prompt_chars(system_prompt, history, prompt_message)

    # Step 2: still too big — truncate personal RAG context next.
    if total_chars > HARD_CAP_TARGET_CHARS and personal_context:
        overflow = total_chars - HARD_CAP_TARGET_CHARS
        personal_context = personal_context[: max(0, len(personal_context) - overflow)]
        system_prompt = get_system_prompt(
            mode,
            personal_context=personal_context,
            style_profile=style_profile,
            session_summaries=session_summaries_text,
            reference_context=reference_context,
            referenced_session_context=referenced_session_context
        )
        trimmed = True
        total_chars = _total_prompt_chars(system_prompt, history, prompt_message)

    # Step 3: last resort — drop oldest history turns, but only down to
    # HISTORY_PROTECTED_FLOOR_TURNS. Recent conversation history is the
    # most user-visible thing to lose (it reads as the bot "forgetting"
    # what was just said), so it's protected as long as possible.
    while total_chars > HARD_CAP_TARGET_CHARS and len(history) > HISTORY_PROTECTED_FLOOR_TURNS:
        history.pop(0)
        trimmed = True
        total_chars = _total_prompt_chars(system_prompt, history, prompt_message)

    # Step 4: genuine last resort — everything else is already trimmed to
    # nothing and we're still over budget even at the protected floor.
    # Better to dip below the floor here than to hard-fail the request
    # with a 500, since npu_server.py will reject an oversized prompt
    # outright rather than gracefully degrading.
    while total_chars > HARD_CAP_TARGET_CHARS and history:
        history.pop(0)
        trimmed = True
        total_chars = _total_prompt_chars(system_prompt, history, prompt_message)

    if trimmed:
        print(
            f"[prompt_trim] session '{session_id}' exceeded safety budget — "
            f"trimmed to fit. final size ~{total_chars} chars "
            f"(~{int(total_chars / HARD_CAP_CHARS_PER_TOKEN)} conservative tok est, "
            f"target was {HARD_CAP_TARGET_CHARS} chars, {len(history)} history turns remaining)"
        )
    # --- END HARD TRIM-TO-FIT SAFETY NET ---

    messages = [{"role": "system", "content": system_prompt}] + \
        history + \
        [{"role": "user", "content": prompt_message}]

    # --- Option 2: validate-and-retry loop ------------------------------
    # Generate, check the RAW (pre-scrub) reply against
    # check_style_violations(), and if it contains a STRUCTURAL violation
    # (see RETRY_TRIGGER_VIOLATIONS above scrub_reply/check_style_violations
    # near AI_TELL_PHRASES) — the underlying answer is list/template-shaped,
    # not just missing a regex-fixable phrase — regenerate with a short
    # correction appended to the system prompt. Up to MAX_STYLE_RETRIES
    # extra attempts; scrub_reply() at the end is still the final backstop
    # regardless of whether retry succeeded, same safety net as before.
    #
    # Wrapped in try/except: on this hardware, npu_server.py can
    # occasionally die mid-request under memory pressure (observed as a
    # genuine crash, not just a slow response). Rather than letting that
    # surface as an unhandled 500 with a full traceback to the caller,
    # catch it here and return a clean, friendly error — the underlying
    # memory/resource issue still needs addressing (num_ctx, keep_alive,
    # pagefile, closing other apps), but at least the app degrades
    # gracefully instead of crashing the request. _call_npu_once() raises
    # HTTPException(503) itself on backend failure, so a failure on any
    # attempt (including a retry) propagates the same clean error as
    # before rather than being silently swallowed.
    raw_reply = await _call_npu_once(messages)
    attempt = 0
    while attempt < MAX_STYLE_RETRIES:
        violations = check_style_violations(raw_reply)
        if not (set(violations) & RETRY_TRIGGER_VIOLATIONS):
            break
        attempt += 1
        print(
            f"[style_retry] session '{session_id}': attempt {attempt}/"
            f"{MAX_STYLE_RETRIES} — regenerating due to {violations}"
        )
        retry_system_prompt = system_prompt + _build_retry_addendum(violations)
        retry_messages = [{"role": "system", "content": retry_system_prompt}] + \
            history + \
            [{"role": "user", "content": prompt_message}]
        raw_reply = await _call_npu_once(retry_messages)
    else:
        # Loop exhausted MAX_STYLE_RETRIES without ever breaking clean —
        # log the final outcome so it's visible whether retry actually
        # helped or just cost time. (No action needed here; scrub_reply()
        # below still runs as the regex-based backstop either way.)
        final_violations = check_style_violations(raw_reply)
        if set(final_violations) & RETRY_TRIGGER_VIOLATIONS:
            print(
                f"[style_retry] session '{session_id}': still violating "
                f"after {MAX_STYLE_RETRIES} retr{'y' if MAX_STYLE_RETRIES == 1 else 'ies'} "
                f"({final_violations}) — falling back to scrub_reply()"
            )

    bot_reply = scrub_reply(raw_reply, session_id=session_id)
    # --- END validate-and-retry loop ------------------------------------

    # Part 5 — save this turn to memory (using history_message, which
    # may be a shorter stand-in for large extracted file/image content)
    save_turn(session_id, history_message, bot_reply, mode)

    # Rolling-summarization rollup — fire-and-forget, doesn't block the
    # reply. Cheap no-op unless this session just crossed
    # ROLLUP_TRIGGER_TURNS un-folded turns (see memory.py). This is what
    # main.py was missing before: memory.py/summarizer.py's rollup
    # machinery was fully implemented and unit-tested, but never actually
    # called from the live /chat path until now.
    asyncio.create_task(_background_rollup(session_id))

    return bot_reply, mode


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat(payload: ChatRequest):
    prompt_message = payload.message
    history_message = payload.message

    if payload.attachments:
        extracted_parts = [decode_app_attachment(att) for att in payload.attachments]
        file_names = [att.get("name", "file") for att in payload.attachments]

        extracted_text = "\n\n".join(extracted_parts)
        prompt_message = f"{payload.message}\n\n{extracted_text}".strip()

        # Keep what's saved to history short, same reasoning as
        # /chat/upload — don't re-pay for large OCR/caption text on
        # every future turn in this session via get_recent_history().
        history_message = f"[Sent file(s): {', '.join(file_names)}] {payload.message}".strip()

    bot_reply, mode = await generate_reply(payload.session_id, prompt_message, history_message=history_message)
    return ChatResponse(reply=bot_reply, mode=mode)


@app.post("/chat/upload", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat_upload(
    file: UploadFile = File(...),
    message: str = Form(""),
    session_id: str = Form("default")
):
    """
    Accepts a file (image, PDF, DOCX, or TXT) alongside an optional
    text message. The file is converted to plain text via media.py
    (OCR/captioning for images, direct extraction for documents),
    combined with any typed message, and sent through the exact same
    pipeline as /chat — mode detection, RAG, history, npu_server.py.
    """
    file_bytes = await file.read()
    extracted = process_uploaded_file(file.filename, file_bytes)

    if message.strip():
        prompt_message = f"{message.strip()}\n\n{extracted}"
    else:
        prompt_message = extracted

    # Keep what's stored in history short — just a reference to the
    # file plus any typed message, not the full extracted text.
    history_message = f"[Sent a file: {file.filename}]"
    if message.strip():
        history_message += f" {message.strip()}"

    bot_reply, mode = await generate_reply(session_id, prompt_message, history_message=history_message)
    return ChatResponse(reply=bot_reply, mode=mode)


@app.get("/history/{session_id}", dependencies=[Depends(verify_api_key)])
async def history(session_id: str, limit: int = 50):
    """
    Returns past turns for a session so the frontend (chat.html) can
    repopulate the chat window on page load/refresh. Reuses the same
    get_recent_history() function already used to build prompts, just
    exposed here for the UI to call directly.
    """
    turns = get_recent_history(session_id, limit=limit)
    return {"session_id": session_id, "messages": turns}


@app.post("/summarize", dependencies=[Depends(verify_api_key)])
async def summarize(payload: SummarizeRequest):
    """
    Part 8 — Call this when a session ends (e.g. app closed, or user
    starts a new session) to generate and store a summary of it.
    Safe to call multiple times — skips if already summarized unless
    triggered again with force (not exposed here, use summarizer.py
    directly with force=True if a re-summary is ever needed).
    """
    summary = await summarize_session(payload.session_id, force=payload.force, npu_lock=npu_lock)
    if summary is None:
        return {"status": "skipped", "reason": "no turns found or already summarized"}
    return {"status": "summarized", "summary": summary}


@app.post("/search", dependencies=[Depends(verify_api_key)])
async def search(payload: SearchRequest):
    """
    Keyword/phrase search across all indexed personal data (raw
    literal matching, not semantic). Returns raw matches with
    surrounding context for the Android app to display directly —
    does NOT go through the LLM, so it's fast and exact.
    """
    results = keyword_search(payload.query, context_lines=payload.context_lines, max_results=payload.max_results)
    if not results:
        return {"status": "no_matches", "results": []}

    matches = results.split("\n---\n")
    return {"status": "ok", "count": len(matches), "results": matches}


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket, session_id: str = Query("default"), api_key: str = Query(None)):
    """
    Keeps a live connection open per session so the bot can push
    proactive follow-up messages (see followup.py) without the app
    needing to poll. Also handles the app's existing ping/pong
    heartbeat. Auth is via ?api_key= since browsers/WebViews don't
    support custom headers on WebSocket connections — not accepting
    the connection here (rather than accepting then closing) makes
    Starlette reject it with a 403, same behavior as before this
    endpoint existed, for an invalid or missing key.
    """
    if api_key != API_KEY:
        return

    await websocket.accept()
    active_connections[session_id] = websocket
    print(f"[ws] session '{session_id}' connected")

    # Deliver anything that was queued while this session was offline
    if pending_messages[session_id]:
        for msg in pending_messages[session_id]:
            await websocket.send_json({"type": "message", "reply": msg, "mode": "proactive"})
        pending_messages[session_id].clear()

    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        print(f"[ws] session '{session_id}' disconnected")
    finally:
        if active_connections.get(session_id) is websocket:
            del active_connections[session_id]