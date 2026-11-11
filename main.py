"""
Bot Bot — FastAPI Backend
Connects to the local OpenVINO NPU inference server (npu_server.py) running
the 'botbot' model. No Ollama anywhere in this stack.
"""

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, Depends, WebSocket, WebSocketDisconnect, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from collections import defaultdict, deque
import asyncio
import time
import re
import os
import secrets
import json
from datetime import datetime, timezone
import logging
from logging.handlers import RotatingFileHandler
import httpx

# Offline-first for HuggingFace loads: MiniLM (rag.py embeddings) and BLIP
# (media.py captions) are cached locally, but their loaders still ping
# huggingface.co at load time. On DNS failure that poisoned the HTTP client
# and retrieval failed open silently (see PROJECT_STATE / the hybrid-search
# test that hit this). Forcing offline mode makes both load from cache with
# zero network dependency at startup.
# NOTE for a FRESH machine (model not yet cached): this blocks the one-time
# download — set HF_HUB_OFFLINE=0 in the environment before first launch,
# then remove it once the cache is populated.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import memory
from mode_router import detect_mode, detect_mode_detailed, get_system_prompt, get_retrieval_policy
from memory import (
    save_turn, get_recent_history, get_recent_summaries, get_all_session_ids,
    get_summary_for_session, get_important_links, maybe_rollup_session,
    is_test_session, session_has_summary, get_sessions_with_last_activity,
    count_turns, save_bot_initiated_message, get_current_session_summary,
    get_full_session_history
)
from rag import search_personal_data, get_style_profile, keyword_search, search_reference_data, is_trivial_message, should_bypass_personal_retrieval, get_known_people
from laya_engine import assess_context_sufficiency, LAYA_ENABLED, is_laya_available
from semantic_style_check import strip_semantic_violations
from summarizer import summarize_session, summarize_turns_batch
from media import process_uploaded_file, decode_app_attachment
from secret_key import API_KEYS
from followup import check_and_send_followups

# --- Persistent debug logging --------------------------------------------
debug_logger = logging.getLogger("botbot.debug")
debug_logger.setLevel(logging.INFO)
_file_handler = RotatingFileHandler(
    "botbot_debug.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
debug_logger.addHandler(_file_handler)
debug_logger.propagate = False


def debug_log(msg: str):
    """Prints to console AND persists to botbot_debug.log."""
    print(msg)
    debug_logger.info(msg)


# --- Context-sufficiency field observation ---------------------------------
sufficiency_logger = logging.getLogger("botbot.sufficiency")
sufficiency_logger.setLevel(logging.INFO)
_sufficiency_file_handler = RotatingFileHandler(
    "context_sufficiency_log.jsonl", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
_sufficiency_file_handler.setFormatter(logging.Formatter("%(message)s"))
sufficiency_logger.addHandler(_sufficiency_file_handler)
sufficiency_logger.propagate = False

UNCERTAINTY_HEDGE_PHRASES = [
    "i don't have that", "i don't have specific", "i don't have any",
    "i don't know", "i'm not sure", "i'm not certain", "not sure about that",
    "don't have that information", "don't have that detail",
    "don't have details on", "no information about", "no record of",
    "can't recall", "don't recall", "not something i have", "unsure about",
    "haven't come across", "don't actually know", "can't say for sure",
]


def _reply_reads_as_uncertain(text: str) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in UNCERTAINTY_HEDGE_PHRASES)


# --- Style safety net ---------------------------------------------------
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF"
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

RP_VERB_PREFIX_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:pauses|sighs|stretches|glances|typing slowly|softly|quietly|whispers|smiles|laughs|nods|shrugs|looks|stares)[^:\n]{0,20}:\s*"
)

STRUCTURE_RULES = [
    {
        "name": "bullet_list",
        "pattern": re.compile(r"^[ \t]{0,3}(?:[-*]\s+|\d{1,3}\.\s+)", re.MULTILINE),
        "action": "strip",
    },
    {
        "name": "header",
        "pattern": re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE),
        "action": "strip",
    },
    {
        "name": "bold_emphasis",
        "pattern": re.compile(r"\*{1,2}([^*\n]{1,80}?)\*{1,2}"),
        "action": "replace",
        "repl": r"\1",
    },
    {
        "name": "template_divider",
        "pattern": re.compile(r"^[ \t]*[-=*_]{3,}[ \t]*\n?", re.MULTILINE),
        "action": "remove_line",
    },
    {
        "name": "bracket_stage_direction",
        "pattern": re.compile(r"\[[^\]\n]{1,80}\]\s*"),
        "action": "strip",
    },
    {
        "name": "asterisk_action",
        "pattern": re.compile(r"(?i)\*[a-z\s,.'-]{1,40}\*\s*"),
        "action": "strip",
    },
    {
        "name": "colon_action_prefix",
        "pattern": RP_VERB_PREFIX_PATTERN,
        "action": "strip",
    },
    {
        "name": "bracket_placeholder",
        "pattern": re.compile(r"\[[A-Z][a-zA-Z' \-]{1,30}\]"),
        "action": "log_only",
    },
    {
        "name": "salutation_closing",
        "pattern": re.compile(
            r"(?im)^\s*(best regards|best|sincerely|warm(?:ly)? regards|"
            r"kind regards|regards|cheers)\s*,\s*$"
        ),
        "action": "log_only",
    },
    {
        "name": "persona_break",
        "pattern": re.compile(r"([a-zA-Z])\1{2,}"),
        "action": "log_only",
        "min_matches": 2,
    },
]


def collapse_vertical_stacking(text: str) -> str:
    """
    Detects pathological 1-2 word newline stacking (slam-poetry / fragmented monologue)
    e.g. 4+ consecutive lines with <= 2 words ending with punctuation (., !, ?).
    Collapses excessive newlines into normal horizontal prose.
    """
    if "\n" not in text:
        return text

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    consecutive_short_punctuated = 0
    max_run = 0
    for line in lines:
        words = line.split()
        if len(words) <= 2 and line.endswith((".", "!", "?")):
            consecutive_short_punctuated += 1
            if consecutive_short_punctuated > max_run:
                max_run = consecutive_short_punctuated
        else:
            consecutive_short_punctuated = 0

    if max_run >= 4:
        # Collapse lines into normal horizontal sentences
        collapsed = " ".join(lines)
        text = re.sub(r"[ \t]{2,}", " ", collapsed).strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > 2:
            text = " ".join(sentences[:2])
    return text


def collapse_repetitive_phrases(text: str) -> str:
    """Collapses consecutive loops of identical sentences or short phrases."""
    if not text:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= 2:
        return text
    deduped = []
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        if deduped and s_clean.lower() == deduped[-1].strip().lower():
            continue
        deduped.append(s_clean)
    return " ".join(deduped)


def check_style_violations(text: str) -> list[str]:
    violations = []
    if EMOJI_PATTERN.search(text):
        violations.append("emoji")
    lowered = text.lower()
    if any(p in lowered for p in AI_TELL_PHRASES):
        violations.append("ai_tell_phrase")
    for rule in STRUCTURE_RULES:
        min_matches = rule.get("min_matches", 1)
        if min_matches == 1:
            if rule["pattern"].search(text):
                violations.append(rule["name"])
        else:
            if len(rule["pattern"].findall(text)) >= min_matches:
                violations.append(rule["name"])
    return violations


def scrub_reply(text: str, session_id: str = "", mode: str = "", is_crisis: bool = False) -> str:
    if not text:
        return text

    raw_cleaned = text.strip()

    violations = check_style_violations(text)
    if violations:
        print(f"[style_check] session '{session_id}': violations found — {violations}")
        debug_logger.info(f"[style_check] session '{session_id}': violations found — {violations}")

    text = EMOJI_PATTERN.sub("", text)

    lowered = text.lower()
    for phrase in AI_TELL_PHRASES:
        if phrase in lowered:
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

    # Strip bracketed actions/stage directions like [glance at phone screen], [sighs], [pauses]
    text = re.sub(r"\[[^\]\n]{1,80}\]\s*", "", text)

    # Strip asterisk action markers like *sighs*, *looks around*
    text = re.sub(r"(?i)\*[a-z\s,.'-]{1,40}\*\s*", "", text)

    # Strip leading colon stage-direction prefixes using the targeted RP verb pattern
    text = RP_VERB_PREFIX_PATTERN.sub("", text)

    # Collapse pathological 1-2 word newline stacking
    text = collapse_vertical_stacking(text)

    # Collapse looped/repeating sentences
    text = collapse_repetitive_phrases(text)

    text = re.sub(r"[ \t]{2,}", " ", text).strip()

    try:
        text, semantic_findings = strip_semantic_violations(text, session_id=session_id, mode=mode)
        if semantic_findings:
            summary = [
                f"{f['category']}:{f['score']}{'*' if f['enforce'] else ''}"
                for f in semantic_findings
            ]
            print(f"[semantic_style_check] session '{session_id}': {summary} (* = stripped)")
            debug_logger.info(
                f"[semantic_style_check] session '{session_id}': {semantic_findings}"
            )
    except Exception as e:
        print(f"[semantic_style_check] error, skipping: {e}")

    # Strip out unsolicited helpline numbers / clinical hotline scripts if the model generated any
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned_sentences = [
        s for s in sentences
        if not re.search(r"\b(?:988\d*|14416|helpline|lifeline|tele-manas|hotline)\b", s, re.IGNORECASE)
    ]
    if cleaned_sentences:
        text = " ".join(cleaned_sentences).strip()
    elif not text.strip() and raw_cleaned:
        text = "I hear you, and I'm right here with you. What's on your mind?"

    # Fallback: if scrub_reply completely wiped the text, fall back to cleaned raw reply
    if not text.strip() and raw_cleaned:
        text = raw_cleaned

    return text


app = FastAPI(title="Bot Bot Backend")

app.add_middleware(
    CORSMiddleware,
    # Honest origin allowlist for browser-based clients (e.g. chat.html,
    # botbot_app running on :3000) AND the Capacitor app's WebView origins
    # (capacitor://localhost on Android, https://localhost on iOS), plus
    # Cloudflare Tunnels and LAN/Tailscale mobile connections via regex.
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "capacitor://localhost",
        "https://localhost",
        "http://localhost",
    ],
    allow_origin_regex=r"^(https://.*\.trycloudflare\.com|http://(192\.168|10\.|100\.)\d+\.\d+(:\d+)?)$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

NPU_SERVER_URL = "http://127.0.0.1:11435/api/chat"
MODEL_NAME = "botbot"
NPU_COOLDOWN_SEC = float(os.environ.get("NPU_COOLDOWN_SEC", "0.0"))
npu_lock = asyncio.Lock()
_last_npu_release: float | None = None

_background_tasks: set[asyncio.Task] = set()


def create_background_task(coro) -> asyncio.Task:
    """Creates an asyncio background task and retains a strong reference until
    it finishes, preventing early garbage collection."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def _call_npu_once(messages: list, num_predict: int = 250) -> str:
    global _last_npu_release
    # num_predict defaults to 250 — the normal chat path's value. The CoT
    # diagnostic endpoint passes a larger budget; with the default, the
    # request body is byte-for-byte what the normal path has always sent.
    # HTTP timeout scales with the output budget: 240s covers the 250-token
    # normal path (observed worst ~130s); large budgets get 900s (700
    # tokens at the observed ~1 tok/s ≈ 700s worst case).
    timeout_s = 240 if num_predict <= 300 else 900
    try:
        async with npu_lock:
            try:
                if NPU_COOLDOWN_SEC > 0 and _last_npu_release is not None:
                    elapsed = time.monotonic() - _last_npu_release
                    remaining = NPU_COOLDOWN_SEC - elapsed
                    if remaining > 0:
                        print(f"[npu_throttle] cooling down {remaining:.1f}s before next call")
                        await asyncio.sleep(remaining)
                async with httpx.AsyncClient(timeout=timeout_s) as client:
                    response = await client.post(
                        NPU_SERVER_URL,
                        json={
                            "model": MODEL_NAME,
                            "messages": messages,
                            "stream": False,
                            "options": {
                                "num_predict": num_predict,
                                "temperature": 0.45,
                                "top_p": 0.85,
                                "repeat_penalty": 1.12
                            }
                        }
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data["message"]["content"]
            finally:
                _last_npu_release = time.monotonic()
    except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.TransportError) as e:
        print(f"[generate_reply] npu_server call failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="The model backend is temporarily unavailable or overloaded. Please try again in a moment."
        )


active_connections: Dict[str, WebSocket] = {}
pending_messages: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
_session_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
PENDING_MSG_CAP = 200  # per-session cap; oldest pushed messages are dropped past this

WS_TICKET_TTL_SECONDS = 30
ws_tickets: Dict[str, Dict[str, Any]] = {}

# --- Duplicate & In-flight chat_message protection ------------------------
# Track in-flight IDs separately from completed IDs so an unacknowledged or
# failed message can be retried by the client, but concurrent duplicates are
# rejected.
PROCESSED_MSG_ID_CAP = 200
_inflight_message_ids: Dict[str, set] = defaultdict(set)
_completed_message_ids: Dict[str, deque] = defaultdict(lambda: deque(maxlen=PROCESSED_MSG_ID_CAP))
_completed_message_id_sets: Dict[str, set] = defaultdict(set)


def _is_message_duplicate_or_inflight(session_id: str, msg_id: str) -> bool:
    """Returns True if msg_id is currently being processed or has already completed."""
    if msg_id in _inflight_message_ids[session_id]:
        return True
    if msg_id in _completed_message_id_sets[session_id]:
        return True
    return False


def _mark_message_inflight(session_id: str, msg_id: str) -> None:
    _inflight_message_ids[session_id].add(msg_id)


def _mark_message_completed(session_id: str, msg_id: str) -> None:
    _inflight_message_ids[session_id].discard(msg_id)
    ids_set = _completed_message_id_sets[session_id]
    dq = _completed_message_ids[session_id]
    if len(dq) == dq.maxlen:
        oldest = dq[0]
        ids_set.discard(oldest)
    dq.append(msg_id)
    ids_set.add(msg_id)


def _mark_message_failed(session_id: str, msg_id: str) -> None:
    _inflight_message_ids[session_id].discard(msg_id)


def _push_pending(session_id: str, payload: Dict[str, Any]) -> None:
    """Append to a session's pending (offline) queue, evicting the oldest
    entries once PENDING_MSG_CAP is exceeded. Without this, a session that
    stays offline while messages keep getting pushed to it (e.g. from
    followups) grows this list unbounded."""
    queue = pending_messages[session_id]
    queue.append(payload)
    if len(queue) > PENDING_MSG_CAP:
        dropped = len(queue) - PENDING_MSG_CAP
        del queue[:dropped]
        print(f"[ws] pending_messages for session '{session_id}' exceeded cap, dropped {dropped} oldest")

def _purge_expired_ws_tickets() -> None:
    """Removes expired, never-consumed tickets from ws_tickets. Without
    this, a ticket that's issued but never used to open a socket (e.g.
    client requested a ticket, then the page closed before connecting)
    sits in ws_tickets forever — _consume_ws_ticket only ever removes an
    entry when it's actually consumed, and checks expiry after popping,
    not proactively. Same class of unbounded-growth issue pending_messages
    had before PENDING_MSG_CAP — swept lazily here, on each new ticket
    issuance, rather than via a separate background task, matching the
    lazy-cleanup style already used elsewhere in this file."""
    now = time.monotonic()
    expired = [t for t, entry in ws_tickets.items() if now > entry["expires"]]
    for t in expired:
        del ws_tickets[t]
    if expired:
        print(f"[ws] purged {len(expired)} expired, unconsumed ws_tickets")


session_guest_status: dict[str, bool] = {}


def _issue_ws_ticket(session_id: str, is_guest: bool = False) -> str:
    _purge_expired_ws_tickets()
    ticket = secrets.token_urlsafe(32)
    ws_tickets[ticket] = {
        "session_id": session_id,
        "is_guest": is_guest,
        "expires": time.monotonic() + WS_TICKET_TTL_SECONDS,
    }
    return ticket


def _consume_ws_ticket(ticket: str, session_id: str) -> tuple[bool, bool]:
    entry = ws_tickets.pop(ticket, None)
    if entry is None:
        return False, False
    if time.monotonic() > entry["expires"]:
        return False, False
    if entry["session_id"] != session_id:
        return False, False
    return True, entry.get("is_guest", False)


# Must stay well under followup.py's ACTIVE_RECENCY_THRESHOLD_SECONDS (5min)
# and ACTIVE_FIRST_DELAY_SECONDS (15min) — otherwise a session's first
# message is never noticed while still "fresh," and every session gets
# misclassified as not-active. 1 minute gives comfortable margin on both
# without polling excessively.
FOLLOWUP_CHECK_INTERVAL_SECONDS = 60

# --- Prompt budget (real-token based) --------------------------------------
# npu_server's compiled MAX_PROMPT_LEN is 6144 tokens. The old char-based
# budget assumed 1.5 chars/token (~2.7x pessimistic vs ~4 chars/token for
# English), which silently wasted most of the context window and caused
# unnecessary history trimming (see the confabulation investigation,
# PROJECT_STATE-style report 2026-08-23). The budget is now measured with
# the model's REAL tokenizer via npu_server /api/estimate_tokens
# (tokenize-only, no NPU inference), targeting 5500 tokens — headroom
# below 6144 for chat-template boundary differences, matching
# npu_server's own NPU_SAFE_PROMPT_TOKENS.
NPU_BASE_URL = "http://127.0.0.1:11435"
PROMPT_TARGET_TOKENS = 5500   # default/cap; the LIVE target is derived from
                              # npu_server's reported effective ceiling
PROMPT_CEILING_MARGIN = 644   # 6144 -> 5500 headroom policy (matches
                              # npu_server's NPU_SAFE_PROMPT_TOKENS)

_prompt_target_cache = {"ts": 0.0, "target": None}


def _get_prompt_target_tokens():
    """Live prompt target, derived from npu_server's REPORTED effective
    ceiling (/api/health) rather than an independently guessed constant —
    so if npu_server ever loads without MAX_PROMPT_LEN (the silent retry
    path) or on a fallback device, main.py budgets against the ceiling
    that actually exists. Cached 60s. Returns None when the ceiling is
    unknown (health unreachable, or loaded without MAX_PROMPT_LEN) —
    callers then fall back to the conservative char-based budget."""
    now = time.time()
    if now - _prompt_target_cache["ts"] < 60:
        return _prompt_target_cache["target"]
    try:
        with httpx.Client(timeout=3.0) as client:
            h = client.get(f"{NPU_BASE_URL}/api/health").json()
        eff = h.get("effective_max_prompt_len")
        target = (max(3000, min(PROMPT_TARGET_TOKENS, eff - PROMPT_CEILING_MARGIN))
                  if eff else None)
        _prompt_target_cache["ts"] = now
        _prompt_target_cache["target"] = target
        if target != PROMPT_TARGET_TOKENS:
            print(f"[prompt_budget] effective ceiling from npu_server: "
                  f"max_prompt_len={eff} -> target={target} tokens")
    except Exception as e:
        # Keep the previously cached target (or None) rather than hammering
        # a down server every call.
        _prompt_target_cache["ts"] = now
        print(f"[prompt_budget] health unavailable ({e}); "
              f"target={_prompt_target_cache['target']}")
    return _prompt_target_cache["target"]


def _measure_prompt_tokens_sync(messages: list):
    """Real tokenizer token count via npu_server (tokenize-only — no model
    inference). Sync HTTP: called from _prepare_prompt_sync's worker thread.
    Returns None on any failure; the budget then falls back to the old
    conservative char-based cap instead of breaking prompt assembly."""
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.post(
                f"{NPU_BASE_URL}/api/estimate_tokens",
                json={"messages": messages},
            )
            resp.raise_for_status()
            return int(resp.json()["prompt_tokens"])
    except Exception as e:
        print(f"[prompt_budget] real token count unavailable ({e}); "
              f"falling back to char-based budget")
        return None


async def push_message_to_session(session_id: str, message: str, mode: str = "proactive"):
    payload = {
        "id": secrets.token_urlsafe(8),
        "type": "message",
        "reply": message,
        "mode": mode,
        "created_at": int(time.time() * 1000)
    }
    # Retain in pending queue until client explicitly acks
    _push_pending(session_id, payload)

    # Persist proactive/followup messages into memory so history endpoint and UI retain it
    save_bot_initiated_message(session_id, message, mode=mode)

    ws = active_connections.get(session_id)
    if ws:
        try:
            await ws.send_json(payload)
            print(f"[ws] delivered payload {payload['id']} to live session '{session_id}'")
        except Exception as e:
            print(f"[ws] live send failed for session '{session_id}': {e}")


async def _followup_loop():
    while True:
        await asyncio.sleep(FOLLOWUP_CHECK_INTERVAL_SECONDS)
        try:
            await check_and_send_followups(
                push_message_to_session, active_connections, _call_npu_once, scrub_reply
            )
        except Exception as e:
            print(f"[followup loop] error: {e}")


@app.on_event("startup")
async def start_followup_loop():
    create_background_task(_followup_loop())


# --- Automatic end-of-session summarization --------------------------------
# Trace finding (2026-08-23): session_summaries had ZERO rows ever —
# summarize_session() was only reachable via the manual /summarize endpoint
# or the backfill script. Nothing ever triggered it automatically: a missing
# feature, not a broken one. This loop adds the trigger, following the same
# background-idle-scan pattern as _followup_loop: every SUMMARY_CHECK_INTERVAL
# it queues summarization for non-test sessions that (a) have been idle past
# SUMMARY_IDLE_THRESHOLD_SECONDS (session effectively over — covers both
# websocket disconnects and abandoned sessions), (b) have no summary yet,
# and (c) have enough turns to be worth summarizing. Runs via the existing
# _background_summarize task (which carries the _summarize_jobs in-flight
# guard) so it never blocks a live /chat response; NPU locking stays inside
# summarizer._call_npu — no outer npu_lock acquisition here (PROJECT_STATE
# §7A deadlock class).
SUMMARY_CHECK_INTERVAL_SECONDS = 600        # scan every 10 minutes
SUMMARY_IDLE_THRESHOLD_SECONDS = 2 * 3600   # session idle >= 2h before summarizing
SUMMARY_MIN_TURNS = 6                       # skip trivial sessions


async def _summary_loop():
    while True:
        await asyncio.sleep(SUMMARY_CHECK_INTERVAL_SECONDS)
        try:
            now = time.time()
            for sid, last_active in get_sessions_with_last_activity():
                if is_test_session(sid):
                    continue  # test sessions are excluded from summaries by design
                if now - last_active < SUMMARY_IDLE_THRESHOLD_SECONDS:
                    continue
                if sid in _summarize_jobs or session_has_summary(sid):
                    continue
                if count_turns(sid) < SUMMARY_MIN_TURNS:
                    continue
                idle_h = int((now - last_active) / 3600)
                print(f"[summary_loop] session '{sid}' idle {idle_h}h with no "
                      f"summary — queuing summarization")
                create_background_task(_background_summarize(sid, force=False))
        except Exception as e:
            print(f"[summary loop] error: {e}")


@app.on_event("startup")
async def start_summary_loop():
    create_background_task(_summary_loop())


SESSION_STOPWORDS = {
    "chat", "chats", "session", "sessions", "today", "yesterday", "tomorrow",
    "home", "work", "main", "page", "user", "admin", "null", "none", "help"
}


def detect_mentioned_session(message: str, known_sessions: list[str], current_session_id: str) -> str | None:
    if not message or not known_sessions:
        return None
    msg = message.lower()
    for sid in known_sessions:
        if sid == current_session_id:
            continue
        sid_clean = sid.strip().lower()
        if len(sid_clean) < 4 or sid_clean in SESSION_STOPWORDS:
            continue
        pattern = rf"\b{re.escape(sid_clean)}\b"
        if re.search(pattern, msg):
            return sid
    return None


async def _background_rollup(session_id: str):
    try:
        await maybe_rollup_session(session_id, summarize_turns_batch, npu_lock)
    except Exception as e:
        print(f"[background_rollup] error for session '{session_id}': {e}")


_summarize_jobs: dict[str, str] = {}


async def _background_summarize(session_id: str, force: bool, known_people: set | None = None):
    _summarize_jobs[session_id] = "in_progress"
    try:
        if known_people is None:
            known_people = await asyncio.to_thread(get_known_people)
        await summarize_session(session_id, force=force, npu_lock=npu_lock, known_people=known_people)
        _summarize_jobs.pop(session_id, None)
    except Exception as e:
        _summarize_jobs[session_id] = f"error: {e}"
        print(f"[background_summarize] error for session '{session_id}': {e}")


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    attachments: Optional[List[Dict[str, Any]]] = None


class ChatResponse(BaseModel):
    reply: str
    mode: str


class SummarizeRequest(BaseModel):
    session_id: str
    force: bool = False


class SearchRequest(BaseModel):
    query: str
    max_results: int = Field(default=15, ge=1, le=50)
    context_lines: int = Field(default=8, ge=1, le=30)


class TicketRequest(BaseModel):
    session_id: str = "default"


# --- Multi-key auth (named, revocable, session-scoped) ----------------------
# verify_api_key authenticates against secret_key.API_KEYS and RETURNS the
# matched entry so endpoints can enforce session scoping server-side.
# Generic 401 for both "unknown key" and "disabled key" — no distinction
# leaks to the client. Session mismatch is a separate 403.
#
# Immediate disable without restart: main re-checks key_state.json (next to
# this file) whenever its mtime changes — {"disabled": {"guest-1": true}}
# rejects that label on the very next request. Key MATERIAL rotation still
# requires a restart (imported once).
KEY_STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "key_state.json")
_key_state_cache = {"mtime": None, "disabled": frozenset()}


def _disabled_key_labels() -> frozenset:
    try:
        mtime = os.stat(KEY_STATE_PATH).st_mtime
    except OSError:
        if _key_state_cache["mtime"] is not None:
            _key_state_cache["mtime"] = None
            _key_state_cache["disabled"] = frozenset()
        return _key_state_cache["disabled"]
    if mtime != _key_state_cache["mtime"]:
        try:
            with open(KEY_STATE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            disabled = frozenset(
                label for label, off in data.get("disabled", {}).items() if off)
        except Exception as e:
            print(f"[auth] key_state.json unreadable, ignoring: {e}")
            disabled = _key_state_cache["disabled"]
        _key_state_cache["mtime"] = mtime
        _key_state_cache["disabled"] = disabled
    return _key_state_cache["disabled"]


async def verify_api_key(x_api_key: str = Header(None)) -> dict:
    """Authenticate the presented key against secret_key.API_KEYS and return
    the matched entry. Disabled keys are skipped before comparison, so a
    disabled key and an unknown key produce the identical generic 401."""
    disabled = _disabled_key_labels()
    matched = None
    for entry in API_KEYS:
        if not entry.get("enabled", False) or entry["label"] in disabled:
            continue
        try:
            if secrets.compare_digest((x_api_key or "").encode("utf-8"), entry["key"].encode("utf-8")):
                matched = entry
                break
        except (TypeError, UnicodeEncodeError):
            continue
    if matched is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return matched


def _enforce_session_scope(key_info: dict, session_id: str) -> None:
    """Server-enforced session scoping: a scoped (non-primary) key may only
    touch its one allowed session_id, regardless of what the client sends.
    The primary key (session_id None) is unrestricted."""
    allowed = key_info.get("session_id")
    if allowed is not None and session_id != allowed:
        raise HTTPException(status_code=403, detail="Session not permitted for this key")


@app.get("/")
async def root():
    return {"status": "Bot Bot backend is running"}


NPU_HEALTH_URL = "http://127.0.0.1:11435/api/health"


@app.get("/api/health")
async def health():
    """Aggregated stack health: this API plus the NPU backend's real model
    state (proxied with a short timeout so /api/health stays fast even when
    npu_server is down). Previously reported a static ok regardless of
    whether inference could actually serve anything."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(NPU_HEALTH_URL)
            resp.raise_for_status()
            data = resp.json()
        npu_backend = {
            "reachable": True,
            "model_loaded": bool(data.get("model_loaded")),
            "loaded_model": data.get("loaded_model"),
        }
    except Exception as e:
        npu_backend = {"reachable": False, "error": str(e)}

    return {
        "status": "ok",
        "npu_backend": npu_backend,
        "can_generate": npu_backend.get("reachable", False)
        and bool(npu_backend.get("model_loaded")),
    }


def _prepare_prompt_sync(session_id: str, prompt_message: str, measure_tokens=None,
                         history_override=None, is_guest: bool = False):
    # Fetch recent history early so Laya routing has conversational context if needed
    history = (history_override if history_override is not None
               else get_recent_history(session_id))

    # 1. Behavioral Mode & Personal Retrieval Need via Laya
    laya_routing = detect_mode_detailed(prompt_message, recent_turns=history)
    mode = laya_routing["mode"]
    is_crisis = laya_routing.get("is_crisis", False)
    policy = get_retrieval_policy(mode)

    # 2. Check personal retrieval bypass (trivial phrases, or Laya high-confidence no-context, or guest session)
    bypass_retrieval = is_guest or should_bypass_personal_retrieval(prompt_message, laya_routing=laya_routing)
    context_lookup_attempted = not bypass_retrieval

    personal_context = ""
    reference_context = ""
    if context_lookup_attempted and not is_guest:
        personal_context = search_personal_data(
            prompt_message,
            top_k=policy["personal_top_k"],
            max_result_chars=policy["personal_max_chars"]
        )
        if policy["reference"]:
            reference_context = search_reference_data(prompt_message, top_k=policy["reference_top_k"])

    session_summaries_text = ""
    referenced_session_context = ""
    user_name = "User" if is_guest else "Dhanush"

    if not is_guest:
        if not is_test_session(session_id):
            summaries = get_recent_summaries(
                exclude_session_id=session_id, limit=policy["summaries_limit"], query=prompt_message
            )
            session_summaries_text = "\n".join(f"- {s}" for s in summaries) if summaries else ""

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

    style_profile = get_style_profile()

    # 3. Assess context sufficiency via Laya typed decisions BEFORE attaching static links
    combined_rag_for_sufficiency = f"{personal_context}\n\n{reference_context}".strip()
    sufficiency_res = assess_context_sufficiency(prompt_message, combined_rag_for_sufficiency)
    sufficiency_state = sufficiency_res.get("decision") if combined_rag_for_sufficiency else "no_context"

    important_links = get_important_links() if not is_guest else ""
    if important_links:
        if personal_context:
            personal_context = f"{personal_context}\n\n{important_links}"
        else:
            personal_context = important_links

    current_session_summary = get_current_session_summary(session_id) or ""

    system_prompt = get_system_prompt(
        mode,
        personal_context=personal_context,
        style_profile=style_profile,
        session_summaries=session_summaries_text,
        reference_context=reference_context,
        referenced_session_context=referenced_session_context,
        sufficiency_state=sufficiency_state,
        user_name=user_name,
        is_crisis=is_crisis,
        current_session_summary=current_session_summary,
    )

    # history_override (CoT replay mode): an explicitly provided historical
    # turn list replaces the CURRENT-session history fetch, so a replayed
    # old turn sees the context it would have had at the time. Default None
    # = normal path, byte-for-byte unchanged.

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

    NPU_HARD_TOKEN_LIMIT = 6144
    HARD_CAP_SAFETY_MARGIN_TOKENS = 600
    HARD_CAP_CHARS_PER_TOKEN = 1.5
    HARD_CAP_TARGET_CHARS = int(
        (NPU_HARD_TOKEN_LIMIT - HARD_CAP_SAFETY_MARGIN_TOKENS) * HARD_CAP_CHARS_PER_TOKEN
    )
    HISTORY_PROTECTED_FLOOR_TURNS = 4

    def _total_prompt_chars(sp: str, hist: list, pm: str) -> int:
        return len(sp) + sum(len(h["content"]) for h in hist) + len(pm)

    total_chars = _total_prompt_chars(system_prompt, history, prompt_message)
    trimmed = False
    trim_details = []

    # --- Budget ceiling: real tokens when measurable, chars otherwise ------
    # The trim CASCADE below is unchanged (same section priority order);
    # only the ceiling it measures against changed: from the pessimistic
    # 8,316-char cap to a real-token target DERIVED from npu_server's
    # reported effective ceiling (single-sourced there). When the ceiling
    # is unknown (npu down, or loaded without MAX_PROMPT_LEN), the old
    # conservative char budget is used and no measurement runs.
    target_tokens = _get_prompt_target_tokens()
    budget_chars = HARD_CAP_TARGET_CHARS
    measured_tokens = None
    if (measure_tokens is not None and target_tokens is not None
            and (total_chars > HARD_CAP_TARGET_CHARS
                 or target_tokens < PROMPT_TARGET_TOKENS)):
        probe = (
            [{"role": "system", "content": system_prompt}]
            + history
            + [{"role": "user", "content": prompt_message}]
        )
        measured_tokens = measure_tokens(probe)
        if measured_tokens is not None:
            # Derive this prompt's real chars-per-token ratio (clamped for
            # safety against degenerate content) and scale the char budget
            # so the cascade targets target_tokens real tokens.
            ratio = total_chars / max(measured_tokens, 1)
            ratio = max(2.0, min(6.0, ratio))
            budget_chars = int(target_tokens * ratio)
            trim_details.append({
                "stage": "measurement",
                "real_tokens": measured_tokens,
                "target_tokens": target_tokens,
                "chars_per_token_ratio": round(ratio, 2),
                "char_budget": budget_chars,
            })

    if total_chars > budget_chars and (reference_context or session_summaries_text):
        overflow = total_chars - budget_chars
        ref_before = len(reference_context)
        sum_before = len(session_summaries_text)
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
            referenced_session_context=referenced_session_context,
            sufficiency_state=sufficiency_state,
            user_name=user_name,
            is_crisis=is_crisis,
            current_session_summary=current_session_summary,
        )
        trimmed = True
        trim_details.append({
            "stage": "reference_then_summaries",
            "reference_chars_removed": ref_before - len(reference_context),
            "summaries_chars_removed": sum_before - len(session_summaries_text),
        })
        total_chars = _total_prompt_chars(system_prompt, history, prompt_message)

    if total_chars > budget_chars and personal_context:
        overflow = total_chars - budget_chars
        personal_before = len(personal_context)
        personal_context = personal_context[: max(0, len(personal_context) - overflow)]
        system_prompt = get_system_prompt(
            mode,
            personal_context=personal_context,
            style_profile=style_profile,
            session_summaries=session_summaries_text,
            reference_context=reference_context,
            referenced_session_context=referenced_session_context,
            sufficiency_state=sufficiency_state,
            user_name=user_name,
            is_crisis=is_crisis,
            current_session_summary=current_session_summary,
        )
        trimmed = True
        trim_details.append({
            "stage": "personal_context",
            "personal_chars_removed": personal_before - len(personal_context),
        })
        total_chars = _total_prompt_chars(system_prompt, history, prompt_message)

    history_turns_before = len(history)
    while total_chars > budget_chars and len(history) > HISTORY_PROTECTED_FLOOR_TURNS:
        history.pop(0)
        trimmed = True
        total_chars = _total_prompt_chars(system_prompt, history, prompt_message)

    history_protected_pops = 0
    while total_chars > budget_chars and history:
        history.pop(0)
        history_protected_pops += 1
        trimmed = True
        total_chars = _total_prompt_chars(system_prompt, history, prompt_message)
    if history_protected_pops:
        trim_details.append({
            "stage": "history_below_protected_floor",
            "turns_removed": history_protected_pops,
        })
    if len(history) != history_turns_before:
        trim_details.append({
            "stage": "history",
            "turns_removed": history_turns_before - len(history),
        })

    # --- Real-token drift check (only when measurement is available) -------
    # The char cascade is ratio-guided, not exact; if the real count is
    # still over target after the cascade, pop history turns in batches
    # and re-measure (each measure is a ~ms tokenize call, no inference).
    if measured_tokens is not None:
        for _ in range(5):
            probe = (
                [{"role": "system", "content": system_prompt}]
                + history
                + [{"role": "user", "content": prompt_message}]
            )
            real_now = measure_tokens(probe)
            if real_now is None:
                break  # measurement lost — char cascade already bounded it
            measured_tokens = real_now
            if real_now <= target_tokens:
                break
            over = real_now - target_tokens
            per_turn = max(1.0, real_now / max(len(history), 1))
            pops = min(len(history), int(over / per_turn) + 1)
            if pops <= 0:
                break
            history = history[pops:]
            trimmed = True
            trim_details.append({
                "stage": "history_real_remeasure",
                "turns_removed": pops,
                "real_tokens_before": real_now,
            })

    if trimmed:
        print(
            f"[prompt_trim] session '{session_id}' exceeded safety budget — "
            f"trimmed to fit. final size ~{total_chars} chars "
            f"(~{int(total_chars / HARD_CAP_CHARS_PER_TOKEN)} conservative tok est, "
            f"target was {budget_chars} chars, {len(history)} history turns remaining)"
        )

    messages = [{"role": "system", "content": system_prompt}] + \
        history + \
        [{"role": "user", "content": prompt_message}]

    return {
        "mode": mode,
        "messages": messages,
        "context_lookup_attempted": context_lookup_attempted,
        "personal_context": personal_context,
        "reference_context": reference_context,
        "session_summaries_text": session_summaries_text,
        "referenced_session_context": referenced_session_context,
        "style_profile": style_profile,
        "trimmed": trimmed,
        "trim_details": trim_details,
        "final_total_chars": total_chars,
        "measured_tokens": measured_tokens,
        "budget_chars": budget_chars,
        "laya_routing": laya_routing,
        "laya_sufficiency": sufficiency_res,
        "sufficiency_state": sufficiency_state,
        "is_crisis": is_crisis,
        "is_guest": is_guest,
    }


async def generate_reply(session_id: str, prompt_message: str, history_message: str = None, is_guest: bool = False) -> tuple[str, str]:
    if history_message is None:
        history_message = prompt_message

    prep = await asyncio.to_thread(
        _prepare_prompt_sync, session_id, prompt_message,
        _measure_prompt_tokens_sync, None, is_guest)

    mode = prep["mode"]
    is_crisis = prep.get("is_crisis", False)
    word_count = len(prompt_message.strip().split())
    laya_no_retrieval = (
        prep.get("laya_routing", {}).get("retrieval_decision") == "no_personal_context_needed"
    )

    # In supportive mode or acute crisis, never cap at short/cold replies
    if mode in ("supportive", "guidance") or is_crisis:
        is_casual_short = False
    else:
        is_casual_short = (
            word_count < 15
            or (mode == "conversational" and (word_count < 25 or laya_no_retrieval))
            or laya_no_retrieval
            or not (prep.get("personal_context") or prep.get("reference_context"))
        )
    predict_tokens = 75 if is_casual_short else 250

    raw_reply = await _call_npu_once(prep["messages"], num_predict=predict_tokens)
    bot_reply = await asyncio.to_thread(scrub_reply, raw_reply, session_id=session_id, mode=mode, is_crisis=is_crisis)
    if is_casual_short:
        sentences = re.split(r"(?<=[.!?])\s+", bot_reply)
        if len(sentences) > 2:
            bot_reply = " ".join(sentences[:2]).strip()

    try:
        context_present = bool(prep["personal_context"]) or bool(prep["reference_context"])
        # Step-1 instrumentation (prompt-budget investigation): per-section
        # char/token estimates + trim details. Pure arithmetic, no NPU cost.
        # Token figures are the chars/2.5 estimate, not real tokenizer counts.
        _cpe = 2.5
        sys_chars = len(prep["messages"][0]["content"])
        hist_chars = sum(len(m["content"]) for m in prep["messages"][1:-1])
        user_chars = len(prep["messages"][-1]["content"])
        section_chars = {
            "system_prompt": sys_chars,
            "history": hist_chars,
            "prompt_message": user_chars,
            "personal_context": len(prep["personal_context"]),
            "reference_context": len(prep["reference_context"]),
            "session_summaries": len(prep["session_summaries_text"]),
            "referenced_session_context": len(prep.get("referenced_session_context", "")),
            "style_profile": len(prep.get("style_profile", "")),
        }
        sufficiency_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "mode": mode,
            "context_lookup_attempted": prep["context_lookup_attempted"],
            "personal_context_chars": len(prep["personal_context"]),
            "reference_context_chars": len(prep["reference_context"]),
            "session_summaries_chars": len(prep["session_summaries_text"]),
            "context_present": context_present,
            "sufficiency_instruction_fired": context_present,
            "trimmed": prep["trimmed"],
            "real_prompt_tokens": prep.get("measured_tokens"),
            "prompt_target_tokens": prep.get("target_tokens"),
            "est_prompt_tokens": int((sys_chars + hist_chars + user_chars) / _cpe),
            "section_chars": section_chars,
            "section_token_estimates": {
                k: int(v / _cpe) for k, v in section_chars.items()
            },
            "trim_details": prep.get("trim_details", []),
            "prompt_message_preview": prompt_message[:200],
            "reply_text": bot_reply,
            "reply_flagged_uncertainty": _reply_reads_as_uncertain(bot_reply),
            # Laya typed-decision telemetry (backward compatible)
            "laya_mode": {
                "chosen_mode": prep.get("laya_routing", {}).get("mode", mode),
                "confidence": prep.get("laya_routing", {}).get("mode_confidence", 1.0),
                "used_fallback": prep.get("laya_routing", {}).get("used_fallback", True),
            },
            "laya_rag_filtered_count": prep.get("laya_rag_filtered_count", 0),
            "laya_sufficiency_decision": prep.get("sufficiency_state", "no_context"),
            "laya_sufficiency_confidence": prep.get("laya_sufficiency", {}).get("confidence", 0.0),
            "laya_latency_ms": round(
                prep.get("laya_routing", {}).get("latency_ms", 0.0)
                + prep.get("laya_sufficiency", {}).get("latency_ms", 0.0),
                2
            ),
        }))
    except Exception as e:
        print(f"[sufficiency_log] error, skipping: {e}")

    await asyncio.to_thread(save_turn, session_id, history_message, bot_reply, mode)
    create_background_task(_background_rollup(session_id))

    return bot_reply, mode


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, key_info: dict = Depends(verify_api_key)):
    _enforce_session_scope(key_info, payload.session_id)
    is_guest = (key_info.get("label") != "primary" or key_info.get("session_id") is not None)
    prompt_message = payload.message
    history_message = payload.message

    if payload.attachments:
        extracted_parts = await asyncio.gather(*[
            asyncio.to_thread(decode_app_attachment, att) for att in payload.attachments
        ])
        file_names = [att.get("name", "file") for att in payload.attachments]

        extracted_text = "\n\n".join(extracted_parts)
        prompt_message = f"{payload.message}\n\n{extracted_text}".strip()
        history_message = f"[Sent file(s): {', '.join(file_names)}] {payload.message}".strip()

    bot_reply, mode = await generate_reply(payload.session_id, prompt_message, history_message=history_message, is_guest=is_guest)
    return ChatResponse(reply=bot_reply, mode=mode)


@app.post("/chat/upload", response_model=ChatResponse)
async def chat_upload(
    key_info: dict = Depends(verify_api_key),
    file: UploadFile = File(...),
    message: str = Form(""),
    session_id: str = Form("default")
):
    _enforce_session_scope(key_info, session_id)
    is_guest = (key_info.get("label") != "primary" or key_info.get("session_id") is not None)
    file_bytes = await file.read()
    extracted = await asyncio.to_thread(process_uploaded_file, file.filename, file_bytes)

    if message.strip():
        prompt_message = f"{message.strip()}\n\n{extracted}"
    else:
        prompt_message = extracted

    history_message = f"[Sent a file: {file.filename}]"
    if message.strip():
        history_message += f" {message.strip()}"

    bot_reply, mode = await generate_reply(session_id, prompt_message, history_message=history_message, is_guest=is_guest)
    return ChatResponse(reply=bot_reply, mode=mode)


@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = Query(default=50, ge=1, le=500), key_info: dict = Depends(verify_api_key)):
    _enforce_session_scope(key_info, session_id)
    turns = get_full_session_history(session_id, limit=limit)
    return {"session_id": session_id, "messages": turns}


@app.post("/summarize")
async def summarize(payload: SummarizeRequest, key_info: dict = Depends(verify_api_key)):
    session_id = payload.session_id
    _enforce_session_scope(key_info, session_id)

    if _summarize_jobs.get(session_id) == "in_progress":
        return {"status": "already_in_progress", "session_id": session_id}

    if not payload.force and session_has_summary(session_id):
        return {"status": "skipped", "reason": "no turns found or already summarized"}

    create_background_task(
        _background_summarize(session_id, payload.force)
    )
    return {"status": "queued", "session_id": session_id}


@app.get("/summarize/status/{session_id}")
async def summarize_status(session_id: str, key_info: dict = Depends(verify_api_key)):
    _enforce_session_scope(key_info, session_id)
    job_state = _summarize_jobs.get(session_id)

    if job_state == "in_progress":
        return {"session_id": session_id, "status": "in_progress"}

    if job_state and job_state.startswith("error"):
        return {"session_id": session_id, "status": "error", "detail": job_state}

    if session_has_summary(session_id):
        return {
            "session_id": session_id,
            "status": "done",
            "summary": get_summary_for_session(session_id),
        }

    return {"session_id": session_id, "status": "not_started"}


@app.post("/debug/trigger_followup")
async def debug_trigger_followup(key_info: dict = Depends(verify_api_key)):
    # No client-supplied session_id here (it targets the most-recently-active
    # session), so session scoping can't apply — instead this endpoint is
    # PRIMARY-ONLY: a scoped (guest) key must not be able to trigger pushes
    # into someone else's session.
    if key_info.get("session_id") is not None:
        raise HTTPException(status_code=403, detail="Session not permitted for this key")
    await check_and_send_followups(
        push_message_to_session, active_connections, _call_npu_once, scrub_reply
    )
    return {"status": "triggered"}


# --- CoT diagnostic endpoint (Stage: manually-triggered replay tool) --------
# SINGLE-CALL chain-of-thought: the model emits <reasoning> + <reply> in one
# generation (no second NPU call). ACTIVATION: POST /chat/cot WITH the
# header "X-CoT-Activate: diagnostic-replay" — two explicit signals (the
# dedicated URL and the header); existing clients never send either, so it
# cannot fire accidentally. PRIMARY-ONLY (guest keys get 403, same pattern
# as /debug/trigger_followup). Output budget is raised for this endpoint
# ONLY: 700 tokens = the normal 300-token reply budget + ~400 tokens of
# reasoning headroom; at the observed ~1 tok/s on this NPU that's a
# multi-minute generation, which is explicitly acceptable for a diagnostic.
# The self-narrated reasoning is NOT ground truth — it is returned with a
# non-authoritative label (small models post-hoc rationalize; see the
# feasibility report). Nothing here is persisted: no save_turn, no rollup,
# no sufficiency-log record.
COT_TRIGGER_HEADER = "X-CoT-Activate"
COT_TRIGGER_VALUE = "diagnostic-replay"
COT_NUM_PREDICT = 700
COT_NOTE = ("model's self-reported reasoning — NOT verified to reflect its "
            "actual internal process; diagnostic signal only")
# Format rules live in the SYSTEM position (appended post-trim in chat_cot),
# NOT the user turn: field testing showed a user-turn format demand got
# ignored entirely (3/3 plain replies, zero tags), while a system-position
# demand produced tag-compliant output. Also appended AFTER the assembled
# system prompt so it never displaces the style profile's last-position
# advantage and never eats RAG context via the trim cascade.
COT_INSTRUCTION = (
    "[DIAGNOSTIC REASONING MODE — this turn only]\n"
    "Output STRICTLY two blocks and nothing else:\n"
    "<reasoning>\n"
    "brief analysis (2-4 sentences): what the user's message actually "
    "means, which context above matters, what the reply should accomplish\n"
    "</reasoning>\n"
    "<reply>\n"
    "your actual reply — in your normal chat voice exactly (casual, "
    "contractions, same length and tone as a regular reply), as if the "
    "reasoning above didn't exist\n"
    "</reply>\n"
    "IMPORTANT: the history above contains plain-text replies from earlier "
    "normal-mode turns — that pattern does NOT apply to this turn. Your "
    "output MUST begin with <reasoning> and the wrapper is MANDATORY even "
    "for short casual replies: NEVER skip the tags, NEVER reply in plain "
    "text, ALWAYS continue to the <reply> block after </reasoning>, and "
    "your output MUST always END with the closing </reply> tag.")


def _parse_cot_output(raw: str):
    """Strict delimited-output parser. Returns (reasoning, reply) or
    (None, failure_detail). The <reasoning> block MUST be present and
    closed, and nothing may appear before it — a malformed/missing
    reasoning block still hard-fails (no-leak guarantee). The ONE leniency
    (added 2026-09): an unclosed <reply> that runs to end-of-generation is
    accepted as the complete reply — the model empirically hits EOS right
    after the reply content in ~80% of turns, the reply content itself is
    unambiguous (nothing follows it), and <reasoning> is verified closed
    first, so nothing can leak into it. Any text AFTER a properly closed
    </reply>, or missing <reply>, is still a hard failure."""
    # Allow <think>...</think> to map cleanly to <reasoning>...</reasoning>
    if "<think>" in raw and "<reasoning>" not in raw:
        raw = raw.replace("<think>", "<reasoning>", 1).replace("</think>", "</reasoning>", 1)

    r_open = raw.find("<reasoning>")
    r_close = raw.find("</reasoning>")
    if r_open == -1 or r_close == -1 or r_close < r_open:
        return None, "missing or malformed <reasoning> block"
    if raw[:r_open].strip():
        return None, "unexpected text before <reasoning>"
    reasoning = raw[r_open + len("<reasoning>"):r_close].strip()
    p_open = raw.find("<reply>", r_close)
    if p_open == -1:
        # If model did not wrap the subsequent text in <reply>, accept text after </reasoning>
        post_reasoning = raw[r_close + len("</reasoning>"):].strip()
        if post_reasoning:
            return reasoning, post_reasoning
        return None, "missing <reply> block after reasoning"
    p_close = raw.find("</reply>", p_open)
    if p_close != -1:
        reply = raw[p_open + len("<reply>"):p_close].strip()
        post = raw[p_close + len("</reply>"):].strip()
        if post:
            return None, "unexpected text after </reply>"
    else:
        # EOS-closed reply: lenient accept (see docstring).
        reply = raw[p_open + len("<reply>"):].strip()
    if not reply:
        return None, "empty <reply> block"
    return reasoning, reply


class CotRequest(BaseModel):
    message: str = ""                      # live mode
    session_id: str = "default"            # live mode
    replay: Optional[dict] = None          # {"session_id": ..., "turn_rowid": int}


@app.post("/chat/cot")
async def chat_cot(payload: CotRequest, request: Request,
                   key_info: dict = Depends(verify_api_key)):
    # Primary-only (guest keys get 403), same pattern as
    # /debug/trigger_followup.
    if key_info.get("session_id") is not None:
        raise HTTPException(status_code=403, detail="Session not permitted for this key")
    # Explicit activation signal: the dedicated URL AND this header must
    # both be present. Missing or wrong value -> 400 (not silent).
    if request.headers.get(COT_TRIGGER_HEADER) != COT_TRIGGER_VALUE:
        raise HTTPException(
            status_code=400,
            detail=f"CoT mode requires the {COT_TRIGGER_HEADER}: "
                   f"{COT_TRIGGER_VALUE} header")

    if payload.replay:
        replay_sid = payload.replay.get("session_id")
        turn_rowid = payload.replay.get("turn_rowid")
        if not replay_sid or not isinstance(turn_rowid, int):
            raise HTTPException(status_code=400, detail=
                                "replay requires {session_id, turn_rowid}")
        message, history_override = memory.get_replay_turn(replay_sid, turn_rowid)
        if message is None:
            raise HTTPException(status_code=404, detail=
                                "replay turn not found (rowid must exist in "
                                "that session and be a user-role row)")
        session_id = replay_sid
    else:
        if not payload.message.strip():
            raise HTTPException(status_code=400, detail="empty message")
        session_id = payload.session_id
        history_override = None
        message = payload.message

    # Same context assembly as the normal path — with the CoT format rules
    # at the USER-TURN position (last thing the model reads). Approved
    # placement revert after 0/6 compliance at system position (recency/
    # positional issue, not wording). Known tradeoff: the instruction adds
    # to prompt_message, so the trim cascade may displace some RAG personal
    # context — actual displacement is measured and reported, not assumed.
    # Prepare prompt using clean user message so RAG/routing/sufficiency are not corrupted
    prep = await asyncio.to_thread(
        _prepare_prompt_sync, session_id, message,
        _measure_prompt_tokens_sync, history_override, False)

    # Attach CoT instructions specifically to the final user prompt turn
    if prep["messages"] and prep["messages"][-1]["role"] == "user":
        prep["messages"][-1]["content"] += "\n\n" + COT_INSTRUCTION

    t0 = time.time()
    raw = await _call_npu_once(prep["messages"], num_predict=COT_NUM_PREDICT)
    duration_s = round(time.time() - t0, 1)

    reasoning, reply = _parse_cot_output(raw)
    if reasoning is None:
        # Explicit failure mode: never guess-split, never emit anything
        # labeled as a reply.
        return {
            "error": "cot_parse_failed",
            "detail": reply,
            "raw_output": raw,
            "note": COT_NOTE,
        }

    # Tone parity with the normal path: the parsed reply goes through the
    # SAME scrub pipeline as /chat (emoji removal, AI-tell phrases,
    # structure rules, semantic style strip).
    reply = await asyncio.to_thread(scrub_reply, reply, session_id=session_id, mode=prep["mode"], is_crisis=prep.get("is_crisis", False))

    return {
        "mode": prep["mode"],
        "reasoning": {"content": reasoning, "note": COT_NOTE},
        "reply": {"content": reply},
        "measured_tokens": prep.get("measured_tokens"),
        "duration_s": duration_s,
        "num_predict": COT_NUM_PREDICT,
        "parse": "ok",
    }

# <!-- COT ENDPOINT -->

@app.post("/search")
async def search(payload: SearchRequest, key_info: dict = Depends(verify_api_key)):
    # Primary-only: /search keyword-scans the personal_data corpus directly,
    # so a scoped (guest) key must not be able to query it (flagged during
    # the guest-access work).
    if key_info.get("session_id") is not None:
        raise HTTPException(status_code=403, detail="Session not permitted for this key")
    # keyword_search() is a synchronous full scan of every .txt under
    # personal_data/ — CPU+IO bound. Run it in a worker thread so the
    # event loop stays responsive (websocket pings/acks, other requests)
    # for the duration of the scan, same pattern as generate_reply().
    results = await asyncio.to_thread(
        keyword_search,
        payload.query,
        context_lines=payload.context_lines,
        max_results=payload.max_results,
    )
    if not results:
        return {"status": "no_matches", "results": []}

    matches = results.split("\n---\n")
    return {"status": "ok", "count": len(matches), "results": matches}


@app.post("/ws/ticket")
async def issue_ws_ticket(payload: TicketRequest, key_info: dict = Depends(verify_api_key)):
    # Scoping is enforced HERE, at ticket issuance: the ticket is then bound
    # to payload.session_id and /ws/chat verifies ticket+session match on
    # connect, so a guest ticket is unusable on any other session.
    _enforce_session_scope(key_info, payload.session_id)
    is_guest = (key_info.get("label") != "primary" or key_info.get("session_id") is not None)
    ticket = _issue_ws_ticket(payload.session_id, is_guest=is_guest)
    return {"ticket": ticket, "expires_in": WS_TICKET_TTL_SECONDS}


async def _process_ws_chat_message(session_id: str, prompt_message: str, history_message: str, msg_id: str = None):
    # Serialize message turns strictly per session so rapid incoming messages
    # never race to build prompts before earlier turns are durably saved.
    async with _session_locks[session_id]:
        try:
            is_guest = session_guest_status.get(session_id, False)
            bot_reply, mode = await generate_reply(session_id, prompt_message, history_message=history_message, is_guest=is_guest)
            # By the time generate_reply() has returned, save_turn() has already
            # completed inside it — the user's turn (and the bot's reply) are
            # durably persisted. Tell the client explicitly, separate from the
            # earlier parse-time `ack`, so "delivered" and "durably stored" are
            # two distinct, honest signals instead of one conflated one.
            if msg_id:
                _mark_message_completed(session_id, msg_id)
                ws = active_connections.get(session_id)
                if ws:
                    try:
                        await ws.send_json({"type": "stored", "id": msg_id})
                    except Exception as e:
                        print(f"[ws] failed to send 'stored' confirmation for {msg_id} in session '{session_id}': {e}")
            await push_message_to_session(session_id, bot_reply, mode=mode)
        except Exception as e:
            if msg_id:
                _mark_message_failed(session_id, msg_id)
            print(f"[ws] reply generation error in session '{session_id}': {e}")
            await push_message_to_session(
                session_id,
                "⚠️ An error occurred processing your request.",
                mode="error"
            )


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket, session_id: str = Query("default"), ticket: str = Query(None)):
    valid, is_guest = _consume_ws_ticket(ticket, session_id) if ticket else (False, False)
    if not ticket or not valid:
        await websocket.close(code=4401)
        return

    await websocket.accept()
    active_connections[session_id] = websocket
    session_guest_status[session_id] = is_guest
    print(f"[ws] session '{session_id}' connected (guest={is_guest})")

    # 1. Drain pending queued messages
    queue = list(pending_messages[session_id])
    for payload in queue:
        try:
            await websocket.send_json(payload)
            print(f"[ws] drained queued message {payload['id']} to session '{session_id}'")
        except Exception:
            break

    # 2. Main incoming message loop
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if msg_type == "ack":
                ack_id = data.get("id")
                if ack_id and session_id in pending_messages:
                    pending_messages[session_id] = [
                        m for m in pending_messages[session_id] if m.get("id") != ack_id
                    ]
                continue

            if msg_type == "chat_message":
                msg_id = data.get("id")
                if msg_id:
                    await websocket.send_json({"type": "ack", "id": msg_id})

                    if _is_message_duplicate_or_inflight(session_id, msg_id):
                        print(f"[ws] duplicate or in-flight chat_message {msg_id} for session '{session_id}', skipping reprocess")
                        continue
                    _mark_message_inflight(session_id, msg_id)

                raw_message = data.get("message") or data.get("text", "")
                attachments = data.get("attachments")

                prompt_message = raw_message
                history_message = raw_message

                if attachments:
                    extracted_parts = await asyncio.gather(*[
                        asyncio.to_thread(decode_app_attachment, att) for att in attachments
                    ])
                    file_names = [att.get("name", "file") for att in attachments]
                    extracted_text = "\n\n".join(extracted_parts)
                    prompt_message = f"{raw_message}\n\n{extracted_text}".strip()
                    history_message = f"[Sent file(s): {', '.join(file_names)}] {raw_message}".strip()

                create_background_task(
                    _process_ws_chat_message(session_id, prompt_message, history_message, msg_id=msg_id)
                )

    except WebSocketDisconnect:
        print(f"[ws] session '{session_id}' disconnected")
    finally:
        if active_connections.get(session_id) is websocket:
            del active_connections[session_id]