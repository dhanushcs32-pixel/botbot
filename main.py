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
from pathlib import Path
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
from few_shot_retriever import (
    retrieve_few_shot_examples,
    format_few_shot_block,
    ENABLE_DYNAMIC_FEW_SHOT,
)
from mode_router import (
    detect_mode,
    detect_mode_detailed,
    get_system_prompt,
    get_retrieval_policy,
    get_sampling_profile,
    SAMPLING_PROFILES,
    ENABLE_DECODE_GUARDS,
    ENABLE_STRUCTURED_PLAN,
    build_structured_plan,
    extract_message_focus,
    STOCK_REGEX,
    pick_opener,
)
from memory import (
    save_turn, get_recent_history, get_recent_summaries, get_all_session_ids,
    get_summary_for_session, get_important_links, maybe_rollup_session,
    is_test_session, session_has_summary, get_sessions_with_last_activity,
    count_turns, save_bot_initiated_message, get_current_session_summary,
    get_full_session_history, deduplicate_assistant_turns, get_last_user_timestamp,
    filter_clean_history,
)
from time_utils import (
    time_note, human_gap, get_time_bucket, get_current_time, parse_timestamp_safe,
    is_new_session, resolve_relative_dates_for_storage, format_memory_dates_for_retrieval
)

ENABLE_TIME_NOTE = os.environ.get("ENABLE_TIME_NOTE", "1").lower() in ("1", "true", "yes")
from rag import (
    search_personal_data, get_style_profile, keyword_search, search_reference_data,
    is_trivial_message, should_bypass_personal_retrieval, get_known_people,
    get_last_laya_filtered_count, should_retrieve, ENABLE_RETRIEVAL_GATE
)
from laya_engine import assess_context_sufficiency, LAYA_ENABLED, is_laya_available
from semantic_style_check import strip_semantic_violations
from summarizer import summarize_session, summarize_turns_batch
from media import process_uploaded_file, decode_app_attachment
from secret_key import API_KEYS
from followup import check_and_send_followups
from tone_reader import orchestrate_turn, ENABLE_LIGHT_ORCHESTRATOR

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
PERSONAL_DATA_DIR = Path(os.getenv("PERSONAL_DATA_DIR", Path(__file__).parent / "personal_data"))
SUFFICIENCY_LOG_PATH = Path(os.getenv("SUFFICIENCY_LOG_PATH", PERSONAL_DATA_DIR / "context_sufficiency_log.jsonl"))

sufficiency_logger = logging.getLogger("botbot.sufficiency")
sufficiency_logger.setLevel(logging.INFO)
_sufficiency_file_handler = RotatingFileHandler(
    str(SUFFICIENCY_LOG_PATH), maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
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
    "\U00002300-\U000023FF"
    "\U00002B00-\U00002BFF"
    "\U0001F000-\U0001F2FF"
    "\uFE0F\u20E3"  # variation selector-16 / keycap: otherwise left behind as an invisible stray char
    "]+",
    flags=re.UNICODE,
)

# Only phrases that are essentially never part of a good companion reply.
# (The bare "i understand" used to be here and deleted whole sentences like
# "As I understand it, you never told me a time." and "I understand how heavy
# that feels." -- plain empathy and honest hedging, exactly what the
# anti-hallucination prompt asks for. The canned "totally okay/normal",
# "remember, it's okay" and "good luck with" were removed for the same reason.)
AI_TELL_PHRASES = [
    "i understand how you feel", "great question", "certainly!", "i'd be happy to",
    "feel free to", "let me know if you have any other questions",
    "i hope this helps", r"\bas an ai (?:language model|assistant)\b", "it's important to note that",
    "sure thing!", "got it!", "no problem at all", "you're welcome!",
    "don't hesitate to reach out",
]

RP_VERB_PREFIX_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:pauses|sighs|stretches|glances|typing slowly|softly|quietly|whispers|smiles|laughs|nods|shrugs|looks|stares)[^:\n]{0,20}:\s*"
)

# *sighs*, *looks at you softly*, *takes a deep breath* ... but NOT ordinary
# emphasis like "I *really* mean it" or math like "5 * 3 * 2".
ASTERISK_ACTION_PATTERN = re.compile(
    r"(?i)(?<!\w)\*(?:sighs?|pauses?|smiles?|laughs?|nods?|shrugs?|looks?|glances?|yawns?|stretche?s?|"
    r"whispers?|grins?|hugs?|giggles?|chuckles?|blinks?|frowns?|winks?|sniffs?|takes?|leans?|"
    r"waits?|tilts?|breathes?|exhales?|inhales?)\b[a-zA-Z\s,.'-]{0,60}\*(?!\w)\s*"
)


def strip_parenthetical_stage_directions(text: str) -> str:
    """Strips parenthetical stage directions like (quiet), (whisper), (softly) at the
    start of lines or sentences, while leaving code and math parentheses untouched."""
    if not text:
        return text
    # At start of message or newline
    text = re.sub(r'(?im)^[ \t]*\([a-zA-Z\s]{2,25}\)\s*:?\s*', '', text)
    # Immediately following sentence-ending punctuation (. ! ?)
    text = re.sub(r'(?i)(?<=[.!?])[ \t]*\([a-zA-Z\s]{2,25}\)\s*:?\s*', ' ', text)
    return text

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
        "name": "asterisk_action",
        "pattern": ASTERISK_ACTION_PATTERN,
        "action": "strip",
    },
    {
        "name": "bold_emphasis",
        "pattern": re.compile(r"\*{1,2}(?!\s)([^*\n]{1,80}?)(?<!\s)\*{1,2}"),
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
        "pattern": re.compile(
            r"(?i)\[(?:sighs?|laughs?|pauses?|smiles?|looks?|glances?|nods?|shrugs?|"
            r"chuckles?|giggles?|whispers?|stares?|takes?|leans?|blinks?|frowns?|winks?)[^\]\n]{0,80}\]\s*"
        ),
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
    Detects pathological 1-word newline stacking (slam-poetry / fragmented monologue)
    e.g. 4+ consecutive lines with 1 word ending with a period.
    Collapses excessive newlines into normal horizontal prose.
    """
    if "\n" not in text:
        return text

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    consecutive_short_punctuated = 0
    max_run = 0
    for line in lines:
        words = line.split()
        if len(words) == 1 and line.endswith("."):
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
    for phrase in AI_TELL_PHRASES:
        if phrase.startswith(r"\b") or "\\" in phrase:
            if re.search(phrase, lowered):
                violations.append("ai_tell_phrase")
                break
        elif phrase in lowered:
            violations.append("ai_tell_phrase")
            break
    for rule in STRUCTURE_RULES:
        min_matches = rule.get("min_matches", 1)
        if min_matches == 1:
            if rule["pattern"].search(text):
                violations.append(rule["name"])
        else:
            if len(rule["pattern"].findall(text)) >= min_matches:
                violations.append(rule["name"])
    return violations


HELPLINE_PATTERN = re.compile(
    r"\b(?:988\d*|14416|tele[\s-]?manas|helplines?|hotlines?|crisis\s+(?:lifeline|line|text)|"
    r"suicide\s+(?:prevention|hotline|lifeline)|aasra|vandrevala|icall|samaritans)\b",
    re.IGNORECASE,
)
HELPLINE_FALLBACK_REPLY = "I'm right here with you. Tell me what's going on."


def scrub_reply(text: str, session_id: str = "", mode: str = "", is_crisis: bool = False, prompt_message: str = "") -> str:
    if not text or not text.strip():
        return ""

    # 1. Strip raw model control tags: <think>...</think>, <|im_end|>, <|endoftext|>, <|im_start|>
    text = re.sub(r'(?s)<think>.*?</think>', '', text)
    text = re.sub(r'<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>', '', text)

    # 1b. Strip accidental timestamp/time-note/plan leaks
    text = re.sub(r'(?i)\s*\[now:[^\]]*\]', '', text)
    text = re.sub(r'(?i)\s*\[after[^\]]*gap\]', '', text)
    text = re.sub(r'(?i)\s*\[you last spoke:[^\]]*\]', '', text)
    text = re.sub(r'(?i)\s*\[plan:[^\]]*\]', '', text)
    text = re.sub(r'(?i)\s*\[examples of how you text[^\]]*\]', '', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # 2. Strip leading role prefixes if the model accidentally emits "Assistant:" or "Bot:" at index 0
    text = re.sub(r'^(?:assistant|bot|bot bot)\s*:\s*', '', text, flags=re.IGNORECASE)

    # 3. Strip vertical dot trails, standalone punctuation lines, and spaced dot chains
    text = re.sub(r'(?:\r?\n\s*[.\-_*?]+\s*)+', '\n', text)
    text = re.sub(r'(?:\.\s+){2,}\.?', '.', text)
    text = re.sub(r'\bJust\.\s*\.*', '', text, flags=re.IGNORECASE)

    # 4. Strip leading "Aah, I see." filler if the model emits it
    text = re.sub(r'^Aah,\s*(?:I\s+see[.,!]?\s*)?', '', text, flags=re.IGNORECASE)

    # 5. Collapse 2x+ consecutive repeated phrases/clauses (e.g. "I'm here. I'm here.")
    text = re.sub(
        r'(\b[\w\']+(?:\s+[\w\']+){0,5}\b)(?:[,.!]+\s*\1\b)+',
        r'\1.',
        text,
        flags=re.IGNORECASE
    )

    # 6. Collapse multi-line vertical stacking into clean horizontal sentences
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not re.fullmatch(r'[.\-_*?\s]+', ln.strip())]
    text = " ".join(lines)

    # 7. Strip roleplay asterisks and emojis via Python post-processing (Phase 4)
    text = re.sub(r'\*[^*]+\*', '', text)
    text = EMOJI_PATTERN.sub('', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    return text.strip()


# --- Decode Guards Validation & Regeneration (Phase 2) -----------------------
VALIDATE_ATTRACTOR_REGEX = re.compile(
    r"(?i)\b(i'm just here|not fixing|let me just be|not going anywhere|aah,? i see|i was so full)\b"
)
VALIDATE_LOOP_PERIOD_REGEX = re.compile(r"(?:\.\s*\n){2,}")
VALIDATE_LOOP_DOTS_REGEX = re.compile(r"(?:\.\s+){2,}\.|\.\s*[\r\n]+\s*\.|(?:\.{2,}\s*){2,}")
VALIDATE_LOOP_REPEATED_CLAUSE_REGEX = re.compile(
    r"(\b[\w\']+(?:\s+[\w\']+){0,3}\b)(?:[,.]\s*\1){2,}", re.IGNORECASE
)
VALIDATE_TIMESTAMP_LEAK_REGEX = re.compile(r"(?i)(\[now:|you last spoke:|\[gap:|\[plan:)")
VALIDATE_NON_PUNCT_REGEX = re.compile(r"[a-zA-Z0-9]")

regeneration_count = 0

# Multi-turn anti-collapse state (bans, idempotency, fatigue)
_SESSION_BANS: Dict[str, set] = {}
_IDEMPOTENCY_CACHE: Dict[str, Tuple[str, str, float]] = {}

BAN_USER_REGEX = re.compile(
    r'(?i)\b(?:stop|quit|don\'t|dont)\s+(?:saying|repeating|using)\s+(?:the\s+(?:term|phrase|word)\s+)?["\']?([^"\'\n.!?]+)["\']?'
)


def extract_and_register_user_bans(session_id: str, message: str) -> set:
    """Detects explicit user ban instructions and registers banned phrases for the session."""
    if session_id not in _SESSION_BANS:
        _SESSION_BANS[session_id] = set()
    matches = BAN_USER_REGEX.findall(message or "")
    for m in matches:
        phrase = m.strip().lower()
        if len(phrase) >= 2:
            _SESSION_BANS[session_id].add(phrase)
            print(f"[user_ban] Registered session ban for '{session_id}': '{phrase}'")
    return _SESSION_BANS[session_id]


def fatigued_ngrams(recent_bot: List[str], n: int = 3, min_count: int = 2) -> set:
    """Extracts n-grams repeated min_count or more times across the last 8 bot replies."""
    from collections import Counter
    c = Counter()
    for r in recent_bot[-8:]:
        words = re.findall(r"\b\w+\b", (r or "").lower())
        for i in range(len(words) - n + 1):
            c[tuple(words[i:i+n])] += 1
    return {ng for ng, count in c.items() if count >= min_count}


def has_fatigue(reply: str, banned_ngrams: set, n: int = 3) -> bool:
    """Checks if reply contains any fatigued n-grams."""
    if not banned_ngrams:
        return False
    words = re.findall(r"\b\w+\b", (reply or "").lower())
    for i in range(len(words) - n + 1):
        if tuple(words[i:i+n]) in banned_ngrams:
            return True
    return False


def has_repeated_opener(reply: str, recent_bot: List[str], max_first_word_repeats: int = 2) -> bool:
    """Checks if reply's first word repeats in >= max_first_word_repeats of the last 5 bot replies."""
    words = re.findall(r"\b\w+\b", (reply or "").lower())
    if not words:
        return False
    first_word = words[0]
    count = 0
    for r in recent_bot[-5:]:
        rw = re.findall(r"\b\w+\b", (r or "").lower())
        if rw and rw[0] == first_word:
            count += 1
    return count >= max_first_word_repeats


def bad_reply(
    reply: str,
    recent_bot_replies: Optional[List[str]] = None,
    session_bans: Optional[set] = None,
    thr: float = 0.50
) -> Tuple[bool, str]:
    """
    Checks if reply contains explicit user bans, fatigued openers, fatigued n-grams,
    stock phrases, duplicates prior replies, or has > thr Jaccard token overlap with
    any of the last 5 bot replies.
    Returns (True, reason) if bad, or (False, "ok") if clean.
    """
    if not reply or not reply.strip():
        return True, "empty_reply"

    reply_lower = reply.lower()

    # 1. Reject explicit user-banned phrases
    if session_bans:
        for banned in session_bans:
            if banned and banned in reply_lower:
                return True, f"banned_phrase:{banned}"

    # 2. Reject repeated openers (first word appears in >= 2 of last 5 replies)
    if recent_bot_replies and has_repeated_opener(reply, recent_bot_replies, max_first_word_repeats=2):
        words = re.findall(r"\b\w+\b", reply_lower)
        fw = words[0] if words else ""
        return True, f"fatigued_opener:{fw}"

    # 3. Reject dynamic fatigue n-grams (3-grams appearing >= 2 times in last 8 bot replies)
    if recent_bot_replies:
        banned_ng = fatigued_ngrams(recent_bot_replies, n=3, min_count=2)
        if has_fatigue(reply, banned_ng, n=3):
            return True, "fatigued_ngram"

    # 4. Reject stock phrases
    if STOCK_REGEX.search(reply):
        return True, "stock_phrase"

    # 5. Cross-turn duplicate / high-overlap check
    if recent_bot_replies:
        rep_words = set(re.findall(r"\b\w+\b", reply_lower))
        for prev in recent_bot_replies[-5:]:
            if not prev:
                continue
            prev_lower = prev.lower()
            if reply_lower == prev_lower:
                return True, "exact_duplicate_reply"
            prev_words = set(re.findall(r"\b\w+\b", prev_lower))
            if rep_words and prev_words:
                jaccard = len(rep_words & prev_words) / len(rep_words | prev_words)
                if jaccard > thr:
                    return True, f"high_consecutive_overlap:{jaccard:.2f}"

    return False, "ok"


def validate_reply(
    text: str,
    recent_bot_replies: Optional[List[str]] = None,
    session_bans: Optional[set] = None
) -> tuple[bool, str]:
    """
    Validates model reply against attractor collapse, loops, timestamp leaks, stock phrases,
    user-banned phrases, and cross-turn repetition.
    Returns (True, "ok") if valid, or (False, reason) if invalid.
    """
    if not text or not text.strip():
        return False, "empty_reply"

    chars = VALIDATE_NON_PUNCT_REGEX.findall(text)
    if len(chars) < 2:
        return False, "empty_or_too_short"

    if VALIDATE_ATTRACTOR_REGEX.search(text):
        return False, "attractor_phrase"

    if VALIDATE_TIMESTAMP_LEAK_REGEX.search(text):
        return False, "timestamp_leak"

    if (
        VALIDATE_LOOP_PERIOD_REGEX.search(text)
        or VALIDATE_LOOP_DOTS_REGEX.search(text)
        or VALIDATE_LOOP_REPEATED_CLAUSE_REGEX.search(text)
    ):
        return False, "loop_or_stutter"

    is_bad, reason = bad_reply(text, recent_bot_replies, session_bans)
    if is_bad:
        return False, reason

    return True, "ok"


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
    allow_origin_regex=r"^https?://(?:localhost|127\.0\.0\.1|10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2}|100\.(?:6[4-9]|[7-9]\d|1[0-1]\d|12[0-7])(?:\.\d{1,3}){2}|.*\.trycloudflare\.com)(?::\d+)?$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB limit for file uploads and attachments
WS_MAX_MESSAGE_BYTES = 2 * 1024 * 1024  # 2 MB limit for WebSocket messages

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


async def _call_npu_once(messages: list, num_predict: int = 256, is_cot: bool = False, options: Optional[dict] = None) -> str:
    global _last_npu_release
    opts = {
        "num_predict": num_predict,
        "temperature": 0.55,
        "top_p": 0.90,
        "repeat_penalty": 1.08,
        "repetition_penalty": 1.08,
        "stop": [chr(10) + "." + chr(10), "." + chr(10) + ".", ". . .", "I'm here. I'm here"],
        "is_cot": is_cot,
    }
    if options:
        opts.update(options)

    effective_num_predict = opts.get("num_predict", num_predict)
    timeout_s = 240 if effective_num_predict <= 320 else 900
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
                            "options": opts,
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


async def push_message_to_session(session_id: str, message: str, mode: str = "proactive", save_to_db: bool = True):
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
    if save_to_db:
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
                push_message_to_session, active_connections, _call_npu_once, scrub_reply,
                is_guest_session=lambda sid: session_guest_status.get(sid, False),
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
                if session_has_summary(sid):
                    continue

                retry_count = 0
                if sid in _summarize_jobs:
                    job = _summarize_jobs[sid]
                    status = job.get("status", "") if isinstance(job, dict) else str(job)
                    if status == "in_progress":
                        continue
                    if status.startswith("error"):
                        retry_count = job.get("retry_count", 0) if isinstance(job, dict) else 1
                        if retry_count >= 3:
                            continue  # Max 3 retries reached
                        backoff = (2 ** retry_count) * 600
                        last_updated = job.get("updated_at", 0) if isinstance(job, dict) else 0
                        if now - last_updated < backoff:
                            continue

                if count_turns(sid) < SUMMARY_MIN_TURNS:
                    continue
                idle_h = int((now - last_active) / 3600)
                print(f"[summary_loop] session '{sid}' idle {idle_h}h with no "
                      f"summary — queuing summarization (attempt {retry_count + 1})")
                create_background_task(_background_summarize(sid, force=False, retry_count=retry_count))
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


_summarize_jobs: dict[str, dict] = {}


async def _background_summarize(session_id: str, force: bool, known_people: set | None = None, retry_count: int = 0):
    _summarize_jobs[session_id] = {"status": "in_progress", "updated_at": time.time(), "retry_count": retry_count}
    try:
        if known_people is None:
            known_people = await asyncio.to_thread(get_known_people)
        is_guest = session_guest_status.get(session_id, False)
        await summarize_session(session_id, force=force, npu_lock=npu_lock, known_people=known_people, is_guest=is_guest)
        _summarize_jobs.pop(session_id, None)
    except Exception as e:
        _summarize_jobs[session_id] = {"status": f"error: {e}", "updated_at": time.time(), "retry_count": retry_count + 1}
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
# Immediate disable without restart: main re-checks key_state.json (in personal_data or next to
# this file) whenever its mtime changes — {"disabled": {"guest-1": true}}
# rejects that label on the very next request. Key MATERIAL rotation still
# requires a restart (imported once).
_personal_key_state = PERSONAL_DATA_DIR / "key_state.json"
_root_key_state = Path(__file__).parent / "key_state.json"
KEY_STATE_PATH = str(_personal_key_state if _personal_key_state.exists() else _root_key_state)
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


def _trim_to_boundary(text: str, max_chars: int) -> str:
    """Trims text to complete chunk or paragraph boundaries (\n---\n or \n\n) without slicing sentences."""
    if len(text) <= max_chars:
        return text
    candidates = [m.start() for m in re.finditer(r"\n---\n|\n\n", text[:max_chars + 1])]
    if candidates:
        return text[:candidates[-1]].rstrip()
    newline_pos = text[:max_chars + 1].rfind("\n")
    if newline_pos > 0:
        return text[:newline_pos].rstrip()
    return text[:max_chars].rstrip()


def _format_history_turns(history: list[dict]) -> list[dict]:
    """
    Formats history turns for prompt inclusion:
    - User turns with >= 1800s gap since previous turn are prefixed with '[after {human_gap(gap_sec)} gap] '.
    - Strict anti-leak: NEVER annotate assistant turns, and strip any leaked time tags from assistant turns.
    - Strips DB metadata (like timestamp) so only {"role", "content"} are sent to LLM.
    """
    formatted = []
    prev_turn_ts = None
    for turn in history:
        role = turn.get("role")
        content = turn.get("content", "")
        turn_ts = parse_timestamp_safe(turn.get("timestamp"))

        annotated_content = content
        if role == "user":
            if prev_turn_ts is not None and turn_ts is not None:
                gap_sec = (turn_ts - prev_turn_ts).total_seconds()
                if gap_sec >= 1800 and not content.startswith("[after "):
                    gap_label = human_gap(gap_sec)
                    annotated_content = f"[after {gap_label} gap] {content}"
            if turn_ts is not None:
                prev_turn_ts = turn_ts
        elif role == "assistant":
            # Strict anti-leak: NEVER annotate assistant turns
            annotated_content = re.sub(r'\[now:[^\]]*\]\s*', '', annotated_content)
            annotated_content = re.sub(r'\[after[^\]]*gap\]\s*', '', annotated_content)
            annotated_content = re.sub(r'\[you last spoke:[^\]]*\]\s*', '', annotated_content)
            annotated_content = annotated_content.strip()
            if turn_ts is not None:
                prev_turn_ts = turn_ts

        formatted.append({"role": role, "content": annotated_content})
    return formatted


# ----------------------------------------------------------------------
# PHASE 6: INPUT UNDERSTANDING & MULTI-FRAGMENT TURN MERGER
# ----------------------------------------------------------------------
APP_NOISE_REACTIONS = [
    (re.compile(r"^(?:Liked|Loved|Disliked|Laughed at|Emphasized|Questioned)\s+[\"“].*?[\"”]$", re.IGNORECASE), "[reaction: liked]"),
    (re.compile(r"^Liked\s+a\s+message$", re.IGNORECASE), "[reaction: liked]"),
    (re.compile(r"^Loved\s+an?\s+(?:image|message|photo|video)$", re.IGNORECASE), "[reaction: loved]"),
    (re.compile(r"^Reacted\s+(.*?)\s+to\s+(?:your\s+message|[\"“].*?[\"”])$", re.IGNORECASE), r"[reaction: \1]"),
    (re.compile(r"^Unsent\s+a\s+message$", re.IGNORECASE), "[reaction: unsent]"),
    (re.compile(r"^(?:<Media omitted>|\[Photo\]|\[Video\]|\[Audio\]|\[Sticker\])$", re.IGNORECASE), "[media]"),
]

APP_NOISE_PREFIX_REGEX = re.compile(
    r"^(?:Liked|Loved|Disliked|Laughed at|Emphasized|Questioned|Reacted\s+.*?)\s+(?:a\s+message|an?\s+image|to\s+your\s+message|to\s+[\"“].*?[\"”]|[\"“].*?[\"”])[\r\n\s]+",
    re.IGNORECASE
)


def clean_user_input(text: str) -> str:
    """
    Detects app-noise patterns and converts them into compact signals or strips boilerplate.
    """
    if not text:
        return ""
    stripped = text.strip()

    # Check exact match for app reactions
    for pattern, repl in APP_NOISE_REACTIONS:
        if pattern.match(stripped):
            return pattern.sub(repl, stripped)

    # If text starts with an app reaction prefix followed by meaningful user text, strip the prefix
    m = APP_NOISE_PREFIX_REGEX.match(stripped)
    if m:
        remaining = stripped[m.end():].strip()
        if remaining:
            return remaining

    return stripped


def merge_user_fragments(*args, **kwargs):
    """
    Multi-Fragment User Message Merger (Phase 6):
    Merges rapid consecutive user messages (inter-message gap < threshold_seconds)
    into a single clean logical user turn and deduplicates them from history.

    Supports two calling conventions:
    1. merge_user_fragments(incoming_message: str, history: list[dict], threshold_seconds=120.0, now=None)
       -> (merged_message, deduplicated_history, fragment_count)
    2. merge_user_fragments(turns: list[dict], threshold_seconds=120.0)
       -> list[dict]
    """
    if not args and not kwargs:
        return "", [], 1

    first_arg = args[0] if args else kwargs.get("incoming_message") or kwargs.get("turns")

    # Convention 2: list of turns passed
    if isinstance(first_arg, list):
        turns = first_arg
        threshold_seconds = float(args[1] if len(args) > 1 else kwargs.get("threshold_seconds", 120.0))
        if not turns:
            return []

        merged_turns = []
        for turn in turns:
            role = turn.get("role")
            content = turn.get("content", "").strip()
            ts = turn.get("timestamp")

            if not merged_turns:
                merged_turns.append(dict(turn))
                continue

            last = merged_turns[-1]
            if role == "user" and last.get("role") == "user":
                # Check gap between timestamps
                gap = None
                t1 = parse_timestamp_safe(last.get("timestamp"))
                t2 = parse_timestamp_safe(ts)
                if t1 and t2:
                    gap = abs((t2 - t1).total_seconds())

                if gap is None or gap < threshold_seconds:
                    # Merge consecutive user turn
                    sep = "\n" if "\n" in content or "\n" in last["content"] else ". "
                    last["content"] = f"{last['content'].rstrip('.?!')}{sep}{content}"
                    continue

            merged_turns.append(dict(turn))
        return merged_turns

    # Convention 1: incoming_message (str) and history (list[dict])
    incoming_message = str(first_arg) if first_arg is not None else ""
    history = list(args[1] if len(args) > 1 else kwargs.get("history", []))
    threshold_seconds = float(args[2] if len(args) > 2 else kwargs.get("threshold_seconds", 120.0))
    now = args[3] if len(args) > 3 else kwargs.get("now", None)

    cur_dt = get_current_time(now)
    cur_epoch = cur_dt.astimezone(timezone.utc).timestamp()

    # Look backwards from end of history for consecutive user turns
    tail_user_turns = []
    inspect_epoch = cur_epoch

    for i in range(len(history) - 1, -1, -1):
        turn = history[i]
        if turn.get("role") != "user":
            break

        turn_ts = parse_timestamp_safe(turn.get("timestamp"))
        if turn_ts:
            turn_epoch = turn_ts.astimezone(timezone.utc).timestamp()
            gap = inspect_epoch - turn_epoch
            if gap < threshold_seconds:
                tail_user_turns.append((i, turn["content"]))
                inspect_epoch = turn_epoch
            else:
                break
        else:
            break

    if not tail_user_turns:
        return incoming_message, history, 1

    tail_user_turns.reverse()
    popped_indices = set(i for i, _ in tail_user_turns)
    remaining_history = [h for i, h in enumerate(history) if i not in popped_indices]

    fragments = [c.strip() for _, c in tail_user_turns if c.strip()]
    if incoming_message.strip():
        fragments.append(incoming_message.strip())

    sep = "\n" if any("\n" in f for f in fragments) else ". "
    merged_message = sep.join(f.rstrip(".?!") if sep == ". " else f for f in fragments)
    fragment_count = len(fragments)

    return merged_message, remaining_history, fragment_count


def extract_input_signals(
    message: str,
    history: list[dict] | None = None,
    gap_seconds: float = 0.0,
    time_bucket: str = "afternoon",
    fragment_count: int = 1
) -> dict:
    """
    Lightweight Input Signals (Phase 6):
    Extracts is_question, is_venting, fragment_count, needs_retrieval, time_bucket, and gap_seconds.
    """
    msg = (message or "").strip()
    msg_lower = msg.lower()

    # 1. is_question
    is_q = "?" in msg or bool(re.search(
        r"^(who|what|when|where|why|how|which|whose|is|are|am|do|does|did|can|could|would|should|will|won't|have|has|had|tell me|remember)\b",
        msg, re.IGNORECASE
    ))

    # 2. is_venting
    VENTING_LEXICON = {
        "hate", "pissed", "angry", "crying", "exhausted", "furious", "overwhelmed",
        "screwed", "hopeless", "depressed", "anxious", "panic", "broken", "terrible",
        "horrible", "awful", "done with", "cant take", "can't take", "ruined", "breakdown",
        "miserable", "hurts", "hurting", "fucking", "screaming"
    }
    PUSH_AWAY_LEXICON = (
        "leave me alone", "nobody cares", "you don't get it", "you dont get it",
        "you're just a bot", "youre just a bot", "stop talking", "shut up", "don't talk to me"
    )
    words = re.findall(r"\b\w+\b", msg_lower)
    has_vent_word = any(w in VENTING_LEXICON for w in words)
    has_push_away = any(p in msg_lower for p in PUSH_AWAY_LEXICON)
    is_long_rant = (len(words) >= 25 and has_vent_word)
    is_venting = has_vent_word or has_push_away or is_long_rant

    # 3. needs_retrieval from Phase 5 gate
    prelim_signals = {"is_question": is_q, "is_venting": is_venting}
    needs_ret = should_retrieve(message, signals=prelim_signals)

    return {
        "is_question": is_q,
        "is_venting": is_venting,
        "fragment_count": fragment_count,
        "needs_retrieval": needs_ret,
        "time_bucket": time_bucket,
        "gap_seconds": gap_seconds
    }


def _prepare_prompt_sync(session_id: str, prompt_message: str, measure_tokens=None,
                         history_override=None, is_guest: bool = False, num_predict: int = 250,
                         now_override: Optional[datetime] = None, last_user_ts_override: Optional[Any] = None):
    # Fetch recent history early so Laya routing has conversational context if needed
    history = (history_override if history_override is not None
               else get_recent_history(session_id))
    history = filter_clean_history(history, max_assistant_turns=3)

    # Phase 6: App noise clean & Multi-fragment merger
    cleaned_input = clean_user_input(prompt_message)
    merged_message, history, fragment_count = merge_user_fragments(
        cleaned_input, history, threshold_seconds=120.0, now=now_override
    )
    prompt_message = merged_message

    # Determine last_user_ts
    if last_user_ts_override is not None:
        last_user_ts = last_user_ts_override
    elif history_override is not None:
        user_turns = [h for h in history if h.get("role") == "user" and h.get("timestamp")]
        last_user_ts = user_turns[-1]["timestamp"] if user_turns else None
    else:
        last_user_ts = get_last_user_timestamp(session_id)

    cur_dt = get_current_time(now_override)
    t_bucket = get_time_bucket(cur_dt.hour)
    if last_user_ts:
        parsed_last = parse_timestamp_safe(last_user_ts)
        gap_sec = (cur_dt.astimezone(timezone.utc) - parsed_last).total_seconds() if parsed_last else 0.0
    else:
        gap_sec = 0.0

    # Phase 6: Input signals
    signals = extract_input_signals(
        message=prompt_message,
        history=history,
        gap_seconds=gap_sec,
        time_bucket=t_bucket,
        fragment_count=fragment_count
    )

    time_note_str = time_note(last_user_ts, now=now_override) if ENABLE_TIME_NOTE else ""

    # Phase 7: Fast CPU Orchestrator (Tone Reader, Smoothed Profile Mixer & Distress Override)
    orch = orchestrate_turn(session_id, prompt_message, signals=signals)
    raw_tone_scores = orch["raw_tone_scores"]
    smoothed_tone_scores = orch["smoothed_tone_scores"]
    distress_flag = orch["distress_flag"]
    chosen_profile = orch["chosen_profile"]
    tone_condition = orch["tone_condition"]
    plan_tone = orch["plan_tone"]

    # Phase 4 & 7: Dynamic Few-Shot Retrieval (Anti-Echo, Diversity Filtered, Tone Conditioned)
    few_shot_examples = []
    few_shot_block = ""
    if ENABLE_DYNAMIC_FEW_SHOT:
        few_shot_examples = retrieve_few_shot_examples(
            prompt_message, recent_history=history, k=4, tone_condition=tone_condition
        )
        few_shot_block = format_few_shot_block(few_shot_examples)

    # 1. Behavioral Mode & Personal Retrieval Need via Laya + Orchestrator
    if distress_flag:
        mode = "supportive"
        is_crisis = True
        laya_routing = {
            "mode": "supportive",
            "is_crisis": True,
            "mode_confidence": 1.0,
            "retrieval_decision": "no_personal_context_needed",
            "retrieval_confidence": 1.0,
        }
    else:
        laya_routing = detect_mode_detailed(prompt_message, recent_turns=history)
        mode = chosen_profile if ENABLE_LIGHT_ORCHESTRATOR else laya_routing["mode"]
        is_crisis = laya_routing.get("is_crisis", False)

    policy = get_retrieval_policy(mode)

    # Phase 4 & 7: Structured Positive Plan
    structured_plan = ""
    if ENABLE_STRUCTURED_PLAN:
        structured_plan = build_structured_plan(
            mode=mode,
            message=prompt_message,
            time_bucket=t_bucket,
            gap_sec=gap_sec,
            is_crisis=is_crisis,
            signals=signals,
            tone_override=plan_tone
        )

    # 2. Phase 5: Fast Retrieval Gate + Legacy bypass checks
    bypass_retrieval = (
        is_guest or
        (not signals["needs_retrieval"]) or
        should_bypass_personal_retrieval(prompt_message, laya_routing=laya_routing)
    )
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
            is_explicit_ref = any(
                kw in prompt_message.lower()
                for kw in ("book", "document", "technique", "therapy", "cbt", "act", "reference", "manual", "chapter", "quote", "according to")
            )
            if is_explicit_ref or (mode not in ("conversational", "supportive")):
                reference_context = search_reference_data(prompt_message, top_k=policy["reference_top_k"])

    session_summaries_text = ""
    referenced_session_context = ""
    user_name = "User" if is_guest else "Dhanush"

    if not is_guest:
        if not is_test_session(session_id):
            MEMORY_RECALL_KEYWORDS = (
                "remember", "talked about", "earlier", "last time", "yesterday", "we said", "we talked", "recall"
            )
            summaries_limit = policy["summaries_limit"]
            if summaries_limit == 0 and any(kw in prompt_message.lower() for kw in MEMORY_RECALL_KEYWORDS):
                summaries_limit = 3

            is_bot_feedback = any(p in prompt_message.lower() for p in ("u just started", "you just started", "going crazy", "werent there", "weren't there", "not helping"))
            summaries = [] if is_bot_feedback else get_recent_summaries(
                exclude_session_id=session_id, limit=summaries_limit, query=prompt_message
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

    # The style profile is the owner's personal behavioural template; guests
    # must not be shaped by it (or see its contents echoed back).
    style_profile = "" if is_guest else get_style_profile()

    current_session_summary = get_current_session_summary(session_id) or ""

    # 3. Assess context sufficiency via Laya typed decisions BEFORE attaching static links.
    # The judge must see EVERYTHING the model is allowed to answer from
    # (personal RAG, reference, session notes, a referenced session, and the
    # rolling summary of this session). Otherwise it calls the context
    # "insufficient" when the answer is only in the notes, and the abstain
    # instruction then contradicts the "use your notes" instruction.
    #
    # It is also only meaningful for factual lookups. On supportive/guidance
    # turns (and crisis turns) nothing "answers" the user, so the gate is
    # skipped and the generic soft sufficiency instruction applies.
    if is_crisis or mode in ("supportive", "guidance"):
        sufficiency_res = {
            "decision": "not_applicable",
            "confidence": 1.0,
            "probabilities": {},
            "latency_ms": 0.0,
            "used_fallback": True,
        }
        sufficiency_state = "not_applicable"
    else:
        combined_rag_for_sufficiency = "\n\n".join(
            part for part in (
                personal_context,
                reference_context,
                current_session_summary,
                referenced_session_context,
                session_summaries_text,
            ) if part
        ).strip()
        sufficiency_res = assess_context_sufficiency(prompt_message, combined_rag_for_sufficiency)
        sufficiency_state = sufficiency_res.get("decision") if combined_rag_for_sufficiency else "no_context"

    # Passed to get_system_prompt() as its own block. Previously it was appended
    # to personal_context, which wrapped it in the "style-only excerpts from other
    # people's chats, never copy" framing and stopped the no-context rule firing.
    important_links = get_important_links() if not is_guest else ""

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
        important_links=important_links,
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
    HARD_CAP_SAFETY_MARGIN_TOKENS = 64
    max_allowed_for_gen = NPU_HARD_TOKEN_LIMIT - num_predict - HARD_CAP_SAFETY_MARGIN_TOKENS
    base_target = _get_prompt_target_tokens() or (NPU_HARD_TOKEN_LIMIT - PROMPT_CEILING_MARGIN)
    target_tokens = min(base_target, max_allowed_for_gen)

    HARD_CAP_CHARS_PER_TOKEN = 1.5
    HARD_CAP_TARGET_CHARS = int(target_tokens * HARD_CAP_CHARS_PER_TOKEN)
    HISTORY_PROTECTED_FLOOR_TURNS = 20

    dynamic_parts = []
    if few_shot_block and ENABLE_DYNAMIC_FEW_SHOT:
        dynamic_parts.append(few_shot_block)
    if time_note_str and ENABLE_TIME_NOTE:
        dynamic_parts.append(time_note_str)
    if structured_plan and ENABLE_STRUCTURED_PLAN:
        dynamic_parts.append(structured_plan)
    dynamic_context_str = "\n\n".join(dynamic_parts)
    dynamic_system_msg = [{"role": "system", "content": dynamic_context_str}] if dynamic_context_str else []

    def _total_prompt_chars(sp: str, hist: list, pm: str) -> int:
        return len(sp) + sum(len(h.get("content", "")) for h in hist) + len(dynamic_context_str) + len(pm)

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
    budget_chars = HARD_CAP_TARGET_CHARS
    measured_tokens = None
    if (measure_tokens is not None and target_tokens is not None
            and (total_chars > HARD_CAP_TARGET_CHARS
                 or target_tokens < PROMPT_TARGET_TOKENS)):
        probe = (
            [{"role": "system", "content": system_prompt}]
            + _format_history_turns(history)
            + dynamic_system_msg
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
            reference_context = _trim_to_boundary(reference_context, keep)
        if overflow > 0 and session_summaries_text:
            session_summaries_text = _trim_to_boundary(session_summaries_text, max(0, len(session_summaries_text) - overflow))
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
            important_links=important_links,
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
        personal_context = _trim_to_boundary(personal_context, max(0, len(personal_context) - overflow))
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
            important_links=important_links,
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
                + _format_history_turns(history)
                + dynamic_system_msg
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
        if not is_crisis and mode not in ("supportive", "guidance"):
            combined_rag_for_sufficiency = "\n\n".join(
                part for part in (
                    personal_context,
                    reference_context,
                    current_session_summary,
                    referenced_session_context,
                    session_summaries_text,
                ) if part
            ).strip()
            sufficiency_res = assess_context_sufficiency(prompt_message, combined_rag_for_sufficiency)
            sufficiency_state = sufficiency_res.get("decision") if combined_rag_for_sufficiency else "no_context"
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
                important_links=important_links,
                current_session_summary=current_session_summary,
            )

        print(
            f"[prompt_trim] session '{session_id}' exceeded safety budget — "
            f"trimmed to fit. final size ~{total_chars} chars "
            f"(~{int(total_chars / HARD_CAP_CHARS_PER_TOKEN)} conservative tok est, "
            f"target was {budget_chars} chars, {len(history)} history turns remaining)"
        )

    formatted_history = _format_history_turns(history)

    messages = (
        [{"role": "system", "content": system_prompt}]
        + formatted_history
        + dynamic_system_msg
        + [{"role": "user", "content": prompt_message}]
    )

    return {
        "mode": mode,
        "messages": messages,
        "time_note": time_note_str,
        "few_shot_examples": few_shot_examples,
        "few_shot_block": few_shot_block,
        "structured_plan": structured_plan,
        "last_user_ts": last_user_ts,
        "signals": signals,
        "raw_tone_scores": raw_tone_scores,
        "smoothed_tone_scores": smoothed_tone_scores,
        "distress_flag": distress_flag,
        "chosen_profile": chosen_profile,
        "orchestration": orch,
        "context_lookup_attempted": context_lookup_attempted,
        "personal_context": personal_context,
        "reference_context": reference_context,
        "session_summaries_text": session_summaries_text,
        "referenced_session_context": referenced_session_context,
        "style_profile": style_profile,
        "trimmed": trimmed,
        "trim_details": trim_details,
        "final_total_chars": total_chars,
        "target_tokens": target_tokens,
        "measured_tokens": measured_tokens,
        "budget_chars": budget_chars,
        "laya_routing": laya_routing,
        "laya_sufficiency": sufficiency_res,
        "laya_rag_filtered_count": get_last_laya_filtered_count(),
        "sufficiency_state": sufficiency_state,
        "is_crisis": is_crisis,
        "is_guest": is_guest,
    }


async def generate_reply(session_id: str, prompt_message: str, history_message: str = None, is_guest: bool = False) -> tuple[str, str]:
    if history_message is None:
        history_message = prompt_message

    # Idempotency check: prevent duplicate sends within 10s window on identical prompt
    now_ts = time.time()
    cached = _IDEMPOTENCY_CACHE.get(session_id)
    if cached:
        c_msg, c_reply, c_ts = cached
        if c_msg.strip().lower() == prompt_message.strip().lower() and (now_ts - c_ts) < 10.0:
            print(f"[idempotency] Returning cached reply for duplicate prompt on session '{session_id}'")
            return c_reply, "cached"

    # Register any explicit user ban commands for this session
    session_bans = extract_and_register_user_bans(session_id, prompt_message)

    cur_dt = get_current_time()
    t_bucket = get_time_bucket(cur_dt.hour)
    last_ts = get_last_user_timestamp(session_id)
    g_sec = (cur_dt.astimezone(timezone.utc) - parse_timestamp_safe(last_ts)).total_seconds() if last_ts else 0.0

    prep = await asyncio.to_thread(
        _prepare_prompt_sync, session_id, prompt_message,
        _measure_prompt_tokens_sync, None, is_guest)

    mode = prep["mode"]
    is_crisis = prep.get("is_crisis", False)

    # Named sampling profile based on mode, user turn length, and positive time cues (Phase 2 & 3)
    profile = get_sampling_profile(mode, prompt_message, time_bucket=t_bucket, gap_sec=g_sec)
    if mode in ("guidance", "coding") and profile.get("num_predict", 160) < 320:
        profile["num_predict"] = 320

    # Fix 1: Let the model write its own opener by default (no pre-fill on first attempt).
    # Opener pre-fill is reserved as a fallback only if regeneration is required.
    recent_bot_replies = [h.get("content", "") for h in prep["messages"] if h.get("role") == "assistant"]
    profile.pop("prefill_opener", None)

    raw_reply = await _call_npu_once(prep["messages"], options=profile)
    bot_reply = await asyncio.to_thread(
        scrub_reply, raw_reply, session_id=session_id, mode=mode, is_crisis=is_crisis, prompt_message=prompt_message
    )

    # Post-generation validation & single-retry regeneration (Phase 2, Step 2 & Fix 1/3)
    is_valid, invalid_reason = validate_reply(bot_reply, recent_bot_replies, session_bans)
    if not is_valid:
        global regeneration_count
        regeneration_count += 1
        print(f"[decode_guard] Validation failed ({invalid_reason}) on session '{session_id}'. Regenerating once with fallback profile & opener...")
        fallback_profile = dict(profile)
        fallback_profile["temperature"] = round(min(1.0, fallback_profile.get("temperature", 0.55) + 0.15), 2)
        fallback_profile["repeat_penalty"] = 1.12
        fallback_profile["repetition_penalty"] = 1.12

        # Pick fallback opener by message type (Fix 1)
        from mode_router import detect_message_type
        msg_type = detect_message_type(prompt_message)
        fallback_opener = pick_opener(recent_bot_replies, profile=mode, message_type=msg_type)
        fallback_profile["prefill_opener"] = fallback_opener

        import random
        fallback_profile["rng_seed"] = random.randrange(2**31)
        try:
            resampled_raw = await _call_npu_once(prep["messages"], options=fallback_profile)
            bot_reply = await asyncio.to_thread(
                scrub_reply, resampled_raw, session_id=session_id, mode=mode, is_crisis=is_crisis, prompt_message=prompt_message
            )
            raw_reply = resampled_raw
            still_valid, still_reason = validate_reply(bot_reply, recent_bot_replies, session_bans)
            if not still_valid:
                print(f"[decode_guard] Fallback retry still failed ({still_reason}). Using clean grounded reaction.")
                if session_bans and any(b in prompt_message.lower() for b in session_bans):
                    bot_reply = "fair enough, my bad on that. what's on your mind?"
                else:
                    bot_reply = "fair enough, what's on your mind?"
        except Exception as e:
            print(f"[decode_guard] single-retry regeneration failed: {e}")
            bot_reply = "fair enough, what's on your mind?"

    # Repetition recovery: Only trigger if the reply contains an actual hardware token stutter
    # (the exact same 1–4 word sequence repeated 4+ times consecutively, e.g. "i'm, i'm, i'm, i'm"),
    # OR if the reply is 95%+ character-for-character identical to the immediately preceding assistant turn (and > 15 chars).
    last_assistant_turn = None
    for msg in reversed(prep["messages"]):
        if msg.get("role") == "assistant":
            last_assistant_turn = msg.get("content", "").strip()
            break

    is_stutter_or_loop = False
    stutter_match = re.search(r'(\b[\w\']+(?:\s+[\w\']+){0,3}\b)(?:[,.]\s*\1){3,}', bot_reply, re.IGNORECASE)
    if stutter_match:
        is_stutter_or_loop = True

    if not is_stutter_or_loop and last_assistant_turn and len(bot_reply) > 15:
        import difflib
        ratio = difflib.SequenceMatcher(None, bot_reply.strip().lower(), last_assistant_turn.lower()).ratio()
        if ratio >= 0.95:
            is_stutter_or_loop = True

    if is_stutter_or_loop:
        # If there is a clean prefix before the stutter, keep it if substantive
        if stutter_match and stutter_match.start() > 20:
            clean_prefix = bot_reply[:stutter_match.start()].rstrip(" ,.:;")
            if len(clean_prefix.split()) >= 3:
                bot_reply = clean_prefix
                is_stutter_or_loop = False

    if is_stutter_or_loop:
        print(f"[loop_recovery] session '{session_id}': re-sampling dynamically (reply was {bot_reply!r})")
        system_msg = prep["messages"][0] if prep["messages"] and prep["messages"][0].get("role") == "system" else {"role": "system", "content": prep.get("system_prompt", "")}
        recent_turns = [m for m in prep["messages"][1:] if m.get("role") in ("user", "assistant")][-6:]
        resample_messages = [system_msg] + recent_turns
        try:
            resampled_raw = await _call_npu_once(resample_messages, num_predict=256)
            bot_reply = await asyncio.to_thread(scrub_reply, resampled_raw, session_id=session_id, mode=mode, prompt_message=prompt_message)
            raw_reply = resampled_raw
        except Exception as e:
            print(f"[loop_recovery] re-sample failed: {e}")

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

    _IDEMPOTENCY_CACHE[session_id] = (prompt_message, bot_reply, time.time())
    return bot_reply, mode


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, key_info: dict = Depends(verify_api_key)):
    _enforce_session_scope(key_info, payload.session_id)
    is_guest = (key_info.get("label") != "primary" or key_info.get("session_id") is not None)
    prompt_message = payload.message
    history_message = payload.message

    if payload.attachments:
        for att in payload.attachments:
            data_b64 = att.get("data", "")
            if len(data_b64) > (MAX_UPLOAD_BYTES * 4 // 3):
                raise HTTPException(status_code=413, detail=f"Attachment '{att.get('name', 'file')}' exceeds 10MB limit")
        extracted_parts = await asyncio.gather(*[
            asyncio.to_thread(decode_app_attachment, att, user_message=payload.message, session_id=payload.session_id) for att in payload.attachments
        ])
        file_names = [att.get("name", "file") for att in payload.attachments]

        extracted_text = "\n\n".join(extracted_parts)
        prompt_message = f"{payload.message}\n\n{extracted_text}".strip()
        if any("[IMAGE CONTENT —" in p for p in extracted_parts):
            history_message = f"{extracted_text}\n{payload.message}".strip()
        else:
            excerpt = extracted_text[:500].strip()
            if excerpt:
                history_message = f"[Sent file(s): {', '.join(file_names)}\nExcerpt: {excerpt}] {payload.message}".strip()
            else:
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
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 10MB limit")
    extracted = await asyncio.to_thread(process_uploaded_file, file.filename, file_bytes, user_message=message, session_id=session_id)

    if message.strip():
        prompt_message = f"{message.strip()}\n\n{extracted}"
    else:
        prompt_message = extracted

    if "[IMAGE CONTENT —" in extracted:
        history_message = f"{extracted}\n{message.strip()}".strip()
    else:
        excerpt = extracted[:500].strip()
        if excerpt:
            history_message = f"[Sent a file: {file.filename}\nExcerpt: {excerpt}]"
        else:
            history_message = f"[Sent a file: {file.filename}]"
        if message.strip():
            history_message += f" {message.strip()}"

    bot_reply, mode = await generate_reply(session_id, prompt_message, history_message=history_message, is_guest=is_guest)
    return ChatResponse(reply=bot_reply, mode=mode)


@app.get("/history/{session_id}")
@app.get("/api/history/{session_id}")
async def history(session_id: str, limit: int = Query(default=200, ge=1, le=500), key_info: dict = Depends(verify_api_key)):
    _enforce_session_scope(key_info, session_id)
    turns = get_full_session_history(session_id, limit=limit)
    return {"session_id": session_id, "messages": turns}


@app.post("/summarize")
async def summarize(payload: SummarizeRequest, key_info: dict = Depends(verify_api_key)):
    session_id = payload.session_id
    _enforce_session_scope(key_info, session_id)

    job_info = _summarize_jobs.get(session_id)
    status = job_info.get("status") if isinstance(job_info, dict) else job_info
    if status == "in_progress":
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
    job_info = _summarize_jobs.get(session_id)
    status = job_info.get("status") if isinstance(job_info, dict) else job_info

    if status == "in_progress":
        return {"session_id": session_id, "status": "in_progress"}

    if status and status.startswith("error"):
        return {"session_id": session_id, "status": "error", "detail": status}

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
        push_message_to_session,
        active_connections,
        _call_npu_once,
        scrub_reply,
        is_guest_session=lambda sid: session_guest_status.get(sid, False),
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
        _measure_prompt_tokens_sync, history_override, False, COT_NUM_PREDICT)

    # Attach CoT instructions specifically to the final user prompt turn
    if prep["messages"] and prep["messages"][-1]["role"] == "user":
        prep["messages"][-1]["content"] += "\n\n" + COT_INSTRUCTION

    t0 = time.time()
    raw = await _call_npu_once(prep["messages"], num_predict=COT_NUM_PREDICT, is_cot=True)
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
            await push_message_to_session(session_id, bot_reply, mode=mode, save_to_db=False)
        except Exception as e:
            if msg_id:
                _mark_message_failed(session_id, msg_id)
            print(f"[ws] reply generation error in session '{session_id}': {e}")
            await push_message_to_session(
                session_id,
                "⚠️ An error occurred processing your request.",
                mode="error",
                save_to_db=False,
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
    drained_ids = set()
    for payload in queue:
        try:
            await websocket.send_json(payload)
            drained_ids.add(payload.get("id"))
            print(f"[ws] drained queued message {payload.get('id')} to session '{session_id}'")
        except Exception:
            break
    if drained_ids:
        pending_messages[session_id] = [m for m in pending_messages[session_id] if m.get("id") not in drained_ids]

    # 2. Main incoming message loop
    try:
        while True:
            try:
                raw_text = await websocket.receive_text()
            except WebSocketDisconnect:
                print(f"[ws] session '{session_id}' disconnected")
                break
            except RuntimeError as e:
                err_msg = str(e).lower()
                if "disconnect" in err_msg or "cannot call \"receive\"" in err_msg or "cannot call 'receive'" in err_msg:
                    print(f"[ws] session '{session_id}' disconnected (RuntimeError)")
                    break
                raise
            except Exception as e:
                err_msg = str(e).lower()
                if "disconnect" in err_msg or "cannot call \"receive\"" in err_msg or "cannot call 'receive'" in err_msg or "connection closed" in err_msg:
                    print(f"[ws] session '{session_id}' disconnected ({e})")
                    break
                print(f"[ws] receive error in session '{session_id}': {e}")
                break

            try:
                data = json.loads(raw_text)
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                print(f"[ws] invalid message format in session '{session_id}': {e}")
                continue

            if not isinstance(data, dict):
                print(f"[ws] non-dict payload received from session '{session_id}'")
                continue

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
                raw_message = data.get("message") or data.get("text", "")
                attachments = data.get("attachments")

                if msg_id:
                    await websocket.send_json({"type": "ack", "id": msg_id})

                    if _is_message_duplicate_or_inflight(session_id, msg_id):
                        print(f"[ws] duplicate or in-flight chat_message {msg_id} for session '{session_id}', skipping reprocess")
                        continue

                prompt_message = raw_message
                history_message = raw_message

                if attachments:
                    oversized = False
                    for att in attachments:
                        data_b64 = att.get("data", "")
                        if len(data_b64) > (MAX_UPLOAD_BYTES * 4 // 3):
                            oversized = True
                            break
                    if oversized:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Attachment exceeds 10MB limit",
                            "id": msg_id,
                        })
                        continue

                    try:
                        extracted_parts = await asyncio.gather(*[
                            asyncio.to_thread(decode_app_attachment, att, user_message=raw_message, session_id=session_id) for att in attachments
                        ])
                    except Exception as e:
                        print(f"[ws] attachment decode failed for {msg_id} in '{session_id}': {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": "Failed to decode attachment",
                            "id": msg_id,
                        })
                        continue

                    file_names = [att.get("name", "file") for att in attachments]
                    extracted_text = "\n\n".join(extracted_parts)
                    prompt_message = f"{raw_message}\n\n{extracted_text}".strip()
                    if any("[IMAGE CONTENT —" in p for p in extracted_parts):
                        history_message = f"{extracted_text}\n{raw_message}".strip()
                    else:
                        excerpt = extracted_text[:500].strip()
                        if excerpt:
                            history_message = f"[Sent file(s): {', '.join(file_names)}\nExcerpt: {excerpt}] {raw_message}".strip()
                        else:
                            history_message = f"[Sent file(s): {', '.join(file_names)}] {raw_message}".strip()

                if msg_id:
                    _mark_message_inflight(session_id, msg_id)

                create_background_task(
                    _process_ws_chat_message(session_id, prompt_message, history_message, msg_id=msg_id)
                )

    except (WebSocketDisconnect, RuntimeError):
        print(f"[ws] session '{session_id}' disconnected")
    except Exception as e:
        print(f"[ws] loop exception in session '{session_id}': {e}")
    finally:
        if active_connections.get(session_id) is websocket:
            del active_connections[session_id]
        if session_id in session_guest_status:
            del session_guest_status[session_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
