"""
tone_reader.py — Fast CPU Tone Reader & Orchestrator for BotBot (Phase 7)

Provides:
1. Fast (< 10ms) CPU tone analyzer combining lexical/punctuation markers,
   Phase 3/6 signals (is_question, is_venting, time_bucket, gap_seconds),
   and prototypical tone anchors.
2. Continuous scores in [0.0, 1.0] for:
   - venting: emotional stress, frustration, sadness, burnout, conflict, pushing bot away
   - joking: banter, teasing, slang, humor, playful exaggeration
   - practical: concrete questions, tech/work/study queries, scheduling, decision-making
   - short_casual: brief check-ins, greetings, reactions, late-night sleepy texts
3. Binary distress_flag: bool triggering on venting >= 0.65 or explicit distress markers.
4. Turn-smoothed EMA profile mixer (0.7 * prev + 0.3 * curr).
5. Hard distress override: immediately bypasses smoothing on distress_flag == True to force
   supportive profile and warm/grounded plan with zero playful leakage.
"""

import os
import re
from typing import Dict, Any, Optional

ENABLE_LIGHT_ORCHESTRATOR = os.environ.get("ENABLE_LIGHT_ORCHESTRATOR", "1").lower() in ("1", "true", "yes")

# Explicit distress and adversarial pushback triggers
DISTRESS_TRIGGERS = (
    "repeating yourself", "repeating urself", "not helping", "stressed",
    "idk what to do", "completely fried", "can't sleep", "cant sleep",
    "hate my life", "nobody cares", "leave me alone", "stop talking",
    "shut up", "you don't get it", "you dont get it", "you're just a bot",
    "youre just a bot", "crying rn", "breakdown", "overwhelmed", "hopeless",
    "furious", "dont talk to me", "don't talk to me", "wanna die", "want to die"
)

VENTING_LEXICON = {
    "hate", "pissed", "angry", "crying", "exhausted", "furious", "overwhelmed",
    "screwed", "hopeless", "depressed", "anxious", "panic", "broken", "terrible",
    "horrible", "awful", "done with", "cant take", "can't take", "ruined",
    "breakdown", "miserable", "fried", "stressed", "drained", "sad", "hurts",
    "pain", "dying", "failed", "failing", "sucks", "crying rn", "hurting"
}

JOKING_LEXICON = {
    "lol", "lmao", "rofl", "haha", "hahaha", "lmfao", "bruh", "bro", "dude",
    "jk", "kidding", "dead", "skull", "wild", "crazy", "hilarious", "silly",
    "bet", "fr", "frfr", "tf", "wtaf", "nah fr", "meme", "clowning"
}

PRACTICAL_PATTERNS = [
    r"\bhow\s+(?:do|can|to|would)\s+(?:i|we)\b",
    r"\bwhere\s+(?:is|are|can|did)\b",
    r"\bwhat\s+(?:time|is|are|does|did)\b",
    r"\bwhen\s+(?:is|are|does|did)\b",
    r"\bwhich\s+(?:one|switch|option|route)\b",
    r"\b(?:code|function|python|install|script|config|setup|fix|terminal|git)\b",
    r"\b(?:deadline|assignment|schedule|calendar|exam|flight|ticket|meeting|appointment)\b",
    r"\b(?:keyboard|tactile|linear|laptop|specs|monitor|buy|recommend)\b",
]

SHORT_CASUAL_PATTERNS = {
    "hey", "yo", "sup", "hi", "hello", "wbu", "hbu", "good morning", "good night",
    "gn", "gm", "alr", "okay", "ok", "k", "cool", "nice", "chilling", "nm",
    "nothing much", "just waking up", "just chilling", "sounds good", "same",
    "fine", "yeah", "yea", "yep", "sure", "got it"
}


DE_ESCALATION_REGEX = re.compile(
    r"(?i)\b(i'?m (?:alr|alright|good|fine|okay|ok) now|feeling better|no stress|chill|not much support needed|all good now|all good|no worries)\b"
)


def analyze_tone(message: str, signals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Fast (< 10ms) rule-and-lexical CPU tone analyzer.
    Returns normalized continuous scores in [0.0, 1.0] for venting, joking, practical,
    and short_casual, plus a binary distress_flag.
    """
    msg = (message or "").strip()
    msg_lower = msg.lower()
    words = re.findall(r"\b\w+\b", msg_lower)
    num_words = len(words)

    # 0. Check explicit recovery/de-escalation cues
    is_de_escalation = bool(DE_ESCALATION_REGEX.search(msg))

    # 1. Distress flag check
    has_distress_trigger = any(trig in msg_lower for trig in DISTRESS_TRIGGERS) and not is_de_escalation

    # 2. Venting score
    vent_matches = sum(1 for w in words if w in VENTING_LEXICON) if not is_de_escalation else 0
    vent_score = 0.0
    if not is_de_escalation:
        if vent_matches > 0:
            vent_score += min(0.60, vent_matches * 0.25)
        if signals and signals.get("is_venting"):
            vent_score += 0.35
        if has_distress_trigger:
            vent_score += 0.45
        if num_words >= 25 and vent_matches >= 1:
            vent_score += 0.20
        vent_score = min(1.0, max(0.0, vent_score))

    # Binary distress trigger
    distress_flag = bool((has_distress_trigger or vent_score >= 0.65) and not is_de_escalation)
    if distress_flag:
        vent_score = max(vent_score, 0.85)

    # 3. Joking score
    joke_matches = sum(1 for w in words if w in JOKING_LEXICON)
    joke_emojis = len(re.findall(r"[💀😂😭🤣]", msg))
    joke_score = 0.0
    if joke_matches > 0:
        joke_score += min(0.70, joke_matches * 0.25)
    if joke_emojis > 0:
        joke_score += min(0.30, joke_emojis * 0.15)
    if re.search(r"\b(?:look at this|you won't believe|no way)\b", msg_lower):
        joke_score += 0.20
    joke_score = min(1.0, max(0.0, joke_score))

    # If distress active, zero out joking
    if distress_flag:
        joke_score = 0.0

    # 4. Practical score
    pract_matches = sum(1 for pat in PRACTICAL_PATTERNS if re.search(pat, msg_lower))
    pract_score = 0.0
    if pract_matches > 0:
        pract_score += min(0.75, pract_matches * 0.30)
    if signals and signals.get("is_question"):
        pract_score += 0.25
    if "?" in msg and num_words >= 4:
        pract_score += 0.15
    pract_score = min(1.0, max(0.0, pract_score))

    # 5. Short Casual score
    casual_score = 0.0
    cleaned_norm = " ".join(words)
    CASUAL_GREETINGS = ("what's up", "whats up", "how r u", "how are you", "good morning", "good night", "hey", "yo", "sup", "hi", "hello")
    if cleaned_norm in SHORT_CASUAL_PATTERNS or msg_lower in SHORT_CASUAL_PATTERNS:
        casual_score += 0.85
    elif any(g in msg_lower for g in CASUAL_GREETINGS) and num_words <= 5:
        casual_score += 0.75
    elif num_words <= 3 and not (signals and signals.get("is_question")):
        casual_score += 0.65
    elif num_words <= 6:
        casual_score += 0.35

    if signals and signals.get("time_bucket") == "late night":
        casual_score += 0.20
    casual_score = min(1.0, max(0.0, casual_score))

    # If none scored prominently, default baseline casual distribution
    total = vent_score + joke_score + pract_score + casual_score
    if total < 0.1:
        casual_score = 0.50
        pract_score = 0.20

    return {
        "venting": round(vent_score, 3),
        "joking": round(joke_score, 3),
        "practical": round(pract_score, 3),
        "short_casual": round(casual_score, 3),
        "distress_flag": distress_flag,
        "is_de_escalation": is_de_escalation
    }


# Session tone tracking for EMA smoothing and multi-turn distress tracking
_SESSION_TONE_STATE: Dict[str, Dict[str, float]] = {}
_SESSION_DISTRESS_COUNT: Dict[str, int] = {}


def get_session_smoothed_tone(session_id: str) -> Dict[str, float]:
    """Returns the current smoothed tone dict for the session."""
    return dict(_SESSION_TONE_STATE.get(session_id, {
        "venting": 0.0,
        "joking": 0.0,
        "practical": 0.20,
        "short_casual": 0.50
    }))


def update_session_smoothed_tone(
    session_id: str,
    current_scores: Dict[str, float],
    distress_flag: bool = False
) -> Dict[str, float]:
    """
    Updates session smoothed tone state:
    - If distress_flag is True: BYPASSES smoothing completely and sets venting=1.0, joking=0.0.
    - If is_de_escalation:
        * Mild worry (distress_count < 2): instant reset to venting=0.0, short_casual=0.80.
        * Sustained high-severity pain (distress_count >= 2): gradual EMA decay (venting=0.5 * prev).
    - Otherwise: applies smoothed[k] = 0.7 * prev[k] + 0.3 * curr[k].
    """
    if current_scores.get("is_de_escalation"):
        prior_distress = _SESSION_DISTRESS_COUNT.get(session_id, 0)
        prev = get_session_smoothed_tone(session_id)
        prev_venting = prev.get("venting", 0.0)

        if prior_distress >= 2 and prev_venting >= 0.80:
            # Sustained high-severity pain: gradual EMA decay
            decayed_venting = round(0.50 * prev_venting, 4)
            _SESSION_DISTRESS_COUNT[session_id] = max(0, prior_distress - 1)
            decayed = {
                "venting": decayed_venting,
                "joking": 0.05,
                "practical": 0.30,
                "short_casual": round(max(0.0, 1.0 - decayed_venting - 0.35), 4)
            }
            _SESSION_TONE_STATE[session_id] = decayed
            return dict(decayed)
        else:
            # Mild worry or single episode: instant reset
            _SESSION_DISTRESS_COUNT[session_id] = 0
            reset_val = {
                "venting": 0.0,
                "joking": 0.0,
                "practical": 0.20,
                "short_casual": 0.80
            }
            _SESSION_TONE_STATE[session_id] = reset_val
            return dict(reset_val)

    if distress_flag:
        _SESSION_DISTRESS_COUNT[session_id] = _SESSION_DISTRESS_COUNT.get(session_id, 0) + 1
        override = {
            "venting": 1.0,
            "joking": 0.0,
            "practical": 0.0,
            "short_casual": 0.0
        }
        _SESSION_TONE_STATE[session_id] = override
        return dict(override)

    # Normal turn: naturally decrement distress counter if present
    if session_id in _SESSION_DISTRESS_COUNT and _SESSION_DISTRESS_COUNT[session_id] > 0:
        _SESSION_DISTRESS_COUNT[session_id] -= 1

    prev = get_session_smoothed_tone(session_id)
    smoothed = {}
    for k in ("venting", "joking", "practical", "short_casual"):
        p_val = prev.get(k, 0.25)
        c_val = current_scores.get(k, 0.25)
        smoothed[k] = round(0.7 * p_val + 0.3 * c_val, 4)

    _SESSION_TONE_STATE[session_id] = smoothed
    return dict(smoothed)


def reset_session_tone(session_id: str) -> None:
    """Resets tone state for a session."""
    _SESSION_TONE_STATE.pop(session_id, None)
    _SESSION_DISTRESS_COUNT.pop(session_id, None)


def orchestrate_turn(
    session_id: str,
    message: str,
    signals: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Orchestrates the turn:
    1. Runs fast CPU tone analysis.
    2. Updates turn-smoothed state (or executes hard distress override).
    3. Maps dominant tone to sampling profile and structured plan tone.
    """
    raw_scores = analyze_tone(message, signals)
    distress_flag = raw_scores["distress_flag"]

    if not ENABLE_LIGHT_ORCHESTRATOR:
        return {
            "raw_tone_scores": raw_scores,
            "smoothed_tone_scores": raw_scores,
            "distress_flag": distress_flag,
            "chosen_profile": "supportive" if distress_flag else "casual",
            "plan_tone": "warm and grounded" if distress_flag else "casual",
            "tone_condition": "supportive" if distress_flag else "casual"
        }

    smoothed = update_session_smoothed_tone(session_id, raw_scores, distress_flag=distress_flag)

    # Hard distress override takes immediate precedence
    if distress_flag:
        chosen_profile = "supportive"
        plan_tone = "warm and grounded"
        tone_condition = "supportive"
    else:
        # Map based on dominant smoothed tone
        v = smoothed.get("venting", 0.0)
        j = smoothed.get("joking", 0.0)
        p = smoothed.get("practical", 0.0)
        c = smoothed.get("short_casual", 0.0)

        words = re.findall(r"\b\w+\b", (message or "").lower())
        if v >= max(j, p, c) and v >= 0.40:
            chosen_profile = "supportive"
            plan_tone = "warm"
            tone_condition = "supportive"
        elif j >= max(v, p, c) and j >= 0.35:
            chosen_profile = "playful"
            plan_tone = "playful"
            tone_condition = "playful"
        elif len(words) <= 3 or (c >= max(v, j, p) and c >= 0.50):
            chosen_profile = "short"
            plan_tone = "casual"
            tone_condition = "short"
        else:
            chosen_profile = "casual"
            plan_tone = "grounded" if p >= 0.45 else "casual"
            tone_condition = "casual"

    return {
        "raw_tone_scores": raw_scores,
        "smoothed_tone_scores": smoothed,
        "distress_flag": distress_flag,
        "chosen_profile": chosen_profile,
        "plan_tone": plan_tone,
        "tone_condition": tone_condition
    }
