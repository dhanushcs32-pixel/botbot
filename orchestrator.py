"""
orchestrator.py — Dynamic Two-Pass Local Orchestrator for Bot Bot (Step 2.6)

- Pure structural fast-path gate (zero hardcoded check-in phrase lists)
- Ultra-compact flat JSON Pass 1 strategist + truncated-JSON salvage parser
- Strict on-demand retrieval gating (zero unsolicited personal chat leaks)
- Grounded, low-temperature dynamic NPU repair pass (never fabricates stories)
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from rag import is_trivial_message, SIGNOFF_REGEX

ENABLE_TWO_PASS_ORCHESTRATOR = os.environ.get("ENABLE_TWO_PASS_ORCHESTRATOR", "1").lower() in ("1", "true", "yes")

VALID_MODES = {"supportive", "guidance", "conversational"}
VALID_STANCES = {
    "ground_and_clarify",
    "candid_pushback",
    "practical_triage",
    "warm_validation",
    "casual_banter",
    "direct_answer",
}
VALID_DEPTHS = {"brief", "medium", "deep"}


def should_run_orchestrator(
    message: str,
    signals: Optional[Dict[str, Any]] = None,
    distress_flag: bool = False,
    is_crisis: bool = False,
    known_people: Optional[set] = None,
) -> bool:
    """
    Pure structural fast-path gate (no hardcoded phrase lists):
    - Always runs on crisis, distress, venting, or when a known person is named.
    - Skips Pass 1 on trivial reactions, signoffs, and ultra-short <= 3 word turns
      (even with '?' like 'U good?' or 'How r u?') so casual turns stay fast on 1 NPU pass.
    """
    if not ENABLE_TWO_PASS_ORCHESTRATOR:
        return False
    msg = (message or "").strip()
    if not msg:
        return False
    if is_crisis or distress_flag:
        return True
    if signals and signals.get("is_venting"):
        return True
    if SIGNOFF_REGEX.search(msg):
        return False
    if is_trivial_message(msg):
        return False

    if known_people:
        for p in known_people:
            if p and re.search(rf"\b{re.escape(p)}\b", msg, re.IGNORECASE):
                return True

    words = re.findall(r"\b\w+\b", msg.lower())
    if len(words) <= 3:
        return False

    return True


STRATEGIST_SYSTEM_PROMPT = """You are a hidden strategist for an offline AI friend texting Dhanush on his device.
Output ONLY one short, flat JSON object (no markdown, no extra text). Keep every string under 18 words.

RULES:
- The companion is an offline AI friend with NO physical body, errands, commute, or real-world outings. Never suggest inventing daily stories, people, or meetings.
- Never recommend confrontational, rude, or accusatory replies, and never recommend generic therapy clichés.
- Retrieval is STRICTLY ON-DEMAND (leave query fields "" unless truly needed):
  * "therapy_q": short psychology search terms ONLY if coping/grounding/reframing concepts from a therapy book will concretely help; else "".
  * "person" & "chat_q": set ONLY if Dhanush explicitly asks about or refers to a past conversation/note with someone; else "".
  * "session_q": set ONLY if Dhanush explicitly refers to an earlier conversation with you; else "".

Exact JSON schema:
{"mode":"supportive|guidance|conversational","stance":"ground_and_clarify|candid_pushback|practical_triage|warm_validation|casual_banter|direct_answer","depth":"brief|medium|deep","state":"2-5 word mood","angle":"1 short concrete sentence on how to reply warmly and plainly","therapy_q":"","person":"","chat_q":"","session_q":""}"""


def build_strategist_messages(
    user_message: str,
    recent_history: Optional[List[Dict[str, Any]]] = None,
    time_note_str: str = "",
    known_people: Optional[set] = None,
    is_crisis: bool = False,
    distress_flag: bool = False,
) -> List[Dict[str, str]]:
    """Builds the compact Pass 1 prompt for the NPU strategist."""
    parts = []
    if time_note_str:
        parts.append(f"Time: {time_note_str}")

    if known_people:
        people_list = ", ".join(sorted(p for p in known_people if p)[:35])
        if people_list:
            parts.append(f"Known archive names: {people_list}")

    if is_crisis or distress_flag:
        parts.append("Alert: Acute distress — prioritize steady, warm grounding.")

    if recent_history:
        hist_lines = []
        for turn in recent_history[-6:]:
            role = "Dhanush" if turn.get("role") == "user" else "Companion"
            content = (turn.get("content") or "").strip().replace("\n", " ")
            if len(content) > 200:
                content = content[:200].rstrip() + "..."
            if content:
                hist_lines.append(f"{role}: {content}")
        if hist_lines:
            parts.append("Recent turns:\n" + "\n".join(hist_lines))

    parts.append(f"Latest message: {user_message.strip()}")
    parts.append("Return ONLY the single-line JSON object.")

    return [
        {"role": "system", "content": STRATEGIST_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _default_strategy(
    fallback_mode: str = "conversational",
    is_crisis: bool = False,
    distress_flag: bool = False,
) -> Dict[str, Any]:
    mode = "supportive" if (is_crisis or distress_flag) else (
        fallback_mode if fallback_mode in VALID_MODES else "conversational"
    )
    stance = "ground_and_clarify" if mode == "supportive" else (
        "direct_answer" if mode == "guidance" else "casual_banter"
    )
    depth = "deep" if (is_crisis or distress_flag or mode in ("supportive", "guidance")) else "medium"
    return {
        "mode": mode,
        "emotional_state": "distressed" if (is_crisis or distress_flag) else "neutral",
        "subtext": "",
        "stance": stance,
        "depth": depth,
        "reply_angle": "",
        "retrieval": {
            "need_therapy_ref": bool(is_crisis or distress_flag),
            "therapy_query": "",
            "need_personal_chats": False,
            "personal_query": "",
            "target_person": "",
            "need_past_sessions": False,
            "session_query": "",
        },
        "parse_ok": False,
    }


def _salvage_truncated_json(text: str) -> Optional[dict]:
    """Recovers flat string/bool JSON key-value pairs if generation truncated before '}'."""
    if "{" not in text:
        return None
    body = text[text.find("{"):]
    found: Dict[str, Any] = {}
    for m in re.finditer(r'"([a-zA-Z0-9_]+)"\s*:\s*"((?:[^"\\]|\\.)*)"', body):
        found[m.group(1)] = m.group(2).replace('\\"', '"').strip()
    for m in re.finditer(r'"([a-zA-Z0-9_]+)"\s*:\s*(true|false)', body, re.IGNORECASE):
        found[m.group(1)] = m.group(2).lower() == "true"
    if any(k in found for k in ("mode", "stance", "depth", "angle", "reply_angle")):
        return found
    return None


def _extract_json_object(raw_text: str) -> Optional[dict]:
    if not raw_text:
        return None
    cleaned = re.sub(r"(?s)<think>.*?</think>", "", raw_text).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                parsed = json.loads(fixed)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

    return _salvage_truncated_json(cleaned)


def parse_strategy_json(
    raw_text: str,
    fallback_mode: str = "conversational",
    is_crisis: bool = False,
    distress_flag: bool = False,
) -> Dict[str, Any]:
    """
    Parses and validates Pass 1 JSON output (supports both compact flat schema
    and nested retrieval schema) with safety invariants.
    """
    default = _default_strategy(fallback_mode=fallback_mode, is_crisis=is_crisis, distress_flag=distress_flag)
    parsed = _extract_json_object(raw_text)
    if not parsed:
        return default

    mode = str(parsed.get("mode", default["mode"])).strip().lower()
    if mode not in VALID_MODES:
        mode = default["mode"]

    stance = str(parsed.get("stance", default["stance"])).strip().lower()
    if stance not in VALID_STANCES:
        stance = default["stance"]

    depth = str(parsed.get("depth", default["depth"])).strip().lower()
    if depth not in VALID_DEPTHS:
        depth = default["depth"]

    if is_crisis or distress_flag:
        mode = "supportive"
        if stance == "casual_banter":
            stance = "ground_and_clarify"
        if depth == "brief":
            depth = "deep"

    raw_ret = parsed.get("retrieval") if isinstance(parsed.get("retrieval"), dict) else {}

    therapy_q = str(parsed.get("therapy_q") or raw_ret.get("therapy_query") or "").strip()[:200]
    person = str(parsed.get("person") or raw_ret.get("target_person") or "").strip()[:60]
    chat_q = str(parsed.get("chat_q") or raw_ret.get("personal_query") or "").strip()[:200]
    session_q = str(parsed.get("session_q") or raw_ret.get("session_query") or "").strip()[:200]

    need_therapy = bool(raw_ret.get("need_therapy_ref", False) or therapy_q)
    need_personal = bool(raw_ret.get("need_personal_chats", False) or chat_q or person)
    need_sessions = bool(raw_ret.get("need_past_sessions", False) or session_q)

    retrieval = {
        "need_therapy_ref": need_therapy,
        "therapy_query": therapy_q,
        "need_personal_chats": need_personal,
        "personal_query": chat_q,
        "target_person": person,
        "need_past_sessions": need_sessions,
        "session_query": session_q,
    }

    emo = str(parsed.get("state") or parsed.get("emotional_state") or "").strip()[:120]
    subtext = str(parsed.get("subtext") or "").strip()[:300]
    angle = str(parsed.get("angle") or parsed.get("reply_angle") or "").strip()[:400]

    return {
        "mode": mode,
        "emotional_state": emo,
        "subtext": subtext,
        "stance": stance,
        "depth": depth,
        "reply_angle": angle,
        "retrieval": retrieval,
        "parse_ok": True,
    }


def _strip_banned_terms(text: str, session_bans: Optional[set]) -> str:
    if not text or not session_bans:
        return text
    out = text
    for b in session_bans:
        if b and len(b) >= 2:
            out = re.sub(re.escape(b), "[omitted]", out, flags=re.IGNORECASE)
    return out


def format_strategy_brief(brief: Dict[str, Any], session_bans: Optional[set] = None) -> str:
    """Formats Pass 1's strategy JSON into a clean internal directive for Pass 2."""
    if not brief:
        return ""
    lines = [
        "[INTERNAL STRATEGY BRIEF — follow this direction naturally in a warm, plain friend voice; never mention this brief:]",
        f"- Stance: {brief.get('stance', 'direct_answer')} | Depth: {brief.get('depth', 'medium')}",
    ]
    emo = _strip_banned_terms(brief.get("emotional_state", ""), session_bans)
    subtext = _strip_banned_terms(brief.get("subtext", ""), session_bans)
    angle = _strip_banned_terms(brief.get("reply_angle", ""), session_bans)
    if emo:
        lines.append(f"- User mood: {emo}")
    if subtext:
        lines.append(f"- Subtext: {subtext}")
    if angle:
        lines.append(f"- Direction: {angle}")
    lines.append("- Grounding: Offline AI friend on his device only — never invent physical errands, outings, people, or unverified facts.")
    return "\n".join(lines)


def describe_rejection_reason(reason: str) -> str:
    """Positive, non-fabricating rewrite guidance."""
    if not reason:
        return "Reply plainly, honestly, and naturally to what Dhanush wrote."
    if reason in ("stock_phrase", "attractor_phrase"):
        return "Write a plain, honest, natural reply directly addressing what Dhanush wrote — avoid canned comfort slogans and do not invent fictional stories."
    if reason in ("physical_claim", "ungrounded_fabrication"):
        return "Respond strictly as an offline AI friend on his device with no physical body, errands, commute, or real-world outings; stick strictly to what is in the conversation."
    if reason.startswith("banned_phrase:"):
        return "Respond calmly without bringing up the topic Dhanush asked you to drop."
    if reason == "timestamp_leak":
        return "Output only plain conversational text with no bracketed tags or metadata."
    if reason in ("loop_or_stutter", "exact_duplicate_reply") or reason.startswith("high_consecutive_overlap"):
        return "Write a fresh, simple reply that moves the conversation forward naturally."
    if reason in ("empty_reply", "empty_or_too_short"):
        return "Write a complete, natural friend reply."
    return "Rewrite in a calm, grounded, natural friend voice without inventing facts."


def build_repair_messages(
    user_message: str,
    failed_draft: str,
    reason: str,
    strategy_brief: Optional[Dict[str, Any]] = None,
    recent_history: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    """
    Builds a grounded, non-fabricating NPU rewrite call.
    Omits poisoned drafts so the 4B model cannot echo clichés or hallucinations.
    """
    fix_guidance = describe_rejection_reason(reason)
    brief_note = ""
    if strategy_brief and strategy_brief.get("reply_angle"):
        brief_note = f"\nDirection: {strategy_brief['reply_angle']}"

    hist_snippet = ""
    if recent_history:
        lines = []
        for t in recent_history[-4:]:
            role = "Dhanush" if t.get("role") == "user" else "You"
            c = (t.get("content") or "").strip().replace("\n", " ")[:180]
            if c:
                lines.append(f"{role}: {c}")
        if lines:
            hist_snippet = "Recent context:\n" + "\n".join(lines) + "\n\n"

    sys_prompt = (
        "You are an offline AI friend texting Dhanush on his device. "
        "You do NOT have a physical body, daily errands, shopping trips, commutes, or real-world sightings. "
        "Never invent stories, fictional routines, people, or meetings. "
        "Be warm, relaxed, grounded, and genuine — never hostile or snappy. "
        "Match length to Dhanush's message (brief for a short check-in). "
        "Output ONLY your plain text reply."
    )

    omit_draft = (
        reason in ("stock_phrase", "attractor_phrase", "physical_claim", "ungrounded_fabrication", "loop_or_stutter")
        or reason.startswith("banned_phrase:")
    )
    draft_line = "" if omit_draft else f"Previous attempt to improve: {failed_draft}\n"

    user_prompt = (
        f"{hist_snippet}"
        f"Dhanush just texted: {user_message}\n"
        f"{draft_line}"
        f"Goal: {fix_guidance}"
        f"{brief_note}\n\n"
        "Write your plain, grounded friend reply now:"
    )
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
