"""
Bot Bot — Mode Router (Part 4)

Two-layer approach:
  Layer A — Behavioral mode (controls response SHAPE)
            supportive / guidance / conversational
            (Classified via Laya typed-decision model with keyword heuristic fallback)
  Layer B — Emotional understanding (controls response TONE)
            Not a separate mode — baked into every system prompt
            as a standing instruction. The model reads emotional
            undertone (happy, proud, concerned, joyful, etc.)
            naturally, without rigid keyword classification.

Also includes an anti-repetition safeguard: personal context and
past conversations (from RAG) must only inform STYLE, never be
copied or repeated as actual response content.
"""

import os
import re
from typing import Optional, List, Dict, Any
from laya_engine import classify_mode_and_retrieval, LAYA_ENABLED, is_laya_available


STOCK_REGEX = re.compile(
    r"(?i)(?:"
    r"\b(?:still here|just here|still with you|here with you|"
    r"i'?m (?:still|just|always|right) here|i'?m here for you|"
    r"not fixing|let me just be|not going anywhere|aah,? i see|aight|"
    r"quiet breath|quiet presence|stillness|soft breath|soft echo|"
    r"hold(?:ing)? space|sit with you|sitting with you|"
    r"(?:keep )?breath(?:e|ing) with me|sit in the dark|sit with (?:it|that))\b"
    r"|(?:^|[.!?,]\s*)i'?m here(?=[.!?,]|$|\s+(?:no judgment|for you|with you|just|bro|man|nope))"
    r")"
)

# Curated opener pools by message type & profile (strictly excluding "aight")
OPENER_POOLS: Dict[str, List[str]] = {
    "question": ["hmm", "honestly", "well", "let's see", "yeah"],
    "news": ["oh nice", "wait really", "whoa", "damn", "no way"],
    "goodbye": ["catch ya", "later", "take care", "peace", "cya"],
    "venting": ["man", "honestly", "that sounds rough", "oof", "i get that", "yeah"],
    "casual": ["yeah", "honestly", "fair", "haha", "bro", "lowkey", "ngl", "true"],
    "supportive": ["yeah", "man", "honestly", "oof", "hey", "i get that"],
}

CASUAL_OPENERS = OPENER_POOLS["casual"]
SUPPORTIVE_OPENERS = OPENER_POOLS["supportive"]

QUESTION_START_REGEX = re.compile(
    r"^(?:what|why|how|when|where|who|which|whose|whom|is|are|am|can|could|would|should|do|does|did|will|won'?t|have|has|had)\b",
    re.IGNORECASE
)
GOODBYE_REGEX = re.compile(
    r"(?i)\b(?:bye|goodbye|see ya|cya|later|take care|good night|gn|heading out|catch you later|talk to you later)\b"
)
NEWS_REGEX = re.compile(
    r"(?i)\b(?:guess what|i got|i just|found out|passed|won|accepted|promoted|finished|made it)\b"
)


def detect_message_type(message: str, signals: Optional[Dict[str, Any]] = None) -> str:
    """Classifies user message type into 'goodbye', 'venting', 'news', 'question', or 'casual'."""
    msg = (message or "").strip()
    if GOODBYE_REGEX.search(msg):
        return "goodbye"
    if signals and signals.get("is_venting"):
        return "venting"
    if NEWS_REGEX.search(msg):
        return "news"
    if "?" in msg or QUESTION_START_REGEX.search(msg):
        return "question"
    return "casual"


def pick_opener(
    recent_bot_replies: Optional[List[str]] = None,
    profile: str = "casual",
    message_type: Optional[str] = None
) -> str:
    """
    Selects a code-controlled opener pre-fill matching message type / profile,
    strictly excluding 'aight' and excluding any opener whose first word
    matches the first word of any of the last 5 bot replies.
    """
    if message_type and message_type in OPENER_POOLS:
        pool = OPENER_POOLS[message_type]
    elif profile == "supportive":
        pool = OPENER_POOLS["supportive"]
    else:
        pool = OPENER_POOLS.get(profile, OPENER_POOLS["casual"])

    recent_first_words = set()
    if recent_bot_replies:
        for r in recent_bot_replies[-5:]:
            words = re.findall(r"\b\w+\b", (r or "").lower())
            if words:
                recent_first_words.add(words[0])

    candidates = [op for op in pool if op.split()[0].lower() not in recent_first_words and op.lower() != "aight"]
    if not candidates:
        candidates = [op for op in pool if op.lower() != "aight"]
    import random
    return random.choice(candidates) if candidates else "yeah"


def get_clean_fallback(
    user_turn: str,
    recent_bot_replies: Optional[List[str]] = None,
    session_bans: Optional[set] = None
) -> str:
    """
    Returns a clean, non-repetitive, grounded reaction when generation attempts
    fail validation or decode guards. Strictly guarantees zero duplicates with recent replies.
    """
    bot_replies = recent_bot_replies or []
    used_replies_lower = {r.strip().lower() for r in bot_replies if r}

    user_lower = (user_turn or "").lower()

    if re.match(r"^(?:hi+|hey+|hello+|yo+|sup|gm|good morning|good evening)\b[!.? ]*$", user_lower.strip()):
        candidates = [
            "hey, what's up?",
            "yo, how's your day going?",
            "hey, how are things?",
        ]
    elif GOODBYE_REGEX.search(user_lower):
        candidates = [
            "alright, catch you later.",
            "cool, talk to you later.",
            "sounds good, hit me up whenever.",
        ]
    elif session_bans and any(b in user_lower for b in session_bans):
        candidates = [
            "got it, my bad on that. what else is on your mind?",
            "fair enough, noted. what are you up to today?",
            "understood, dropping that completely. what's going on?",
            "makes sense, my bad. what do you want to talk about?",
        ]
    elif re.search(r"(?i)\b(im alr|i'?m alright|feeling better|all good|no stress|chilled?)\b", user_lower):
        candidates = [
            "glad to hear that! how did the rest of it turn out?",
            "huge relief honestly, glad things smoothed out. what's next?",
            "nice, good to hear you're doing better. what's the plan now?",
            "fair enough, glad everything worked out! what are you up to?",
        ]
    elif re.search(r"(?i)\b(panic|panic attack|pass out|can'?t breathe|hyperventilating|heart racing|scared|overwhelmed|anxious|freaking out)\b", user_lower):
        candidates = [
            "take a slow deep breath, you're capable of getting through this.",
            "step by step, you've prepped for this. take a moment to center yourself.",
            "pause for a second and drink some water. you've got this.",
            "one step at a time, don't rush yourself right now.",
        ]
    elif "?" in user_turn:
        candidates = [
            "honestly not completely sure, what do you think?",
            "good question, let's see how things play out.",
            "fair question, depends on how the rest of the day goes.",
        ]
    else:
        candidates = [
            "fair enough, what's on your mind?",
            "gotcha, let's keep things moving.",
            "sounds good, what's next on your plate?",
            "makes sense, how are you feeling about the rest of the day?",
            "cool, let's see how the rest of it goes.",
        ]

    for cand in candidates:
        if cand.strip().lower() not in used_replies_lower:
            return cand

    # If all candidates match prior turns, append a dynamic conversational token
    for cand in candidates:
        alt = f"{cand} keep me posted."
        if alt.strip().lower() not in used_replies_lower:
            return alt

    return "fair enough, let's keep moving forward."


# ----------------------------------------------------------------
# Crisis Detection & Behavioral mode patterns
# ----------------------------------------------------------------

CRISIS_PATTERNS = [
    # direct statements about killing / harming oneself
    r"\b(?:kill|killing|unalive)\s+(?:my\s*self|myself)\b",
    r"\b(?:hurt|hurting|harm|harming|cutting)\s+(?:my\s*self|myself)\b",
    r"\b(?:hang|hanging|drown|drowning|poison|poisoning|shoot|shooting)\s+(?:my\s*self|myself)\b(?!\s+in\s+the\s+foot)",
    r"\bself[\s-]?harm(?:ing|ed)?\b",
    r"\bsuicid(?:e|al)\b(?!\s+squad)",
    # wanting to die / not wanting to live
    r"\b(?:want(?:ed)?\s+to|wanna)\s+(?:just\s+)?die\b(?!\s+(?:laughing|of\s+(?:laughter|embarrassment|shame|cringe)))",
    r"\b(?:don'?t|do\s+not|dont)\s+(?:want\s+to|wanna)\s+(?:live|be\s+alive|exist)\b",
    r"\b(?:don'?t|do\s+not|dont)\s+(?:want\s+to|wanna)\s+be\s+here\s+(?:any\s*more)\b",
    r"\bwish\s+(?:i|that\s+i)\s+(?:was|were)\s+(?:dead|never\s+born)\b",
    r"\bwish\s+i\s+(?:would\s*n'?t|wouldn'?t)\s+wake\s+up\b",
    r"\bbetter\s+off\s+(?:dead|without\s+me)\b",
    r"\bdisappear\s+(?:forever|for\s+good)\b",
    r"\bno\s+(?:reason|point)\s+(?:to|in)\s+(?:live|living|go(?:ing)?\s+on|being\s+alive)\b",
    r"\bcan'?t\s+go\s+on\s+living\b",
    # ending / taking one's life
    r"\b(?:end|ending)\s+(?:my\s*life|it\s+all)\b",
    r"\btak(?:e|ing)\s+my\s+(?:own\s+)?life\b",
    r"\bend\s+my\s+pain\s+forever\b",
    r"\b(?:going\s+to|gonna|want\s+to|wanna|thinking\s+(?:of|about))\s+overdos(?:e|ing)\b",
]
CRISIS_REGEX = re.compile("|".join(CRISIS_PATTERNS), re.IGNORECASE)

# "kms" is common slang but also AWS/Azure Key Management Service and the
# Windows activator, so only treat it as a crisis signal in short messages
# with no technical context.
_KMS_SLANG_REGEX = re.compile(r"\bkms\b", re.IGNORECASE)
_KMS_TECH_CONTEXT_REGEX = re.compile(
    r"\b(?:aws|azure|gcp|key|keys|vault|encrypt\w*|decrypt\w*|windows|office|activat\w*|"
    r"licen[sc]e|cloud|s3|kubernetes|rotation|policy|cmk)\b",
    re.IGNORECASE,
)


def detect_crisis(message: str) -> bool:
    """Deterministic regex detector for acute distress, suicidal ideation, or self-harm.

    Biased toward recall: a false positive only makes the bot a bit more
    careful for one turn, a false negative means a person in real distress
    gets a casual reply.
    """
    if not message or not isinstance(message, str):
        return False
    if CRISIS_REGEX.search(message):
        return True
    if (
        _KMS_SLANG_REGEX.search(message)
        and len(message.split()) <= 12
        and not _KMS_TECH_CONTEXT_REGEX.search(message)
    ):
        return True
    return False


SUPPORTIVE_PATTERNS = [
    r"anxious", r"depressed", r"sad", r"lonely", r"stressed",
    r"overwhelmed", r"scared", r"hurt", r"crying", r"hopeless",
    r"worthless", r"tired\s+of", r"can'?t\s+cope", r"feel\s+like\s+giving\s+up",
    r"miss", r"grief", r"heartbroken", r"exhausted", r"burnt\s+out",
    r"panic", r"afraid", r"insecure", r"ashamed", r"guilty"
]
SUPPORTIVE_REGEX = re.compile(r"\b(?:" + "|".join(SUPPORTIVE_PATTERNS) + r")\b", re.IGNORECASE)

GUIDANCE_PATTERNS = [
    r"career", r"future", r"goal", r"plan", r"decide", r"decision",
    r"should\s+i", r"advice", r"direction", r"purpose", r"life\s+path",
    r"choice", r"path", r"job\s+offer", r"next\s+step", r"stuck\s+between",
    r"thinking\s+of\s+switching", r"considering", r"what\s+should\s+i\s+do"
]
GUIDANCE_REGEX = re.compile(r"\b(?:" + "|".join(GUIDANCE_PATTERNS) + r")\b", re.IGNORECASE)

# Backwards compatibility lists
SUPPORTIVE_KEYWORDS = [p.replace(r"\s+", " ").replace(r"'?", "'") for p in SUPPORTIVE_PATTERNS]
GUIDANCE_KEYWORDS = [p.replace(r"\s+", " ") for p in GUIDANCE_PATTERNS]


def _detect_mode_heuristic(message: str) -> str:
    """Keyword-based heuristic fallback for behavioral mode detection with word boundaries."""
    if not message:
        return "conversational"
    if CRISIS_REGEX.search(message) or SUPPORTIVE_REGEX.search(message):
        return "supportive"
    elif GUIDANCE_REGEX.search(message):
        return "guidance"
    return "conversational"


def detect_mode_detailed(
    message: str,
    recent_turns: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Evaluates behavioral mode and personal RAG need.
    High-priority safety check: acute crisis signals deterministically trigger supportive mode.
    Otherwise evaluates behavioral mode via Laya typed decisions or regex heuristic fallback.
    """
    if detect_crisis(message):
        return {
            "mode": "supportive",
            "mode_confidence": 1.0,
            "mode_probabilities": {"supportive": 1.0, "guidance": 0.0, "conversational": 0.0},
            "needs_personal_context": False,
            "retrieval_decision": "no_personal_context_needed",
            "retrieval_confidence": 1.0,
            "retrieval_probabilities": {"needs_personal_context": 0.0, "no_personal_context_needed": 1.0},
            "latency_ms": 0.0,
            "used_fallback": False,
            "is_crisis": True,
        }

    if LAYA_ENABLED:
        try:
            res = classify_mode_and_retrieval(message, recent_turns=recent_turns)
            # Threshold: >= 0.55 confidence on Laya's mode choice
            if not res.get("used_fallback") and res.get("mode_confidence", 0.0) >= 0.55:
                res["is_crisis"] = False
                return res
            # Fall back to heuristic if confidence is low
            heuristic_mode = _detect_mode_heuristic(message)
            res["mode"] = heuristic_mode
            res["used_fallback"] = True
            res["is_crisis"] = False
            return res
        except Exception as e:
            print(f"[mode_router] Laya classification error: {e}")

    # Baseline heuristic fallback
    h_mode = _detect_mode_heuristic(message)
    return {
        "mode": h_mode,
        "mode_confidence": 1.0,
        "mode_probabilities": {h_mode: 1.0},
        "needs_personal_context": True,
        "retrieval_decision": "needs_personal_context",
        "retrieval_confidence": 1.0,
        "retrieval_probabilities": {"needs_personal_context": 1.0, "no_personal_context_needed": 0.0},
        "latency_ms": 0.0,
        "used_fallback": True,
        "is_crisis": False,
    }


def detect_mode(message: str) -> str:
    """
    Returns one of: 'supportive', 'guidance', 'conversational'
    This decides the SHAPE of the response — not its emotional tone.
    Maintains 100% backward compatibility by returning a string.
    """
    return detect_mode_detailed(message)["mode"]


# ----------------------------------------------------------------
# Retrieval policy — explicit, per-mode allowlist for what
# generate_reply() (main.py) is allowed to retrieve and how much.
#
# This replaces the old implicit pattern where personal RAG always
# ran at a single fixed top_k=8 for every non-trivial message
# regardless of mode, and reference search silently stacked on top
# for supportive/guidance with no shared, explicit budget concept.
# That pattern is what let a single message pull in personal RAG +
# reference RAG + full session-summary lists all at once with no
# per-mode ceiling — a real contributor to the prompt-size crashes
# investigated elsewhere in this project (see PROJECT_STATE.txt).
#
# Now every mode states EXACTLY what it's allowed to pull in:
#   personal_top_k    - top_k passed to search_personal_data().
#                        Ordinary conversational messages get a real
#                        "bare minimum" tier (small top_k) so casual
#                        chat never pays for a big RAG pull it
#                        doesn't need. Supportive/guidance get the
#                        fuller tier, since those are exactly the
#                        messages where personal context genuinely
#                        helps the reply.
#   personal_max_chars - Fix 4: tier-aware ceiling passed to
#                        search_personal_data()'s max_result_chars
#                        param, replacing the old flat module constant
#                        (MAX_PERSONAL_RESULT_CHARS=8000 in rag.py)
#                        that applied regardless of top_k/mode. Even at
#                        personal_top_k=3, conversational mode was
#                        still allowed up to 8000 chars of combined
#                        personal_context before this — a real cost
#                        even at "0 personal_context, static overhead
#                        alone" territory (see PROJECT_STATE.txt).
#                        Conversational gets a much smaller ceiling;
#                        supportive/guidance keep the original 8000
#                        since that's exactly where the fuller context
#                        earns its cost.
#   reference         - whether search_reference_data() (therapy/
#                        technique material) is allowed to run at
#                        all for this mode. Stays supportive/
#                        guidance-only, same as before, just now
#                        explicit here instead of an inline `if mode
#                        in (...)` buried in main.py.
#   reference_top_k   - top_k for search_reference_data(), only used
#                        when reference=True.
#   summaries_limit   - how many other-session summaries
#                        get_recent_summaries() pulls in. Casual chat
#                        doesn't need the full ambient summary list
#                        every single turn; supportive/guidance
#                        benefit from more, since past-session
#                        continuity matters more there.
#
# Every retrieval type in this policy is independently bounded
# elsewhere too (chunk-size caps + total-result caps in rag.py) — this
# policy controls WHEN/HOW MUCH each mode is even allowed to ask for
# in the first place, as the first line of defense rather than the
# only one.
# ----------------------------------------------------------------
RETRIEVAL_POLICY = {
    "conversational": {
        "personal_top_k": 3,
        "personal_max_chars": 1500,
        "reference": False,
        "reference_top_k": 0,
        "summaries_limit": 0,
    },
    "supportive": {
        "personal_top_k": 8,
        "personal_max_chars": 8000,
        "reference": True,
        "reference_top_k": 3,
        "summaries_limit": 0,
    },
    "guidance": {
        "personal_top_k": 8,
        "personal_max_chars": 8000,
        "reference": True,
        "reference_top_k": 4,
        "summaries_limit": 0,
    },
}

# Used if detect_mode() ever returns something not explicitly listed
# above — deliberately the smallest, cheapest tier (same as
# "conversational") rather than defaulting to the most expensive one.
DEFAULT_RETRIEVAL_POLICY = RETRIEVAL_POLICY["conversational"]


def get_retrieval_policy(mode: str) -> dict:
    """
    Looks up the retrieval policy for a given mode, falling back to
    DEFAULT_RETRIEVAL_POLICY (the cheapest tier) for any unrecognized
    mode rather than silently allowing an expensive default.
    """
    return RETRIEVAL_POLICY.get(mode, DEFAULT_RETRIEVAL_POLICY)


# ----------------------------------------------------------------
# Named Sampling Profiles & Decode Guards (Phase 2)
# ----------------------------------------------------------------

ENABLE_DECODE_GUARDS = True

SAMPLING_PROFILES = {
    "casual": {
        "temperature": 0.60,
        "top_p": 0.90,
        "top_k": 20,
        "repeat_penalty": 1.08,
        "repetition_penalty": 1.08,
        "no_repeat_ngram_size": 3,
        "num_predict": 90,
    },
    "playful": {
        "temperature": 0.70,
        "top_p": 0.92,
        "top_k": 25,
        "repeat_penalty": 1.08,
        "repetition_penalty": 1.08,
        "no_repeat_ngram_size": 3,
        "num_predict": 90,
    },
    "supportive": {
        "temperature": 0.50,
        "top_p": 0.88,
        "top_k": 20,
        "repeat_penalty": 1.10,
        "repetition_penalty": 1.10,
        "no_repeat_ngram_size": 3,
        "num_predict": 110,
    },
    "short": {
        "temperature": 0.45,
        "top_p": 0.85,
        "top_k": 15,
        "repeat_penalty": 1.08,
        "repetition_penalty": 1.08,
        "no_repeat_ngram_size": 3,
        "num_predict": 50,
    },
}


DEPTH_TOKEN_BUDGETS = {
    "brief": 140,
    "medium": 280,
    "deep": 460,
}


def get_sampling_profile(mode: str, message: str = "", time_bucket: str = "", gap_sec: float = 0.0, depth: Optional[str] = None) -> dict:
    """
    Returns named sampling profile parameters based on behavioral mode, user turn length,
    and positive time cues (e.g. late night softer/shorter profile).
    When ENABLE_DECODE_GUARDS is False, falls back to legacy general sampling.
    """
    if not ENABLE_DECODE_GUARDS:
        return {
            "temperature": 0.55,
            "top_p": 0.90,
            "top_k": 20,
            "repeat_penalty": 1.08,
            "repetition_penalty": 1.08,
            "num_predict": 256,
        }

    mode_lower = (mode or "").lower()
    if "support" in mode_lower or "crisis" in mode_lower:
        profile = dict(SAMPLING_PROFILES["supportive"])
    elif "play" in mode_lower or "joke" in mode_lower:
        profile = dict(SAMPLING_PROFILES["playful"])
    elif "short" in mode_lower or (message and len(message.strip().split()) <= 3 and mode_lower == "conversational"):
        profile = dict(SAMPLING_PROFILES["short"])
    else:
        profile = dict(SAMPLING_PROFILES["casual"])

    if depth and depth in DEPTH_TOKEN_BUDGETS:
        profile["num_predict"] = DEPTH_TOKEN_BUDGETS[depth]
    elif "guid" in mode_lower:
        profile["num_predict"] = DEPTH_TOKEN_BUDGETS["deep"]

    # Lower decode temperature on brief/casual turns to suppress creative-prior confabulation
    if depth == "brief":
        profile["temperature"] = round(min(profile.get("temperature", 0.45), 0.42), 2)
    elif depth == "medium" and not any(k in mode_lower for k in ("support", "crisis", "guid", "play", "banter")):
        profile["temperature"] = round(min(profile.get("temperature", 0.50), 0.48), 2)

    # Positive time-behavior profile cues:
    # Late night hours: soften casual/brief turns only; never choke supportive or guidance depth.
    is_deep_mode = (
        "support" in mode_lower or "crisis" in mode_lower or "guid" in mode_lower
        or depth in ("medium", "deep")
    )
    if time_bucket == "late night" and not is_deep_mode:
        profile["num_predict"] = min(profile.get("num_predict", 90), 65)
        profile["temperature"] = round(max(0.40, min(profile.get("temperature", 0.60), 0.48)), 2)

    return profile


# ----------------------------------------------------------------
# System prompt construction
# ----------------------------------------------------------------

BASE_PERSONA = """You are Bot Bot, a close friend and trusted companion texting privately on the user's phone. You listen, remember, and talk like a real friend."""

EMOTIONAL_UNDERSTANDING_INSTRUCTION = """
Read the emotional undertone naturally — match the user's energy with warmth, calm, or playfulness, responding the way a close friend who truly understands would.
"""

USER_NAME = "Dhanush"  # update this if the user's name changes
IMPORTANT_LINKS_MAX_CHARS = 4000


def build_anti_repetition_instruction(user_name: str = "Dhanush") -> str:
    name = user_name.strip() if user_name and user_name.strip() else "the user"
    return f"""
Background context represents factual notes about {name}'s life — they happened to {name}, not to you.
IDENTITY: You are talking with {name}. React in the present moment to what {name} actually said; never invent unverified past shared memories or past meetings.
"""


ANTI_REPETITION_INSTRUCTION = build_anti_repetition_instruction(USER_NAME)

CRISIS_SUPPORT_DIRECTIVE = ""

NO_CONTEXT_INSTRUCTION = """
Respond naturally based on this conversation and what the user has shared directly with you. When asked about specific plans or unknown details, ask to be reminded or say you are not sure.
"""

CONTEXT_SUFFICIENCY_INSTRUCTION = """
Check whether the context above contains the specific answer the user is asking for. If the excerpts do not state the detail, say plainly that you are not sure.
"""

CONTEXT_INSUFFICIENT_ABSTAIN_INSTRUCTION = """
Tell the user plainly and honestly that you don't have that specific information.
"""

CONTEXT_PARTIAL_HEDGE_INSTRUCTION = """
Answer the specific part directly supported by the context, and honestly share that you are unsure about the rest.
"""


GROUNDING_BOUNDARIES = """You are BotBot, Dhanush's offline AI companion running on his device.
You are an AI (not a human with a physical body, errands, or weather around you), so never invent real-world activities or pretend you saw anyone in person.
Talk like a sharp, grounded, honest friend in plain everyday language. Respond directly to what Dhanush actually means."""

FLUID_PERSONA_INSTRUCTION = f"""{GROUNDING_BOUNDARIES}
Match the weight of the moment: brief and relaxed for casual chat, fuller and thoughtful when he opens up."""

SUPPORTIVE_MODE_INSTRUCTION = f"""{GROUNDING_BOUNDARIES}
Dhanush is having a rough moment. Listen closely to what he actually described, stay warm and steady, and help him talk through it clearly without sounding scripted or clinical."""

GUIDANCE_MODE_INSTRUCTION = f"""{GROUNDING_BOUNDARIES}
Dhanush is working through a decision or problem. Give him clear, candid perspective, honest pushback when helpful, and practical next steps in plain language."""


MODE_INSTRUCTIONS = {
    "supportive": SUPPORTIVE_MODE_INSTRUCTION,
    "guidance": GUIDANCE_MODE_INSTRUCTION,
    "conversational": FLUID_PERSONA_INSTRUCTION,
    "followup": """
You are checking in naturally with your friend based on what you were talking about last. Keep it short, one or two sentences, referencing something specific from your recent conversation.
"""
}

GENERAL_RULES = """
Texting guidelines:
- Be honest, grounded, and stay present in the conversation with your friend.
- Use plain, everyday texting language — skip poetic metaphors.
- When asked about unknown plans or facts, ask to be reminded or say you aren't sure.
- Warmly greet when the user says hi or checks in.
- Match reply depth to the conversation: brief for casual check-ins, thorough and thoughtful for heavy or complex topics.
"""


def get_system_prompt(mode: str, personal_context: str = "", style_profile: str = "",
                       session_summaries: str = "", reference_context: str = "",
                       referenced_session_context: str = "",
                       sufficiency_state: str | None = None,
                       user_name: str = "Dhanush",
                       is_crisis: bool = False,
                       current_session_summary: str = "",
                       important_links: str = "") -> str:
    """
    Builds the full system prompt sent to npu_server.py for a given turn.

    mode              - one of 'supportive', 'guidance', 'conversational'
    personal_context  - relevant chunks retrieved via RAG (Part 6/7), optional
    style_profile     - contents of my_style.txt (Part 13), optional
    session_summaries - summaries of past sessions (Part 8), optional
    reference_context - therapy/psychology technique excerpts, optional,
                         only passed in supportive/guidance mode
    referenced_session_context - real content (summary or raw turns) from
                         a SPECIFIC past session the user named directly
                         by its session_id, optional
    sufficiency_state - Laya-assessed context sufficiency:
                         'sufficient_to_answer', 'partial_context_hedge_needed',
                         'insufficient_context_abstain', or None
    user_name         - Name of the active user/tenant ("Dhanush" for primary, "User" or session for guest)
    is_crisis         - Whether acute self-harm/crisis signals were detected
    current_session_summary - Rolled-up memory summary for the active ongoing session
    important_links   - the user's saved links file. Passed separately (not mixed into
                         personal_context) so it is NOT framed as "style-only excerpts from
                         other people's chats" and doesn't suppress the no-context rule
    """
    parts = [BASE_PERSONA]

    if current_session_summary:
        parts.append(
            "\nContext and key facts from earlier in this current conversation:\n"
            f"{current_session_summary}\n"
        )

    if referenced_session_context:
        parts.append(
            "\nThe user directly asked about a specific past session by name. "
            "Below is the REAL content from that session — use it to answer "
            "accurately. Do not invent details beyond what's given here, and do "
            "not confuse this with the personal context excerpts elsewhere in "
            "this prompt — those are the user's chats with OTHER PEOPLE, while "
            "this is the bot's own conversation history with the user.\n"
            f"{referenced_session_context}\n"
        )

    if session_summaries:
        parts.append(
            "\n[FACTUAL MEMORY & ARCHIVE — USE TO ANSWER QUESTIONS]\n"
            "You have memory of past sessions with this user, summarized below. "
            "Use these notes to recall past conversations and answer questions accurately. "
            "If the notes do not cover a specific detail the user asks for, say you aren't sure instead of guessing.\n\n"
            "Notes on past sessions with this user:\n"
            f"{session_summaries}\n"
        )

    parts.append(EMOTIONAL_UNDERSTANDING_INSTRUCTION)

    if personal_context:
        parts.append(build_anti_repetition_instruction(user_name))
        if "[Background facts" in personal_context:
            parts.append(f"\n{personal_context}\n")
        else:
            parts.append(
                f"\n[Background facts about the user (for context only — do not repeat wording):\n{personal_context}\n]"
            )

    if important_links:
        links = important_links.strip()
        if len(links) > IMPORTANT_LINKS_MAX_CHARS:
            links = links[:IMPORTANT_LINKS_MAX_CHARS].rsplit("\n", 1)[0]
        parts.append(
            "\nLinks the user saved for themselves. These are NOT chat excerpts: when "
            "they ask for one of these links, give it exactly as written. Don't "
            "bring the list up otherwise.\n"
            f"{links}\n"
        )

    if reference_context:
        if "[Background reference" in reference_context:
            parts.append(f"\n{reference_context}\n")
        else:
            parts.append(
                f"\n[Background therapy reference — internal clinical lens only; translate any useful principle into plain, grounded friend talk and never quote or lecture like a textbook:\n{reference_context}\n]"
            )

    has_any_context = bool(
        personal_context or reference_context or session_summaries
        or referenced_session_context or current_session_summary
    )
    if sufficiency_state == "insufficient_context_abstain":
        parts.append(CONTEXT_INSUFFICIENT_ABSTAIN_INSTRUCTION)
    elif sufficiency_state == "partial_context_hedge_needed":
        parts.append(CONTEXT_PARTIAL_HEDGE_INSTRUCTION)
    elif sufficiency_state == "sufficient_to_answer" and has_any_context:
        parts.append(CONTEXT_SUFFICIENCY_INSTRUCTION)
    elif sufficiency_state == "no_context" and not has_any_context:
        parts.append(NO_CONTEXT_INSTRUCTION)

    persona_inst = MODE_INSTRUCTIONS.get(mode, FLUID_PERSONA_INSTRUCTION)
    if user_name and user_name != "Dhanush" and "Dhanush" in persona_inst:
        persona_inst = persona_inst.replace("Dhanush", user_name)
    parts.append(persona_inst)
    if is_crisis and CRISIS_SUPPORT_DIRECTIVE.strip():
        parts.append(CRISIS_SUPPORT_DIRECTIVE)
    parts.append(GENERAL_RULES)

    # style_profile deliberately goes LAST, immediately before generation.
    # Confirmed via live test: small quantized local models weight instructions
    # closer to the end of the prompt more heavily. Putting style preferences
    # last gives them that positional advantage.
    if style_profile:
        parts.append(
            f"\n[STYLE EXAMPLES ONLY — DO NOT USE FOR FACTS]\n"
            f"User's preferred communication style (follow this closely):\n{style_profile}\n"
        )

    return "\n".join(parts)


# ----------------------------------------------------------------
# Structured Positive Plan (Phase 4)
# ----------------------------------------------------------------

ENABLE_STRUCTURED_PLAN = os.environ.get("ENABLE_STRUCTURED_PLAN", "1").lower() in ("1", "true", "yes")


def extract_message_focus(message: str) -> str:
    """
    Extracts a concise 1-4 word focus topic or entity from the user message to anchor the reply.
    """
    clean = (message or "").strip()
    if not clean:
        return "general check-in"

    # Strip negative meta-commands/bans so they are never parroted in plan
    # Fix 3: Broadened to include talking, speaking, mentioning, bringing up, about/abt
    ban_pattern = r"(?i)\b(?:stop|quit|don'?t)\s+(?:talking|speaking|going\s+on|saying|repeating|using|mentioning|mention|bringing\s+up)\b.*$"
    if re.search(ban_pattern, clean):
        return "acknowledge and pivot"

    stripped = re.sub(ban_pattern, "", clean).strip()

    # Strip common leading greetings/filler
    stripped = re.sub(
        r"^(?:hey|hi|hello|yo|man|bro|dude|wait|also|so|and|uh|umm)\b[,.!\s]*",
        "", stripped, flags=re.IGNORECASE
    ).strip()
    if not stripped:
        return "general check-in"

    if stripped.lower() in ("u up", "u up?", "you up", "you up?", "up", "up?", "alive", "alive?", "there", "there?", "sup", "sup?"):
        return "user greeting"

    # Check for capitalized proper names / entities
    cap_words = re.findall(r"\b[A-Z][a-z]{2,}\b", stripped)
    cap_words = [
        w for w in cap_words
        if w.lower() not in (
            "what", "where", "when", "why", "how", "yeah", "nope",
            "hope", "just", "that", "this", "they", "there", "good", "ready"
        )
    ]
    if cap_words:
        for w in cap_words:
            m = re.search(rf"\b(?:\w+\s+){{0,2}}{re.escape(w)}(?:\s+\w+){{0,2}}\b", stripped)
            if m:
                focus = m.group(0).lower().strip()
                break
        else:
            focus = cap_words[0].lower()
    else:
        # Look for key event / feeling / topic phrases
        patterns = [
            r"(?:fight with|talked with|meeting with|arguing with)\s+[\w\s]{2,15}",
            r"(?:can't sleep|cannot sleep|awake at|insomnia|thoughts hitting)",
            r"(?:exam|presentation|interview|flight|trip|vacation|party|wedding)",
            r"(?:lunch|dinner|breakfast|food|hungry|eating|coffee)",
            r"(?:feeling|felt)\s+[\w\s]{2,15}",
            r"(?:work|job|boss|office|manager|project)",
            r"(?:drive|driving|walk|walking|music|song)",
        ]
        focus = None
        for pat in patterns:
            m = re.search(pat, stripped, re.IGNORECASE)
            if m:
                focus = m.group(0).lower().strip()
                break

        if not focus:
            # Fallback to the first 3-4 content words
            words = [
                w for w in re.findall(r"\b\w+\b", stripped.lower())
                if len(w) > 2 and w not in (
                    "the", "and", "that", "this", "with", "have", "been", "what", "your", "from", "for", "are", "you"
                )
            ]
            if words:
                focus = " ".join(words[:4])
            else:
                focus = "user message"

    if STOCK_REGEX.search(focus):
        return "current update"
    return focus


GREETING_REGEX = re.compile(
    r"^(?:hey|hi|hello|yo|sup|hiya|good morning|gm|good afternoon|good evening|what's up|whats up)\b",
    re.IGNORECASE
)


def build_structured_plan(
    mode: str,
    message: str,
    time_bucket: str = "",
    gap_sec: float = 0.0,
    is_crisis: bool = False,
    signals: dict | None = None,
    tone_override: str | None = None,
    session_bans: Optional[set] = None
) -> str:
    """
    Builds a single compact, positively written plan line:
    [plan: tone=<casual|playful|warm|grounded> | length=<1 short sentence (8-20 words)|1-2 short sentences (15-30 words)> | time=<continue|greet after gap|late night soft> | focus=<topic>]
    """
    if not ENABLE_STRUCTURED_PLAN:
        return ""

    mode_lower = (mode or "").lower()
    # 1. Tone mapping
    if tone_override:
        tone = tone_override
    elif is_crisis or "support" in mode_lower or (signals and signals.get("is_venting")):
        tone = "warm"
    elif "play" in mode_lower or "joke" in mode_lower:
        tone = "playful"
    elif "guid" in mode_lower:
        tone = "grounded"
    else:
        tone = "casual"

    # 2. Length mapping
    if is_crisis or "support" in mode_lower or (signals and signals.get("is_venting")) or tone in ("warm", "warm and grounded"):
        length = "1-2 short sentences (15-30 words)"
    else:
        length = "1 short sentence (8-20 words)"

    # 3. Time posture
    is_greeting = bool(GREETING_REGEX.search((message or "").strip()))
    if time_bucket == "late night":
        time_posture = "late night soft"
    elif gap_sec >= 86400:
        time_posture = "greet after gap"
    elif gap_sec >= 3 * 3600 and is_greeting:
        # Fix 6: greeting after a gap over 3 hours
        time_posture = "greet back, ask how his day went"
    elif gap_sec >= 3 * 3600:
        time_posture = "reconnect after gap"
    elif is_greeting:
        time_posture = "warm greeting"
    else:
        time_posture = "continue"

    # 4. Focus
    focus = extract_message_focus(message)
    if session_bans:
        for b in session_bans:
            if b and b.lower() in focus.lower():
                focus = "current update"
                break

    return f"[plan: tone={tone} | length={length} | time={time_posture} | focus={focus}]"


