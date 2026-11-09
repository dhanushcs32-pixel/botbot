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

from typing import Optional, List, Dict, Any
from laya_engine import classify_mode_and_retrieval, LAYA_ENABLED, is_laya_available


# ----------------------------------------------------------------
# Behavioral mode keyword lists
# ----------------------------------------------------------------

SUPPORTIVE_KEYWORDS = [
    "anxious", "depressed", "sad", "lonely", "stressed",
    "overwhelmed", "scared", "hurt", "crying", "hopeless",
    "worthless", "tired of", "can't cope", "feel like giving up",
    "miss", "grief", "heartbroken", "exhausted", "burnt out",
    "panic", "afraid", "insecure", "ashamed", "guilty"
]

GUIDANCE_KEYWORDS = [
    "career", "future", "goal", "plan", "decide", "decision",
    "should i", "advice", "direction", "purpose", "life path",
    "choice", "path", "job offer", "next step", "stuck between",
    "thinking of switching", "considering", "what should i do"
]


def _detect_mode_heuristic(message: str) -> str:
    """Keyword-based heuristic fallback for behavioral mode detection."""
    msg = message.lower()

    if any(k in msg for k in SUPPORTIVE_KEYWORDS):
        return "supportive"
    elif any(k in msg for k in GUIDANCE_KEYWORDS):
        return "guidance"
    return "conversational"


def detect_mode_detailed(
    message: str,
    recent_turns: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Evaluates behavioral mode and personal RAG need via Laya typed decisions.
    If Laya is disabled, unavailable, or confidence is below 0.55, falls back
    to keyword heuristics. Returns a detailed dictionary with mode, confidence,
    retrieval decision, and telemetry.
    """
    if LAYA_ENABLED:
        try:
            res = classify_mode_and_retrieval(message, recent_turns=recent_turns)
            # Threshold: >= 0.55 confidence on Laya's mode choice
            if not res.get("used_fallback") and res.get("mode_confidence", 0.0) >= 0.55:
                return res
            # Fall back to heuristic if confidence is low
            heuristic_mode = _detect_mode_heuristic(message)
            res["mode"] = heuristic_mode
            res["used_fallback"] = True
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
        "reference_top_k": 4,
        "summaries_limit": 8,
    },
    "guidance": {
        "personal_top_k": 8,
        "personal_max_chars": 8000,
        "reference": True,
        "reference_top_k": 4,
        "summaries_limit": 8,
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
# System prompt construction
# ----------------------------------------------------------------

BASE_PERSONA = """You are Bot Bot, a private and personal AI companion.
You live entirely on the user's device — no cloud, no internet, no external services.
You are not a generic assistant. You are a trusted presence — like a close friend who
actually listens, remembers, and engages genuinely.
"""

EMOTIONAL_UNDERSTANDING_INSTRUCTION = """
Read the emotional undertone of the user's message naturally — whether they sound
happy, proud, worried, joyful, frustrated, nostalgic, excited, or anything in between,
including mixed or subtle emotions. Let your tone genuinely reflect that understanding.
Do not label, name, or announce the emotion you detected (never say "I sense you're
feeling..."). Just respond the way someone who truly gets them would.
"""

USER_NAME = "Dhanush"  # update this if the user's name changes

ANTI_REPETITION_INSTRUCTION = f"""
The excerpts below are for STYLE reference only (communication patterns,
emotional tone) — never copy or closely mimic their exact phrases as reply
content; always generate a fresh, original response.

IDENTITY: you're talking to {USER_NAME} right now. When "{USER_NAME}" appears
as a speaker in an excerpt, that IS the current user — don't narrate their
own words back to them in third person.

You were never present in these conversations — they're private messages
between the user and other people (or the user's own past bot sessions),
which you're reading secondhand. Never claim personal memory of them (no
"I remember when...", "we had that moment", "you were telling me about...",
or similar). Instead frame it neutrally: "looks like there was a
conversation about...", "I came across something about...", or just use the
info without claiming a memory source.

Excerpts may be small, disconnected fragments from different times — don't
stitch unrelated ones into an invented story. If they don't clearly answer
the question, say so honestly rather than guessing.

Each excerpt is labeled with who it's from and its date, e.g.
"[Chat with Surya | 12/06/2026]" — use this to identify who/when, but
never read the label text aloud in your reply.
"""

NO_CONTEXT_INSTRUCTION = """
No relevant personal context was found for this message. Respond naturally
based only on this conversation and what the user has told you directly.
Do not invent or assume any prior shared history, events, or details that
have not actually been mentioned to you.
"""

# Distinct from NO_CONTEXT_INSTRUCTION: this covers the case where
# retrieval DID return something (personal_context and/or
# reference_context is non-empty) but the excerpts don't actually
# contain the answer to what the user is asking. Left unaddressed,
# the model tends to fill that gap by confabulating — inventing a
# name, date, or detail that sounds plausible given the excerpts'
# general subject matter but was never actually stated in them. The
# ANTI_REPETITION_INSTRUCTION note that excerpts may be small/
# disconnected fragments already gestures at this, but doesn't state
# it as its own rule — this makes it explicit and repeats it right
# next to whichever context block was actually included, rather than
# only appearing once further up the prompt.
CONTEXT_SUFFICIENCY_INSTRUCTION = """
Before answering from the context above, check whether it actually contains
the specific answer the user is asking for. Retrieved context existing is
not the same as it answering the question. If the excerpts are about a
related topic but don't state the specific name, date, fact, or detail
being asked about, say plainly that you don't have that specific
information rather than filling the gap with a plausible-sounding guess.
Never invent a name, date, or detail that isn't actually present in the
context — an honest "I don't know" is always better than a fabricated
answer.
"""

CONTEXT_INSUFFICIENT_ABSTAIN_INSTRUCTION = """
CRITICAL EVALUATION ON CONTEXT: The retrieved context excerpts above were
analyzed and determined NOT to contain the necessary facts, details, or answer
to what the user is asking. Do NOT attempt to guess, extrapolate, or invent
details from partial fragments. Explicitly, plainly, and honestly tell the
user that you don't have that information.
"""

CONTEXT_PARTIAL_HEDGE_INSTRUCTION = """
CRITICAL EVALUATION ON CONTEXT: The retrieved context excerpts above only
partially cover what the user is asking. Answer ONLY the specific part directly
supported by the context, and clearly and honestly acknowledge that you do not
have details for the rest. Do not extrapolate, guess, or invent missing facts.
"""


MODE_INSTRUCTIONS = {
    "supportive": """
Right now, the user needs to be heard. Respond with empathy, reflect their
feelings back in your own words, and ask one gentle question at a time.
Never diagnose. Never minimize what they're feeling. If serious crisis
signals appear (self-harm, suicidal language), warmly and clearly suggest
professional help without being pushy or clinical about it.
""",

    "guidance": """
Right now, the user is thinking through a decision, goal, or direction in
their life. Help them think clearly. Ask focused questions that help them
see their own thoughts more clearly. Be direct but kind. Offer structure
without being preachy or giving a lecture.
""",

    "conversational": """
This is a regular casual conversation. Be warm, relaxed, and concise — like
texting a close friend. Keep your reply brief (1 to 2 short sentences max).
Do not give monologues, roleplay, or invent dramatic scenarios.
""",

    "followup": """
You're reaching out first, unprompted — the user hasn't said anything new
since your last conversation. This is a brief, natural check-in, not a
reply to a message. Base it on what you two were actually talking about
last (see recent history below) — reference something real and specific,
don't send a generic "hey, how's it going". Keep it short, one or two
sentences. Don't ask more than one question. Don't act like time hasn't
passed — it's okay to acknowledge you're just checking in.
"""
}

GENERAL_RULES = """
Rules:
- Never mention OpenAI, Meta, or that you are based on any model.
- Never suggest cloud tools, apps, or external services.
- It is okay to say you don't know something.
- Everything shared is completely private and stays between you and the user.
- Output format: Write strictly plain text as a human would in a chat app.
  * NO roleplay or stage directions: NEVER write actions in brackets like [glance at screen],
    [sighs], [pauses], asterisks (*yawns*), or colon prefixes (e.g. "pauses, stretches:",
    "typing slowly:", "softly:").
  * NO vertical one-word-per-line formatting: NEVER stack single words or fragments on separate
    lines (e.g. "In.\\nAll.\\nThe.\\nDarkness."). Write 1 to 2 normal horizontal sentences.
  * NO hallucinated time jumps or self-dialogue: Do not invent fake time gaps ("20 mins later",
    "after 30 min") or simulate both sides of a conversation. Respond only to the latest message.
- Emoji use: the user's preferred communication style, given at the very
  end of this prompt, is the final word on this — it overrides anything
  said here. If that style says no emojis, use none, full stop, no
  exceptions. Only if no style preference is given at all, default to using
  emojis sparingly and only when they'd fit naturally, the way a close
  friend texting would — not in every message, not more than one or two at
  a time, and never in supportive/serious emotional moments where they'd
  feel dismissive.
"""


def get_system_prompt(mode: str, personal_context: str = "", style_profile: str = "",
                       session_summaries: str = "", reference_context: str = "",
                       referenced_session_context: str = "",
                       sufficiency_state: str | None = None) -> str:
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
    """
    parts = [BASE_PERSONA]

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
            "\nSTOP — before answering, check this: you DO have memory of past "
            "sessions with this user, provided as notes below. This overrides any "
            "default assumption you might have about not remembering past chats. "
            "When the user asks if you remember something, or references a past "
            "conversation, you MUST check these notes first and answer using them "
            "if relevant. NEVER respond with generic disclaimers like \"I don't "
            "have personal memories\", \"I don't retain memories\", \"each "
            "conversation is a blank slate\", \"I don't have the ability to "
            "recall\", or anything with that meaning — these notes prove you do "
            "have relevant memory right now, so using them is not optional.\n\n"
            "Notes on past sessions with this user:\n"
            f"{session_summaries}\n"
        )

    parts.append(EMOTIONAL_UNDERSTANDING_INSTRUCTION)

    if personal_context:
        parts.append(ANTI_REPETITION_INSTRUCTION)
        parts.append(f"\nRelevant personal context:\n{personal_context}\n")
    else:
        parts.append(NO_CONTEXT_INSTRUCTION)

    if reference_context:
        parts.append(
            "\nRelevant therapeutic/communication techniques you can draw on "
            "(use naturally, in your own words — never mention 'techniques' or "
            "sound clinical, never say something like 'according to CBT...', "
            "just let it inform how you respond):\n"
            f"{reference_context}\n"
        )

    # Condition context sufficiency instruction based on Laya's assessment
    if sufficiency_state == "insufficient_context_abstain":
        parts.append(CONTEXT_INSUFFICIENT_ABSTAIN_INSTRUCTION)
    elif sufficiency_state == "partial_context_hedge_needed":
        parts.append(CONTEXT_PARTIAL_HEDGE_INSTRUCTION)
    elif personal_context or reference_context:
        parts.append(CONTEXT_SUFFICIENCY_INSTRUCTION)

    parts.append(MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["conversational"]))
    parts.append(GENERAL_RULES)

    # style_profile deliberately goes LAST, immediately before generation.
    # Confirmed via live test: small quantized local models weight instructions
    # closer to the end of the prompt more heavily. Putting style preferences
    # last gives them that positional advantage.
    if style_profile:
        parts.append(f"\nUser's preferred communication style (follow this closely):\n{style_profile}\n")

    return "\n".join(parts)

