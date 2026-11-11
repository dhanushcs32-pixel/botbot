"""
Bot Bot — Semantic Style Check (Tier 2)

Catches AI-esque phrasing that AI_TELL_PHRASES (exact substring matching,
in main.py) can't: paraphrases and novel variants with zero word overlap
but the same underlying meaning, e.g. "I completely get where you're
coming from" vs. the literal phrase "I understand how you feel". Exact
matching is blind to this; embedding similarity isn't.

Deliberately reuses rag.py's get_embedding_fn() (all-MiniLM-L6-v2,
already lazy-loaded there for RAG search) rather than loading a second
model — this stays essentially free relative to a 45-90s NPU generation
call: embedding a handful of short sentences on CPU is single-digit-to-
low-double-digit milliseconds once the model is warm.

TIERED ENFORCEMENT (updated 2026-08-08 — see PROJECT_STATE.txt):
Every sentence above SIMILARITY_THRESHOLD is still logged, same as
before. But only sentences that are BOTH in an ENFORCE_CATEGORIES
category AND above the stricter ENFORCE_SIMILARITY_THRESHOLD are
actually auto-stripped from the reply, via strip_semantic_violations().

Enforcement is deliberately narrower than logging, not just stricter:
"ai_tell" and "formal_signoff" are near-never a legitimate part of a
good reply, so auto-removing them is low-risk. "unsolicited_advice" and
"hedging" are a different story — their seed phrases ("take things one
day at a time", "everyone's experience is different") sit close in
embedding space to genuinely good supportive-mode replies that
mode_router.py deliberately routes toward. Auto-stripping those risks
deleting a correct, appropriate reply, not just an AI-tell. They stay
log-only until real traffic shows a threshold that's actually safe.

check_semantic_violations() is unchanged in shape — still returns every
finding, still doesn't touch the text. strip_semantic_violations() is
the new entry point main.py's scrub_reply() should call instead: it
runs the same check, then removes only the enforce-eligible sentences
and returns (cleaned_text, findings) so logging stays exactly as
informative as before, including sentences that were flagged but NOT
removed.
"""

import re

# Seed phrases are full example SENTENCES, not short fragments like
# AI_TELL_PHRASES — embeddings compare meaning, not substrings, so a seed
# needs to actually read like a sentence for similarity scoring to be
# meaningful. Organized by category purely for logging/debugging
# readability; category isn't used in the similarity math itself.
SEED_PHRASES = {
    "ai_tell": [
        "I understand how you feel.",
        "That's a great question.",
        "I'd be happy to help with that.",
        "I hope this helps!",
        "It's important to note that everyone's situation is different.",
        "Let me know if you have any other questions.",
    ],
    "unsolicited_advice": [
        "It's important to remember that self-care matters.",
        "Try to stay positive during difficult times.",
        "Remember to take things one day at a time.",
        "Make sure to prioritize your wellbeing.",
    ],
    "formal_signoff": [
        "Please don't hesitate to reach out if you need anything else.",
        "Wishing you all the best going forward.",
        "Feel free to reach out anytime.",
    ],
    "hedging": [
        "I could be wrong, but it seems like this might be the case.",
        "This is just my perspective, so take it with a grain of salt.",
        "Everyone's experience is different, so your mileage may vary.",
    ],
}

# Cosine similarity threshold above which a sentence gets flagged (and
# logged). Conservative (high) starting point on purpose.
SIMILARITY_THRESHOLD = 0.72

# Enforcement is narrower than logging on two axes: fewer categories AND
# a stricter score. Only "ai_tell" and "formal_signoff" are auto-strip
# eligible — see module docstring for why "unsolicited_advice" and
# "hedging" are excluded. The higher bar (vs. SIMILARITY_THRESHOLD) is a
# second, independent safety margin on top of the category restriction:
# even within an approved category, only strip when the match is close
# to unambiguous.
ENFORCE_CATEGORIES = {"ai_tell", "formal_signoff"}
ENFORCE_SIMILARITY_THRESHOLD = 0.80

_seed_embeddings_cache = {"data": None}  # {(category, phrase): embedding}

# Simple sentence splitter consistent with the boundary logic already used
# in main.py's AI-tell-phrase removal (., !, ?, or newline as a hard
# boundary) — texting-voice replies are fragmented and low-punctuation, so
# newline has to count as a sentence boundary here too.
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    pieces = [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]
    return pieces


def _get_seed_embeddings():
    """
    Lazily computes and caches embeddings for every seed phrase, once per
    process lifetime. Imports get_embedding_fn from rag.py rather than
    loading a separate model — same lazy-loaded all-MiniLM-L6-v2 instance
    RAG search already uses.
    """
    if _seed_embeddings_cache["data"] is not None:
        return _seed_embeddings_cache["data"]

    from rag import get_embedding_fn
    embed_fn = get_embedding_fn()

    flat_phrases = []
    flat_labels = []  # (category, phrase) pairs, same order as flat_phrases
    for category, phrases in SEED_PHRASES.items():
        for phrase in phrases:
            flat_phrases.append(phrase)
            flat_labels.append((category, phrase))

    vectors = embed_fn(flat_phrases)
    _seed_embeddings_cache["data"] = list(zip(flat_labels, vectors))
    return _seed_embeddings_cache["data"]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def check_semantic_violations(text: str, session_id: str = "") -> list[dict]:
    """
    Splits `text` into sentences, embeds them, and compares each against
    every seed phrase via cosine similarity. Returns a list of
    {"sentence": ..., "category": ..., "matched_seed": ..., "score": ...,
    "enforce": ...} for every sentence that crosses SIMILARITY_THRESHOLD
    against at least one seed — its single best match. "enforce" is True
    only when the match is also in ENFORCE_CATEGORIES and clears
    ENFORCE_SIMILARITY_THRESHOLD — see module docstring.

    This function itself never modifies `text` — it only detects and
    returns findings. Callers that want the auto-strip behavior should
    call strip_semantic_violations() instead, which uses this function
    internally and then removes the enforce-eligible sentences.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    try:
        from rag import get_embedding_fn
        embed_fn = get_embedding_fn()
        sentence_vectors = embed_fn(sentences)
        seed_data = _get_seed_embeddings()
    except Exception as e:
        # Embedding model unavailable/failed to load — fail open (no
        # findings) rather than breaking the reply pipeline over a
        # diagnostic/enforcement feature.
        print(f"[semantic_style_check] embedding failed, skipping: {e}")
        return []

    findings = []
    for sentence, sentence_vec in zip(sentences, sentence_vectors):
        best_score = 0.0
        best_category = None
        best_seed = None
        for (category, seed_phrase), seed_vec in seed_data:
            score = _cosine_similarity(sentence_vec, seed_vec)
            if score > best_score:
                best_score = score
                best_category = category
                best_seed = seed_phrase

        if best_score >= SIMILARITY_THRESHOLD:
            findings.append({
                "sentence": sentence,
                "category": best_category,
                "matched_seed": best_seed,
                "score": round(best_score, 3),
                "enforce": (
                    best_category in ENFORCE_CATEGORIES
                    and best_score >= ENFORCE_SIMILARITY_THRESHOLD
                ),
            })

    return findings


def strip_semantic_violations(text: str, session_id: str = "", mode: str = "") -> tuple[str, list[dict]]:
    """
    Runs check_semantic_violations(), then removes only the sentences
    flagged with "enforce": True (see ENFORCE_CATEGORIES /
    ENFORCE_SIMILARITY_THRESHOLD). Returns (cleaned_text, findings) —
    findings includes every flagged sentence, enforced or not, so logging
    stays exactly as informative as it was in log-only mode.

    If mode == "supportive", formal_signoff stripping is relaxed so genuine
    warmth/support sentences are not stripped from a vulnerable user.

    Removal is done via exact substring replacement (each sentence text
    came from splitting the original string, so it's guaranteed to be a
    literal substring of it) rather than rejoining the sentence list —
    that preserves the original spacing/newlines around whatever's left,
    which matters for fragmented, low-punctuation texting-voice replies.
    """
    findings = check_semantic_violations(text, session_id=session_id)
    if not findings:
        return text, findings

    cleaned = text
    for f in findings:
        # In supportive mode, do not strip warm signoffs like "Feel free to reach out anytime"
        if mode == "supportive" and f.get("category") == "formal_signoff":
            continue
        if f["enforce"]:
            cleaned = cleaned.replace(f["sentence"], "", 1)

    if cleaned != text:
        # Collapse whitespace/blank lines left behind by removed
        # sentences — same cleanup main.py already applies after its own
        # AI_TELL_PHRASES removal.
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return cleaned, findings
