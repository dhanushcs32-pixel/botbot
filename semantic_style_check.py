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

LOG-ONLY BY DELIBERATE DESIGN, at least initially (see PROJECT_STATE.txt):
similarity thresholds need real tuning against actual traffic before it's
safe to auto-strip anything. A same-precedent choice was made for
bracket_placeholder/salutation_closing in main.py's STRUCTURE_RULES —
shipping straight to auto-fix risks false-positive-deleting a perfectly
fine sentence that just happens to sit near a seed phrase in embedding
space. Run this logging what it WOULD flag and at what score first, then
decide a real cutoff.
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
        "As an AI, I don't have personal experiences.",
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
        "I'm always here if you need to talk.",
    ],
    "hedging": [
        "I could be wrong, but it seems like this might be the case.",
        "This is just my perspective, so take it with a grain of salt.",
        "Everyone's experience is different, so your mileage may vary.",
    ],
}

# Cosine similarity threshold above which a sentence gets flagged.
# Conservative (high) starting point on purpose — log-only mode means
# false positives just add log noise, but it's still worth starting
# strict and loosening only once real traffic shows it's too quiet.
SIMILARITY_THRESHOLD = 0.72

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
    {"sentence": ..., "category": ..., "matched_seed": ..., "score": ...}
    for every sentence that crosses SIMILARITY_THRESHOLD against at least
    one seed — its single best match.

    LOG-ONLY: this function only detects and returns findings, it does NOT
    modify `text`. Wiring it into main.py's scrub_reply() should call this
    for its return value and log it, not use it to alter the reply, until
    a real threshold is chosen from observed traffic (see module
    docstring).
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
        # log-only diagnostic feature.
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
            })

    return findings
