"""
Bot Bot — Laya Typed-Decision Engine (laya_engine.py)

Encapsulates the 'convaiinnovations/laya' ModernBERT (~421M) typed-decision model:
- Runs strictly on CPU in the main.py process under torch.inference_mode()
- Zero NPU contention (never interacts with npu_server or npu_lock)
- Lazy singleton loader with clean offline-first fallback
- Provides typed decision helpers for:
  1. Mode & personal RAG need classification (single-pass multi-question predict)
  2. Candidate chunk reranking & irrelevant chunk filtering (batched predict_batch)
  3. Context sufficiency assessment (sufficient / partial / insufficient with token budget)
"""

import os
# Prevent TensorFlow / Abseil deadlock & TF DLL scanning on Windows before any imports
os.environ.setdefault("USE_TF", "0")

import time
import logging
import threading
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger("botbot.laya")

# Feature flag & Model ID configuration
LAYA_ENABLED = os.environ.get("BOTBOT_LAYA_ENABLED", "1").lower() in ("1", "true", "yes")
LAYA_MODEL_ID = os.environ.get("BOTBOT_LAYA_MODEL_PATH", "convaiinnovations/laya")

# Internal singleton state
_laya_lock = threading.Lock()
_laya_model = None
_laya_load_attempted = False
_laya_available = False
_warned_unavailable = False


def is_laya_available() -> bool:
    """Returns True if Laya is enabled and successfully loaded."""
    if not LAYA_ENABLED:
        return False
    if not _laya_load_attempted:
        get_laya_model()
    return _laya_available


def get_laya_model() -> Optional[Any]:
    """
    Lazy thread-safe singleton loader for the Laya model on CPU.
    Respects HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE flags.
    If weights are not found or loading fails, logs a clear warning once,
    marks Laya unavailable for the process lifetime, and returns None.
    """
    global _laya_model, _laya_load_attempted, _laya_available, _warned_unavailable

    if not LAYA_ENABLED:
        return None

    if _laya_load_attempted:
        return _laya_model if _laya_available else None

    with _laya_lock:
        if _laya_load_attempted:
            return _laya_model if _laya_available else None

        _laya_load_attempted = True
        t0 = time.perf_counter()

        # 1. Attempt loading via official `laya` package (fast load: skips throwaway random init)
        try:
            import torch
            try:
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("medium")
                target_threads = min(8, os.cpu_count() or 8)
                if torch.get_num_threads() != target_threads:
                    torch.set_num_threads(target_threads)
            except Exception:
                pass

            import laya  # type: ignore
            print(f"[laya] Loading model '{LAYA_MODEL_ID}' via laya SDK...")
            _laya_model = laya.load(LAYA_MODEL_ID)
            _laya_available = True
            dur_ms = (time.perf_counter() - t0) * 1000
            print(f"[laya] Model loaded successfully via laya SDK in {dur_ms:.1f}ms")
            return _laya_model
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"[laya] laya.load failed: {e}")

        # 2. Attempt loading via HuggingFace transformers fallback
        try:
            import torch
            from transformers import AutoModel  # type: ignore

            local_only = (
                os.environ.get("HF_HUB_OFFLINE", "1") == "1"
                or os.environ.get("TRANSFORMERS_OFFLINE", "1") == "1"
            )

            print(f"[laya] Loading model '{LAYA_MODEL_ID}' via transformers (local_files_only={local_only})...")
            model = AutoModel.from_pretrained(
                LAYA_MODEL_ID,
                trust_remote_code=True,
                local_files_only=local_only,
            )
            model.eval()
            _laya_model = model
            _laya_available = True
            dur_ms = (time.perf_counter() - t0) * 1000
            print(f"[laya] Model loaded successfully via transformers in {dur_ms:.1f}ms")
            return _laya_model
        except Exception as e:
            _laya_available = False
            _laya_model = None
            if not _warned_unavailable:
                _warned_unavailable = True
                print(
                    f"[laya] Notice: Laya model ('{LAYA_MODEL_ID}') unavailable ({e}). "
                    "Running with clean fallback to existing heuristics / RRF. "
                    "To download Laya weights, run with HF_HUB_OFFLINE=0."
                )
            return None


def _predict_raw(model: Any, state_text: str, questions: dict, max_len: Optional[int] = None) -> Optional[dict]:
    """
    Executes a typed prediction against the loaded model under torch.inference_mode().
    Handles both Laya agent interface and transformers AutoModel wrapper.
    """
    try:
        import torch
        try:
            target_threads = min(8, os.cpu_count() or 8)
            if torch.get_num_threads() != target_threads:
                torch.set_num_threads(target_threads)
        except Exception:
            pass

        with torch.inference_mode():
            if hasattr(model, "predict"):
                if max_len is not None:
                    try:
                        return model.predict(state_text, questions, max_len=max_len)
                    except TypeError:
                        return model.predict(state_text, questions)
                return model.predict(state_text, questions)
            elif callable(model):
                return model(state_text, questions)
    except Exception as e:
        logger.warning(f"[laya] _predict_raw encountered error: {e}")
    return None


def _predict_batch_raw(
    model: Any,
    states: List[str],
    questions: dict,
    max_len: int = 192
) -> Optional[List[dict]]:
    """
    Executes a batched typed prediction against the loaded model under torch.inference_mode().
    Calls model.predict_batch() in a single forward pass, or falls back to looping predict().
    """
    if not states:
        return []
    try:
        import torch
        try:
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("medium")
            target_threads = min(8, os.cpu_count() or 8)
            if torch.get_num_threads() != target_threads:
                torch.set_num_threads(target_threads)
        except Exception:
            pass

        with torch.inference_mode():
            if hasattr(model, "predict_batch"):
                try:
                    return model.predict_batch(states, questions, max_len=max_len)
                except TypeError:
                    return model.predict_batch(states, questions)
            elif hasattr(model, "predict"):
                try:
                    return [model.predict(s, questions, max_len=max_len) for s in states]
                except TypeError:
                    return [model.predict(s, questions) for s in states]
            elif callable(model):
                return [model(s, questions) for s in states]
    except Exception as e:
        logger.warning(f"[laya] _predict_batch_raw encountered error: {e}")
    return None


# ---------------------------------------------------------------------------
# Fallback logic for when Laya is disabled or unavailable
# ---------------------------------------------------------------------------

_SUPPORTIVE_KEYWORDS = frozenset([
    "anxious", "depressed", "sad", "lonely", "stressed",
    "overwhelmed", "scared", "hurt", "crying", "hopeless",
    "worthless", "tired of", "can't cope", "feel like giving up",
    "miss", "grief", "heartbroken", "exhausted", "burnt out",
    "panic", "afraid", "insecure", "ashamed", "guilty"
])

_GUIDANCE_KEYWORDS = frozenset([
    "career", "future", "goal", "plan", "decide", "decision",
    "should i", "advice", "direction", "purpose", "life path",
    "choice", "path", "job offer", "next step", "stuck between",
    "thinking of switching", "considering", "what should i do"
])

_TRIVIAL_PHRASES = frozenset([
    "ok", "okay", "k", "cool", "nice", "yeah", "yes", "yep", "no", "nope",
    "nah", "lol", "haha", "hahaha", "lmao", "thanks", "thank you", "thx",
    "gm", "gn", "good morning", "good night", "hey", "hi", "hello", "yo",
    "sup", "bye", "cya", "see ya", "sure", "alright", "got it", "understood",
    "great", "awesome", "perfect", "sounds good", "fair enough", "true",
    "right", "wow", "damn", "omg", "idk", "idc", "np", "nw",
])


def _heuristic_classify_mode(text: str) -> str:
    low = text.lower()
    if any(k in low for k in _SUPPORTIVE_KEYWORDS):
        return "supportive"
    elif any(k in low for k in _GUIDANCE_KEYWORDS):
        return "guidance"
    return "conversational"


def _heuristic_needs_retrieval(text: str) -> bool:
    cleaned = text.strip().lower().rstrip("!?. ")
    if cleaned in _TRIVIAL_PHRASES:
        return False
    if len(cleaned) <= 4 and cleaned.isalpha():
        return False
    return True


# ---------------------------------------------------------------------------
# Typed Decision Helper 1: Single-Pass Multi-Question Mode & Retrieval Classification
# ---------------------------------------------------------------------------

def classify_mode_and_retrieval(
    user_message: str,
    recent_turns: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Evaluates behavioral mode across ["conversational", "supportive", "guidance"]
    and retrieval need across ["needs_personal_context", "no_personal_context_needed"]
    in a SINGLE forward pass via agent.predict(state, questions).
    """
    t0 = time.perf_counter()
    model = get_laya_model()

    if model is None:
        mode = _heuristic_classify_mode(user_message)
        needs_ctx = _heuristic_needs_retrieval(user_message)
        dur_ms = (time.perf_counter() - t0) * 1000
        return {
            "mode": mode,
            "mode_confidence": 1.0,
            "mode_probabilities": {mode: 1.0},
            "needs_personal_context": needs_ctx,
            "retrieval_decision": "needs_personal_context" if needs_ctx else "no_personal_context_needed",
            "retrieval_confidence": 1.0,
            "retrieval_probabilities": {
                "needs_personal_context": 1.0 if needs_ctx else 0.0,
                "no_personal_context_needed": 0.0 if needs_ctx else 1.0,
            },
            "latency_ms": round(dur_ms, 2),
            "used_fallback": True,
        }

    # Prepare context state
    context_str = user_message
    if recent_turns:
        hist_lines = [f"{t.get('role', 'user')}: {t.get('content', '')}" for t in recent_turns[-4:]]
        context_str = "Recent conversation:\n" + "\n".join(hist_lines) + f"\nCurrent user message: {user_message}"

    # Single-pass multi-question dictionary with calibrated choice criteria
    questions = {
        "mode": {
            "type": "choice",
            "instructions": "What conversational mode should be used for this user message?",
            "criteria": {
                "conversational": "Casual talk, greetings, banter, or friendly chat",
                "supportive": "Emotional distress, pain, sadness, anxiety, or feeling overwhelmed",
                "guidance": "Asking for advice, direction, or career and life planning",
            },
        },
        "retrieval": {
            "type": "choice",
            "instructions": "Determine if answering this message requires retrieving personal context (user memories, notes, past events, specific people, or project details).",
            "criteria": {
                "needs_personal_context": "Asks about or refers to specific people, past events, projects, personal history, or user memories.",
                "no_personal_context_needed": "General conversation, greeting, banter, advice request, or universal question without personal memory lookup.",
            },
        },
    }

    raw = _predict_raw(model, context_str, questions, max_len=192)
    dur_ms = (time.perf_counter() - t0) * 1000

    if not raw or "answers" not in raw:
        # Fall back gracefully
        mode = _heuristic_classify_mode(user_message)
        needs_ctx = _heuristic_needs_retrieval(user_message)
        return {
            "mode": mode,
            "mode_confidence": 1.0,
            "mode_probabilities": {mode: 1.0},
            "needs_personal_context": needs_ctx,
            "retrieval_decision": "needs_personal_context" if needs_ctx else "no_personal_context_needed",
            "retrieval_confidence": 1.0,
            "retrieval_probabilities": {
                "needs_personal_context": 1.0 if needs_ctx else 0.0,
                "no_personal_context_needed": 0.0 if needs_ctx else 1.0,
            },
            "latency_ms": round(dur_ms, 2),
            "used_fallback": True,
        }

    mode_ans = raw["answers"].get("mode", {})
    ret_ans = raw["answers"].get("retrieval", {})

    mode_choice = mode_ans.get("choice", _heuristic_classify_mode(user_message))
    mode_conf = float(mode_ans.get("answer_confidence", mode_ans.get("confidence", 0.5)))
    mode_probs = mode_ans.get("probabilities", {mode_choice: mode_conf})

    ret_choice = ret_ans.get("choice", "needs_personal_context")
    ret_conf = float(ret_ans.get("answer_confidence", ret_ans.get("confidence", 0.5)))
    ret_probs = ret_ans.get("probabilities", {ret_choice: ret_conf})

    return {
        "mode": mode_choice,
        "mode_confidence": mode_conf,
        "mode_probabilities": mode_probs,
        "needs_personal_context": (ret_choice == "needs_personal_context"),
        "retrieval_decision": ret_choice,
        "retrieval_confidence": ret_conf,
        "retrieval_probabilities": ret_probs,
        "latency_ms": round(dur_ms, 2),
        "used_fallback": False,
    }


# ---------------------------------------------------------------------------
# Typed Decision Helper 2: Candidate Chunk Reranking & Filtering (Batched)
# ---------------------------------------------------------------------------

MAX_LAYA_RERANK_CANDIDATES = 5


def rerank_and_filter_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    relevance_threshold: float = 0.35,
    max_eval_chunks: int = MAX_LAYA_RERANK_CANDIDATES,
    max_chars: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Given candidate chunks from RRF fusion, uses Laya to score candidate excerpts
    in a SINGLE FORWARD PASS via agent.predict() using choice selection over candidate excerpts.
    This avoids running multiple independent forward passes across the candidate pool,
    bringing chunk reranking latency well under 1.2s on CPU.

    Scored options include each candidate excerpt snippet plus a 'none_relevant' safeguard option.
    Filters out chunks with negligible probability or where none_relevant dominates.
    Sorts surviving chunks by composite relevance score descending.

    Performance & Remainder Budget:
    - Candidate pool evaluated with Laya is bounded to MAX_LAYA_RERANK_CANDIDATES (5).
    - Remainder chunks beyond the top 5 are dropped unless at least 3 of the top 5 were
      classified as relevant AND the character budget still has room. Un-evaluated lower-rank
      RRF tail chunks are NEVER appended when top-ranked RRF chunks failed the relevance check.
    - Snippets are capped to ~200 characters in the choice options.
    - The full original chunk text is kept 100% intact in the returned kept_chunks
      for downstream RAG prompt assembly.

    Returns (kept_chunks, diagnostics_list).
    """
    if not chunks:
        return [], []

    model = get_laya_model()
    if model is None:
        diagnostics = [
            {"chunk_index": i, "kept": True, "used_fallback": True}
            for i in range(len(chunks))
        ]
        return chunks, diagnostics

    t0 = time.perf_counter()
    eval_candidates = chunks[:max_eval_chunks]
    remainder = chunks[max_eval_chunks:]

    # Prepare single-pass multi-excerpt choice question
    state = f"User query: {query}"
    criteria = {}
    for i, c in enumerate(eval_candidates):
        text = c.get("text", c.get("content", c.get("doc", "")))
        snippet = text[:200].replace("\n", " ").strip()
        criteria[f"excerpt_{i}"] = f"Excerpt {i+1}: {snippet}"
    criteria["none_relevant"] = "None of the candidate excerpts are relevant to the user query."

    questions = {
        "relevance": {
            "type": "choice",
            "instructions": f"Which candidate excerpt contains the most relevant context to answer the user query: '{query}'?",
            "criteria": criteria,
        }
    }

    raw = _predict_raw(model, state, questions, max_len=256)
    dur_ms = (time.perf_counter() - t0) * 1000

    scored_chunks = []
    diagnostics = []

    if raw and "answers" in raw and "relevance" in raw["answers"]:
        ans = raw["answers"]["relevance"]
        probs = ans.get("probabilities", {})
        p_none = probs.get("none_relevant", 0.0)

        # Check if the model answered with standard per-chunk 3-way choices (e.g. from MockLayaModel)
        if "directly_relevant" in probs or ans.get("choice") in ("directly_relevant", "tangential_context", "irrelevant"):
            choice = ans.get("choice", "directly_relevant")
            conf = float(ans.get("confidence", 0.5))
            p_dir = probs.get("directly_relevant", 0.5 if choice == "directly_relevant" else 0.0)
            p_tan = probs.get("tangential_context", 0.5 if choice == "tangential_context" else 0.0)
            kept = (choice != "irrelevant") and ((p_dir + p_tan) >= relevance_threshold)
            comp = round(p_dir * 1.0 + p_tan * 0.5, 3)
            for idx, chunk in enumerate(eval_candidates):
                diag = {
                    "chunk_index": idx,
                    "choice": choice,
                    "confidence": conf,
                    "composite_relevance": comp,
                    "probabilities": probs,
                    "kept": kept,
                    "used_fallback": False,
                    "batch_latency_ms": round(dur_ms, 2),
                }
                diagnostics.append(diag)
                if kept:
                    scored_chunks.append((comp, chunk))
        else:
            # Single-pass multi-excerpt probability distribution
            for idx, chunk in enumerate(eval_candidates):
                p_chunk = probs.get(f"excerpt_{idx}", 0.0)
                if p_chunk >= 0.12 or (p_chunk > p_none and p_chunk >= 0.08):
                    choice = "directly_relevant"
                    kept = True
                    comp = round(p_chunk, 3)
                elif p_chunk >= 0.04 and p_chunk >= (p_none * 0.25):
                    choice = "tangential_context"
                    kept = True
                    comp = round(p_chunk * 0.7, 3)
                else:
                    choice = "irrelevant"
                    kept = False
                    comp = round(p_chunk, 3)

                diag = {
                    "chunk_index": idx,
                    "choice": choice,
                    "confidence": round(p_chunk, 3),
                    "composite_relevance": comp,
                    "probabilities": {
                        "directly_relevant": p_chunk if choice == "directly_relevant" else 0.0,
                        "tangential_context": p_chunk if choice == "tangential_context" else 0.0,
                        "irrelevant": round(1.0 - p_chunk, 3),
                    },
                    "kept": kept,
                    "used_fallback": False,
                    "batch_latency_ms": round(dur_ms, 2),
                }
                diagnostics.append(diag)
                if kept:
                    scored_chunks.append((comp, chunk))
    else:
        # Fallback if raw prediction failed
        for idx, chunk in enumerate(eval_candidates):
            diag = {
                "chunk_index": idx,
                "choice": "unknown",
                "confidence": 0.0,
                "composite_relevance": 0.5,
                "probabilities": {},
                "kept": True,
                "used_fallback": True,
                "batch_latency_ms": round(dur_ms, 2),
            }
            diagnostics.append(diag)
            scored_chunks.append((0.5, chunk))

    # Sort surviving evaluated chunks by composite relevance descending
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    kept_chunks = [item[1] for item in scored_chunks]

    # Remainder handling beyond MAX_LAYA_RERANK_CANDIDATES:
    # Never append un-evaluated lower-rank RRF tail chunks when the top-ranked RRF chunks failed relevance!
    # Only include remainder chunks if at least 3 of the top evaluated chunks were relevant
    # and the character budget still has room.
    num_top_relevant = len(scored_chunks)
    if num_top_relevant >= 3 and remainder:
        current_chars = sum(len(c.get("text", c.get("content", c.get("doc", "")))) for c in kept_chunks)
        char_limit = max_chars if max_chars is not None else 3500
        for rem_chunk in remainder:
            chunk_len = len(rem_chunk.get("text", rem_chunk.get("content", rem_chunk.get("doc", ""))))
            if current_chars + chunk_len <= char_limit:
                kept_chunks.append(rem_chunk)
                current_chars += chunk_len
            else:
                break

    return kept_chunks, diagnostics


# ---------------------------------------------------------------------------
# Typed Decision Helper 3: Context Sufficiency Assessment (Token-Budget Aware)
# ---------------------------------------------------------------------------

SUFFICIENCY_MIN_CONFIDENCE = 0.55
SUFFICIENCY_DIGEST_CHARS = 1400


def assess_context_sufficiency(
    query: str,
    assembled_context: str
) -> Dict[str, Any]:
    """
    Evaluates whether the assembled context answers the query:
    returns {"decision": "sufficient_to_answer" | "partial_context_hedge_needed" | "insufficient_context_abstain", ...}

    Token-budget awareness:
    Assembled context across personal and reference data can exceed 512 tokens (up to ~3000+ chars).
    If assembled_context exceeds ~1400 chars, creates a concise digest across all excerpts so the
    entire context is represented within Laya's 512-token limit without silent right-truncation.
    """
    t0 = time.perf_counter()
    if not assembled_context or not assembled_context.strip():
        dur_ms = (time.perf_counter() - t0) * 1000
        return {
            "decision": "no_context",
            "confidence": 1.0,
            "probabilities": {"no_context": 1.0},
            "latency_ms": round(dur_ms, 2),
            "used_fallback": True,
        }

    model = get_laya_model()
    if model is None:
        dur_ms = (time.perf_counter() - t0) * 1000
        return {
            "decision": "sufficient_to_answer",
            "confidence": 0.5,
            "probabilities": {
                "sufficient_to_answer": 0.5,
                "partial_context_hedge_needed": 0.3,
                "insufficient_context_abstain": 0.2,
            },
            "latency_ms": round(dur_ms, 2),
            "used_fallback": True,
        }

    # Format context within Laya's 512-token budget (~1400 chars digest).
    # Every excerpt gets an equal share, so later excerpts are never silently
    # dropped by a hard cut on the joined string (that made the judge say
    # "insufficient" when the answer was in an excerpt it never saw).
    if len(assembled_context) <= SUFFICIENCY_DIGEST_CHARS:
        context_digest = assembled_context.strip()
    else:
        paragraphs = [p.strip() for p in assembled_context.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            sep = "\n---\n"
            per_part_budget = max(
                40, (SUFFICIENCY_DIGEST_CHARS - len(sep) * (len(paragraphs) - 1)) // len(paragraphs)
            )
            digest_parts = [
                p[:per_part_budget].rstrip() + ("..." if len(p) > per_part_budget else "")
                for p in paragraphs
            ]
            context_digest = sep.join(digest_parts)[:SUFFICIENCY_DIGEST_CHARS + 200]
        else:
            context_digest = assembled_context[:SUFFICIENCY_DIGEST_CHARS].rstrip() + "..."

    state = (
        f"User question: {query}\n\n"
        f"Retrieved context excerpts:\n{context_digest}\n\n"
        "Evaluate whether the retrieved context contains the specific information requested."
    )

    questions = {
        "sufficiency": {
            "type": "choice",
            "instructions": "Evaluate context sufficiency for the query.",
            "criteria": {
                "sufficient": "Context answers the question fully",
                "partial": "Context partially answers or is missing key information",
                "insufficient": "Context is irrelevant or does not answer the question",
            },
        }
    }

    raw = _predict_raw(model, state, questions)
    dur_ms = (time.perf_counter() - t0) * 1000

    if not raw or "answers" not in raw or "sufficiency" not in raw["answers"]:
        return {
            "decision": "sufficient_to_answer",
            "confidence": 0.5,
            "probabilities": {
                "sufficient_to_answer": 0.5,
                "partial_context_hedge_needed": 0.3,
                "insufficient_context_abstain": 0.2,
            },
            "latency_ms": round(dur_ms, 2),
            "used_fallback": True,
        }

    ans = raw["answers"]["sufficiency"]
    raw_choice = ans.get("choice", "sufficient")
    conf = float(ans.get("confidence", 0.5))
    raw_probs = ans.get("probabilities", {})

    # Map calibrated choices to canonical status
    choice_map = {
        "sufficient": "sufficient_to_answer",
        "sufficient_to_answer": "sufficient_to_answer",
        "partial": "partial_context_hedge_needed",
        "partial_context_hedge_needed": "partial_context_hedge_needed",
        "insufficient": "insufficient_context_abstain",
        "insufficient_context_abstain": "insufficient_context_abstain",
    }
    canonical_decision = choice_map.get(raw_choice, "sufficient_to_answer")

    # Same idea as the 0.55 threshold used for mode classification: a low-
    # confidence verdict is softened one step instead of being trusted. A weak
    # "insufficient" becomes "partial" (hedge) and a weak "partial" becomes
    # "sufficient" (the generic soft sufficiency instruction still applies).
    raw_decision = canonical_decision
    if conf < SUFFICIENCY_MIN_CONFIDENCE:
        if canonical_decision == "insufficient_context_abstain":
            canonical_decision = "partial_context_hedge_needed"
        elif canonical_decision == "partial_context_hedge_needed":
            canonical_decision = "sufficient_to_answer"

    canonical_probs = {
        "sufficient_to_answer": raw_probs.get("sufficient", raw_probs.get("sufficient_to_answer", 0.0)),
        "partial_context_hedge_needed": raw_probs.get("partial", raw_probs.get("partial_context_hedge_needed", 0.0)),
        "insufficient_context_abstain": raw_probs.get("insufficient", raw_probs.get("insufficient_context_abstain", 0.0)),
    }
    if not any(canonical_probs.values()):
        canonical_probs[canonical_decision] = conf

    return {
        "decision": canonical_decision,
        "confidence": conf,
        "probabilities": canonical_probs,
        "raw_decision": raw_decision,
        "downgraded_for_low_confidence": raw_decision != canonical_decision,
        "latency_ms": round(dur_ms, 2),
        "used_fallback": False,
    }
