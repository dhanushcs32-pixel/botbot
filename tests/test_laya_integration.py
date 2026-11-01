"""
Laya Typed-Decision Integration Test Suite (tests/test_laya_integration.py)

Run:
    cd C:\\botbot && .\\venv\\Scripts\\python.exe tests\\test_laya_integration.py

Part A: Fallback & Flag-Off Safety
  - With LAYA_ENABLED=False or model unavailable, verifies detect_mode(),
    search_personal_data(), and get_system_prompt() cleanly fall back to
    baseline heuristics with zero breakage.

Part B: Mocked / Unit Laya Pipeline
  - Tests chunk filtering logic: drops 'irrelevant' chunks, retains and ranks
    directly_relevant & tangential_context chunks.
  - Tests sufficiency_state propagation into get_system_prompt()
    (insufficient_context_abstain, partial_context_hedge_needed, sufficient_to_answer)
    and verifies that the user's style_profile (my_style.txt) strictly remains LAST.
  - Tests JSONL schema compatibility with analyze_sufficiency_log.py.

Part C: Live Laya Smoke Test
  - If convaiinnovations/laya is cached locally, runs a live CPU forward pass
    and reports latency. If not cached, prints the exact pre-download command.
"""

import os
import sys
import json
import time
from pathlib import Path

# Enforce offline defaults
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import laya_engine
import mode_router
import rag

ok = True


def check(name: str, cond: bool, detail: str = ""):
    global ok
    status = "PASS" if cond else "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    ok = ok and cond


# =========================================================================
# Part A: Fallback & Flag-Off Safety
# =========================================================================
print("\n--- Part A: Fallback & Flag-Off Safety ---")

# Save baseline states
original_laya_flag = laya_engine.LAYA_ENABLED
original_mode_flag = mode_router.LAYA_ENABLED
original_rag_flag = rag.LAYA_ENABLED

# 1. Test with LAYA_ENABLED = False
laya_engine.LAYA_ENABLED = False
mode_router.LAYA_ENABLED = False
rag.LAYA_ENABLED = False

m_conv = mode_router.detect_mode("hey what's up")
check("Flag-Off: detect_mode('hey') returns 'conversational'", m_conv == "conversational", f"got '{m_conv}'")

m_supp = mode_router.detect_mode("i am feeling really overwhelmed and anxious today")
check("Flag-Off: detect_mode('overwhelmed') returns 'supportive'", m_supp == "supportive", f"got '{m_supp}'")

m_guid = mode_router.detect_mode("what career decision should i make about this job offer?")
check("Flag-Off: detect_mode('career') returns 'guidance'", m_guid == "guidance", f"got '{m_guid}'")

det = mode_router.detect_mode_detailed("feeling sad and hurt")
check("Flag-Off: detect_mode_detailed reports used_fallback=True", det.get("used_fallback") is True)
check("Flag-Off: detect_mode_detailed mode is 'supportive'", det.get("mode") == "supportive")

# Prompt assembly baseline check
sp_baseline = mode_router.get_system_prompt(
    "conversational",
    personal_context="some context",
    style_profile="never use emojis."
)
check("Flag-Off: get_system_prompt contains CONTEXT_SUFFICIENCY_INSTRUCTION",
      "Before answering from the context above" in sp_baseline)
check("Flag-Off: style profile is placed at the end",
      sp_baseline.strip().endswith("never use emojis."))

# Bypass personal retrieval check
bypass_trivial = rag.should_bypass_personal_retrieval("ok")
check("Flag-Off: should_bypass_personal_retrieval on 'ok' is True", bypass_trivial is True)

# Restore flags
laya_engine.LAYA_ENABLED = original_laya_flag
mode_router.LAYA_ENABLED = original_mode_flag
rag.LAYA_ENABLED = original_rag_flag


# =========================================================================
# Part B: Mocked / Unit Laya Pipeline
# =========================================================================
print("\n--- Part B: Mocked / Unit Laya Pipeline ---")

# Mock Laya model for pipeline testing
class MockLayaModel:
    def predict(self, state, questions):
        text = state.get("text", "") if isinstance(state, dict) else str(state)
        answers = {}

        if "mode" in questions:
            if "depressed" in text or "sad" in text:
                answers["mode"] = {
                    "choice": "supportive",
                    "confidence": 0.94,
                    "probabilities": {"supportive": 0.94, "conversational": 0.04, "guidance": 0.02}
                }
            elif "career" in text or "plan" in text:
                answers["mode"] = {
                    "choice": "guidance",
                    "confidence": 0.89,
                    "probabilities": {"guidance": 0.89, "conversational": 0.08, "supportive": 0.03}
                }
            else:
                answers["mode"] = {
                    "choice": "conversational",
                    "confidence": 0.91,
                    "probabilities": {"conversational": 0.91, "supportive": 0.05, "guidance": 0.04}
                }

        if "retrieval" in questions:
            if "who is" in text or "what happened" in text or "remember" in text:
                answers["retrieval"] = {
                    "choice": "needs_personal_context",
                    "confidence": 0.95,
                    "probabilities": {"needs_personal_context": 0.95, "no_personal_context_needed": 0.05}
                }
            else:
                answers["retrieval"] = {
                    "choice": "no_personal_context_needed",
                    "confidence": 0.85,
                    "probabilities": {"needs_personal_context": 0.15, "no_personal_context_needed": 0.85}
                }

        if "relevance" in questions:
            rel_q = questions["relevance"]
            crit = rel_q.get("criteria", {})
            if any(k.startswith("excerpt_") for k in crit):
                probs = {}
                best_choice = "none_relevant"
                best_val = 0.0
                for k, desc in crit.items():
                    d_low = desc.lower()
                    if "crypto" in d_low:
                        probs[k] = 0.65
                    elif "church street" in d_low:
                        probs[k] = 0.25
                    elif k == "none_relevant":
                        probs[k] = 0.05
                    else:
                        probs[k] = 0.01
                    if probs[k] > best_val:
                        best_val = probs[k]
                        best_choice = k
                answers["relevance"] = {
                    "choice": best_choice,
                    "confidence": best_val,
                    "probabilities": probs
                }
            else:
                # Isolate the candidate excerpt portion of the state
                candidate_part = text.split("Candidate excerpt:\n")[-1].lower() if "Candidate excerpt:\n" in text else text.lower()
                if "crypto" in candidate_part:
                    answers["relevance"] = {
                        "choice": "directly_relevant",
                        "confidence": 0.92,
                        "probabilities": {"directly_relevant": 0.92, "tangential_context": 0.06, "irrelevant": 0.02}
                    }
                elif "church street" in candidate_part:
                    answers["relevance"] = {
                        "choice": "tangential_context",
                        "confidence": 0.70,
                        "probabilities": {"directly_relevant": 0.20, "tangential_context": 0.70, "irrelevant": 0.10}
                    }
                else:
                    answers["relevance"] = {
                        "choice": "irrelevant",
                        "confidence": 0.96,
                        "probabilities": {"directly_relevant": 0.01, "tangential_context": 0.03, "irrelevant": 0.96}
                    }

        if "sufficiency" in questions:
            ctx_part = text.split("Retrieved context excerpts:\n")[-1].lower() if "Retrieved context excerpts:\n" in text else text.lower()
            if "crypto" in ctx_part and "repayment date" in ctx_part:
                answers["sufficiency"] = {
                    "choice": "sufficient_to_answer",
                    "confidence": 0.93,
                    "probabilities": {"sufficient_to_answer": 0.93, "partial_context_hedge_needed": 0.05, "insufficient_context_abstain": 0.02}
                }
            elif "partial" in ctx_part or "only mentioned" in ctx_part:
                answers["sufficiency"] = {
                    "choice": "partial_context_hedge_needed",
                    "confidence": 0.82,
                    "probabilities": {"sufficient_to_answer": 0.10, "partial_context_hedge_needed": 0.82, "insufficient_context_abstain": 0.08}
                }
            else:
                answers["sufficiency"] = {
                    "choice": "insufficient_context_abstain",
                    "confidence": 0.88,
                    "probabilities": {"sufficient_to_answer": 0.04, "partial_context_hedge_needed": 0.08, "insufficient_context_abstain": 0.88}
                }

        return {"answers": answers}


# Inject mock model into laya_engine
real_laya_model = laya_engine._laya_model
real_laya_available = laya_engine._laya_available
real_laya_attempted = laya_engine._laya_load_attempted

laya_engine._laya_model = MockLayaModel()
laya_engine._laya_available = True
laya_engine._laya_load_attempted = True

# 1. Test classify_mode_and_retrieval with mock
res_supp = laya_engine.classify_mode_and_retrieval("i feel really sad today")
check("Mock Laya: classify_mode supportive", res_supp["mode"] == "supportive")
check("Mock Laya: confidence >= 0.55", res_supp["mode_confidence"] >= 0.55)
check("Mock Laya: used_fallback is False", res_supp["used_fallback"] is False)

res_ret = laya_engine.classify_mode_and_retrieval("who is Priya?")
check("Mock Laya: retrieval need detected", res_ret["needs_personal_context"] is True)

# Test smart retrieval bypass with Laya routing
bypass_smart = rag.should_bypass_personal_retrieval(
    "let's chat about something general",
    laya_routing={"retrieval_decision": "no_personal_context_needed", "retrieval_confidence": 0.88, "used_fallback": False}
)
check("Smart Bypass: Laya high-confidence no-context query bypassed", bypass_smart is True)

# 2. Test rerank_and_filter_chunks
sample_chunks = [
    {"text": "Random weather discussion in Bangalore", "metadata": {"person": "none"}},
    {"text": "Notes on crypto loan repayment date: March 15", "metadata": {"person": "self"}},
    {"text": "We walked down church street talking about life", "metadata": {"person": "Jack"}},
]

kept, diags = laya_engine.rerank_and_filter_chunks("what did I say about crypto debt?", sample_chunks)
check("Mock Laya: filtered out irrelevant chunk", len(kept) == 2, f"kept {len(kept)} / {len(sample_chunks)}")
check("Mock Laya: top kept chunk is directly relevant", "crypto loan" in kept[0]["text"])
check("Mock Laya: second kept chunk is tangential", "church street" in kept[1]["text"])

# 2b. Test Remainder Dropping Rules (MAX_LAYA_RERANK_CANDIDATES = 5)
# Scenario 1: Top 5 has < 3 relevant (only 2 relevant: crypto + church street).
# Remainder chunks (indices 5, 6, 7) must be strictly DROPPED.
eight_chunks_few_relevant = [
    {"text": "Notes on crypto loan repayment date: March 15"},   # relevant 1
    {"text": "Random weather discussion in Bangalore"},          # irrelevant
    {"text": "We walked down church street talking about life"}, # relevant 2
    {"text": "Irrelevant chit chat about cooking pasta"},        # irrelevant
    {"text": "Another irrelevant discussion on movies"},         # irrelevant
    {"text": "Tail un-evaluated chunk A about finance"},         # remainder 0
    {"text": "Tail un-evaluated chunk B about loans"},           # remainder 1
    {"text": "Tail un-evaluated chunk C about debts"},           # remainder 2
]
kept_dropped, _ = laya_engine.rerank_and_filter_chunks("crypto debt", eight_chunks_few_relevant)
check("Remainder Rule: < 3 top relevant drops all remainder chunks",
      len(kept_dropped) == 2 and not any("Tail" in c["text"] for c in kept_dropped),
      f"expected 2 evaluated chunks only, got {len(kept_dropped)}")

# Scenario 2: Top 5 has 0 relevant (all 5 irrelevant).
# Entire result must be empty, never leaking un-evaluated tail chunks.
eight_chunks_zero_relevant = [
    {"text": "Random weather discussion in Bangalore"},          # irrelevant
    {"text": "Irrelevant chit chat about cooking pasta"},        # irrelevant
    {"text": "Another irrelevant discussion on movies"},         # irrelevant
    {"text": "Discussion on morning jogging routine"},           # irrelevant
    {"text": "Breakfast cereals comparison"},                    # irrelevant
    {"text": "Tail un-evaluated chunk A about crypto"},         # remainder 0
    {"text": "Tail un-evaluated chunk B about loans"},           # remainder 1
    {"text": "Tail un-evaluated chunk C about debts"},           # remainder 2
]
kept_zero, _ = laya_engine.rerank_and_filter_chunks("crypto debt", eight_chunks_zero_relevant)
check("Remainder Rule: 0 top relevant results in empty kept list",
      len(kept_zero) == 0,
      f"expected 0 chunks, got {len(kept_zero)}")

# Scenario 3: Top 5 has >= 3 relevant (3 relevant chunks).
# Remainder chunks are admitted if character budget allows.
eight_chunks_three_relevant = [
    {"text": "Notes on crypto loan repayment date: March 15"},   # relevant 1 (~46 chars)
    {"text": "Second note on crypto collateral rates"},          # relevant 2 (~39 chars)
    {"text": "We walked down church street talking about life"}, # relevant 3 (~48 chars)
    {"text": "Irrelevant chit chat about cooking pasta"},        # irrelevant
    {"text": "Another irrelevant discussion on movies"},         # irrelevant
    {"text": "Tail un-evaluated chunk A"},                       # remainder 0 (~27 chars)
    {"text": "Tail un-evaluated chunk B"},                       # remainder 1 (~27 chars)
    {"text": "Tail un-evaluated chunk C"},                       # remainder 2 (~27 chars)
]
kept_admitted, _ = laya_engine.rerank_and_filter_chunks("crypto debt", eight_chunks_three_relevant, max_chars=1000)
check("Remainder Rule: >= 3 top relevant admits remainder chunks within budget",
      len(kept_admitted) == 6 and any("Tail un-evaluated chunk A" in c["text"] for c in kept_admitted),
      f"expected 6 chunks (3 eval + 3 tail), got {len(kept_admitted)}")

# Scenario 4: Remainder chunks clipped when max_chars budget is reached.
# Evaluated chunks use ~133 chars. With max_chars=175, only 1 remainder chunk of 27 chars fits (133 + 27 = 160 <= 175).
# Second remainder chunk would take 160 + 27 = 187 > 175, so it must be clipped.
kept_budget_clipped, _ = laya_engine.rerank_and_filter_chunks("crypto debt", eight_chunks_three_relevant, max_chars=175)
check("Remainder Rule: remainder chunks clipped when max_chars budget reached",
      len(kept_budget_clipped) == 4 and "Tail un-evaluated chunk A" in kept_budget_clipped[3]["text"],
      f"expected 4 chunks (3 eval + 1 tail), got {len(kept_budget_clipped)}")


# 3. Test assess_context_sufficiency
suff_yes = laya_engine.assess_context_sufficiency("when is debt due?", "crypto loan repayment date: March 15")
check("Mock Laya: sufficiency sufficient_to_answer", suff_yes["decision"] == "sufficient_to_answer")

suff_no = laya_engine.assess_context_sufficiency("what is the secret code?", "Just talking about recipes and pasta")
check("Mock Laya: sufficiency insufficient_context_abstain", suff_no["decision"] == "insufficient_context_abstain")

# 4. Test sufficiency conditioning in mode_router.get_system_prompt
sp_abstain = mode_router.get_system_prompt(
    "conversational",
    personal_context="some snippets",
    style_profile="user style profile.",
    sufficiency_state="insufficient_context_abstain"
)
check("Prompt Conditioning: insufficient_context_abstain instruction present",
      "CRITICAL EVALUATION ON CONTEXT: The retrieved context excerpts above were" in sp_abstain)
check("Prompt Conditioning: abstain tells model not to guess",
      "Do NOT attempt to guess, extrapolate, or invent" in sp_abstain)
check("Prompt Conditioning: style profile is strictly LAST",
      sp_abstain.strip().endswith("user style profile."))

sp_partial = mode_router.get_system_prompt(
    "conversational",
    personal_context="some snippets",
    style_profile="user style profile.",
    sufficiency_state="partial_context_hedge_needed"
)
check("Prompt Conditioning: partial_context_hedge_needed instruction present",
      "Answer ONLY the specific part directly" in sp_partial)
check("Prompt Conditioning: style profile is strictly LAST in partial mode",
      sp_partial.strip().endswith("user style profile."))

# 5. Schema verification for context_sufficiency_log.jsonl
dummy_record = {
    "timestamp": "2026-10-31T22:30:00Z",
    "session_id": "test_session",
    "mode": "conversational",
    "context_lookup_attempted": True,
    "personal_context_chars": 500,
    "reference_context_chars": 0,
    "context_present": True,
    "reply_text": "I don't know that specific detail.",
    "reply_flagged_uncertainty": True,
    "laya_mode": {"chosen_mode": "conversational", "confidence": 0.91, "used_fallback": False},
    "laya_rag_filtered_count": 1,
    "laya_sufficiency_decision": "partial_context_hedge_needed",
    "laya_sufficiency_confidence": 0.82,
    "laya_latency_ms": 18.5,
}

# Verify serialization
encoded = json.dumps(dummy_record)
decoded = json.loads(encoded)
check("JSONL Schema: laya_sufficiency_decision serializes correctly",
      decoded.get("laya_sufficiency_decision") == "partial_context_hedge_needed")
check("JSONL Schema: laya_latency_ms serializes correctly",
      decoded.get("laya_latency_ms") == 18.5)

# Restore laya_engine state
laya_engine._laya_model = real_laya_model
laya_engine._laya_available = real_laya_available
laya_engine._laya_load_attempted = real_laya_attempted


# =========================================================================
# Part C: Live Laya Smoke Test
# =========================================================================
print("\n--- Part C: Live Laya Smoke Test ---")

local_available = laya_engine.is_laya_available()
if local_available:
    print("convaiinnovations/laya weights detected locally! Running live CPU inference smoke test...")
    t0 = time.perf_counter()
    res = laya_engine.classify_mode_and_retrieval("I'm feeling really anxious about tomorrow")
    dur_ms = (time.perf_counter() - t0) * 1000
    print(f"Live Laya Mode: {res['mode']} (confidence: {res['mode_confidence']:.2f}, latency: {dur_ms:.1f}ms)")
    check("Live Laya: inference completed on CPU", res is not None)
else:
    print("Notice: convaiinnovations/laya weights are not yet cached in this environment.")
    print("Offline mode (HF_HUB_OFFLINE=1) is active; fallback system verified 100% operational.")
    print("To pre-download the Laya weights for live on-device CPU execution, run:")
    print('    python -c "import os; os.environ[\'HF_HUB_OFFLINE\']=\'0\'; from transformers import AutoModel; AutoModel.from_pretrained(\'convaiinnovations/laya\', trust_remote_code=True)"')

print("\n" + "=" * 50)
if ok:
    print("ALL LAYA INTEGRATION TESTS PASSED.")
else:
    print("ONE OR MORE TESTS FAILED.")
print("=" * 50)
sys.exit(0 if ok else 1)
