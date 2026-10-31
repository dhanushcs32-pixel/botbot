"""Prompt-budget verification (real Qwen tokenizer, no NPU server needed).

Run:  cd C:\\botbot && .\\venv\\Scripts\\python.exe tests\\test_prompt_budget.py

Part A: historical replay — reconstructs approximate prompts for the
crypto/debt turns (2026-08-27, session testtest) that triggered trimming
under the old char budget, and shows old-vs-new budget outcomes with REAL
Qwen token counts.
Part B: edge case — a genuinely oversized session must still trim correctly
(cascade fires, final real tokens land at/below the 5500 target).

Uses transformers AutoTokenizer on the local model dir (same tokenizer
family npu_server serves); production measures via /api/estimate_tokens.
"""
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import memory
import rag
import main

QWEN_DIR = r"c:\botbot\qwen2.5-7b-instruct-int4-ov"

ok = True
def check(name, cond, detail=""):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    ok = ok and cond


from transformers import AutoTokenizer
print("loading real Qwen tokenizer (local files)...")
_tok = AutoTokenizer.from_pretrained(QWEN_DIR, local_files_only=True)

def count_tokens(messages):
    """Mirror of npu_server /api/estimate_tokens: chat template + count."""
    prompt = _tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
    if not isinstance(prompt, str):
        prompt = str(prompt)
    return len(_tok(prompt)["input_ids"])

# ---------- Part A: historical replay of the crypto/debt sequence ----------
OLD_CHAR_CAP = 8316          # (6144-600)*1.5 — the old HARD_CAP_TARGET_CHARS
NEW_TARGET_TOKENS = main.PROMPT_TARGET_TOKENS

conn = sqlite3.connect(memory.DB_PATH)
rows = conn.execute(
    "SELECT id, role, content, timestamp FROM conversations "
    "WHERE session_id='testtest' AND timestamp LIKE '2026-08-27T18%' "
    "ORDER BY id"
).fetchall()
conn.close()

print(f"\nPart A — crypto/debt sequence: {len(rows)} rows on 2026-08-27 evening")
print(f"{'ts':<10} {'chars':>6} {'est_tok':>7} {'REAL_tok':>8}   old-trim  new-trim")
turn_stats = []
for turn_id, role, content, ts in rows:
    if role != "user":
        continue
    c2 = sqlite3.connect(memory.DB_PATH)
    hist = c2.execute(
        "SELECT role, content FROM conversations "
        "WHERE session_id='testtest' AND id < ? ORDER BY id DESC LIMIT 12",
        (turn_id,)).fetchall()
    c2.close()
    hist = list(reversed(hist))
    hist_msgs = [{"role": "assistant" if r == "assistant" else "user", "content": c}
                 for r, c in hist]
    # Approximate system prompt: current retrieval for this message (the
    # historical one differed slightly — acknowledged approximation).
    personal = (rag.search_personal_data(content, top_k=4)
                if not rag.is_trivial_message(content) else "")
    system_prompt = main.get_system_prompt(
        "conversational", personal_context=personal, style_profile="",
        session_summaries="", reference_context="",
        referenced_session_context="")
    messages = ([{"role": "system", "content": system_prompt}] + hist_msgs +
                [{"role": "user", "content": content}])
    total_chars = sum(len(m["content"]) for m in messages)
    real = count_tokens(messages)
    old_trim = total_chars > OLD_CHAR_CAP
    new_trim = real > NEW_TARGET_TOKENS
    turn_stats.append((ts[11:19], total_chars, real, old_trim, new_trim))
    print(f"{ts[11:19]:<10} {total_chars:>6} {int(total_chars/2.5):>7} {real:>8}   "
          f"{str(old_trim):<8}  {new_trim}")

if turn_stats:
    old_fired = sum(1 for t in turn_stats if t[3])
    new_fired = sum(1 for t in turn_stats if t[4])
    check("old char budget would have trimmed (reproduces the investigation)",
          old_fired > 0, f"{old_fired}/{len(turn_stats)} turns")
    check("new real-token budget trims fewer or none of these turns",
          new_fired < old_fired or new_fired == 0,
          f"{new_fired}/{len(turn_stats)} turns")
    max_real = max(t[2] for t in turn_stats)
    check("all reconstructed prompts fit the 5500-token target anyway",
          max_real <= NEW_TARGET_TOKENS, f"max real tokens = {max_real}")

# ---------- Part B: edge case — genuinely oversized context ----------
# Simulate a HEALTHY npu_server reporting its ceiling (the live target
# source), then a genuinely oversized retrieval result.
real_target_fn = main._get_prompt_target_tokens
main._get_prompt_target_tokens = lambda: 5500
real_search = main.search_personal_data
# Varied sentences — repetitive text compresses to ~5 chars/token under BPE,
# which would (correctly) fit the budget and never trigger the cascade.
_base = [f"Context detail number {i} covers unrelated topic {i * 7} with "
         f"varied wording and different structure, item {i}."
         for i in range(60)]
def huge_search(query, **kwargs):
    return " ".join(_base * 12)[:40000]
main.search_personal_data = huge_search
try:
    prep_big = main._prepare_prompt_sync(
        "budgetedge1",
        "tell me about this context please",
        measure_tokens=count_tokens)
finally:
    main.search_personal_data = real_search

real_big = prep_big.get("measured_tokens")
stages = [d["stage"] for d in prep_big["trim_details"]]
check("oversized context: trim cascade fired", prep_big["trimmed"] is True,
      f"stages={stages}")
check("oversized context: final real tokens at/below target (+150 slack)",
      real_big is not None and real_big <= NEW_TARGET_TOKENS + 150,
      f"final real tokens={real_big}")
check("oversized context: personal_context was the trimmed section",
      "personal_context" in stages,
      f"stages={stages}")
check("oversized context: messages structure intact",
      prep_big["messages"][0]["role"] == "system"
      and prep_big["messages"][-1]["role"] == "user")

# ---------- Part C: degraded ceiling (npu_server reports a smaller one) ----------
# This is the Fix-1-followup scenario: if npu_server ever loads without
# MAX_PROMPT_LEN or on a fallback device, health reports a smaller
# effective ceiling and main.py must budget against THAT number.
DEGRADED = 4096 - 644  # what main derives for a 4096 effective ceiling
main._get_prompt_target_tokens = lambda: DEGRADED
main.search_personal_data = huge_search
try:
    prep_deg = main._prepare_prompt_sync(
        "budgetedge1",
        "tell me about this context please",
        measure_tokens=count_tokens)
finally:
    main.search_personal_data = real_search
    main._get_prompt_target_tokens = lambda: DEGRADED  # degraded for Part C

real_deg = prep_deg.get("measured_tokens")
deg_stages = [d["stage"] for d in prep_deg["trim_details"]]
check("degraded ceiling: trim fired and used the reduced target",
      prep_deg["trimmed"] is True and "personal_context" in deg_stages
      and any(d.get("target_tokens") == DEGRADED for d in prep_deg["trim_details"]),
      f"stages={deg_stages}, target={DEGRADED}")
check("degraded ceiling: final real tokens at/below reduced target (+150)",
      real_deg is not None and real_deg <= DEGRADED + 150,
      f"final real tokens={real_deg}")

# restore the real target function after the degraded simulation
main._get_prompt_target_tokens = real_target_fn

# Restore the real DB path and confirm the small-prompt fast path: a prompt
# under the old conservative cap pays zero measurement overhead.
memory.DB_PATH = Path(r"c:\botbot\memory.db")
called = {"n": 0}
def counting_measure(messages):
    called["n"] += 1
    return count_tokens(messages)
prep_small = main._prepare_prompt_sync(
    "budgetedge1", "hi", measure_tokens=counting_measure)
check("small prompt: no measurement call, no trim (zero overhead path)",
      called["n"] == 0 and prep_small["trimmed"] is False,
      f"measure calls={called['n']}")

print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)

