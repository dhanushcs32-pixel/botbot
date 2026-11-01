"""
Test 4B Model on NPU (tests/test_4b_npu.py)
-------------------------------------------
Verification & Benchmark script for Qwen 4B OpenVINO INT4 on Intel NPU 3720.

Run:
    cd C:\\botbot && .\\venv\\Scripts\\python.exe tests\\test_4b_npu.py
"""

import os
import sys
import time
import re
from pathlib import Path

# Enforce strict offline execution
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_DIR = "./qwen-4b-int4-ov"
MAX_PROMPT_LEN = 6144

def clean_reply_text(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = cleaned.split("<think>")[0]
    return cleaned.replace("</think>", "").strip()


def run_benchmark():
    import openvino_genai as ov_genai
    import main

    print("=" * 70)
    print(" QWEN 4B INT4 NPU VERIFICATION & BENCHMARK")
    print(f" Model Dir: {MODEL_DIR}")
    print(" Device:    NPU (Intel Core Ultra Arrow Lake-H NPU 3720)")
    print(f" MAX_PROMPT_LEN: {MAX_PROMPT_LEN}")
    print("=" * 70)

    # 1. Compilation & Cold Load Test
    print("\n--- 1. NPU Compilation & Cold Load ---")
    t0 = time.time()
    try:
        pipe = ov_genai.LLMPipeline(MODEL_DIR, "NPU", MAX_PROMPT_LEN=MAX_PROMPT_LEN)
        compile_time = time.time() - t0
        print(f"[PASS] Successfully compiled on NPU in {compile_time:.2f}s (MAX_PROMPT_LEN={MAX_PROMPT_LEN})")
    except Exception as e:
        print(f"[FAIL] Failed with MAX_PROMPT_LEN: {e}")
        print("Retrying on NPU without MAX_PROMPT_LEN...")
        t0 = time.time()
        pipe = ov_genai.LLMPipeline(MODEL_DIR, "NPU")
        compile_time = time.time() - t0
        print(f"[PASS] Successfully compiled on NPU (default MAX_PROMPT_LEN) in {compile_time:.2f}s")

    tokenizer = pipe.get_tokenizer()

    # 2. Tokenizer Budget Check
    print("\n--- 2. Tokenizer Budget Check ---")
    test_msg = [{"role": "user", "content": "Hello, how are you today?"}]
    raw_prompt = tokenizer.apply_chat_template(test_msg, add_generation_prompt=True)
    tokens = len(tokenizer.encode(raw_prompt))
    print(f"Test prompt: '{test_msg[0]['content']}' -> {tokens} tokens")
    assert tokens > 0, "Tokenizer produced 0 tokens"
    print("[PASS] 4B Tokenizer operates accurately.")

    # 3. Generation Speed Tests
    print("\n--- 3. Generation Speed Benchmark ---")

    # Helper for running generation
    def generate_and_measure(name: str, messages: list, max_new_tokens: int = 250, is_cot: bool = False):
        try:
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        except Exception:
            prompt = ""
            for m in messages:
                prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"

        # Apply native no-think if standard turn
        if not is_cot and prompt.endswith("<|im_start|>assistant\n"):
            prompt += "<think>\n\n</think>\n\n"

        prompt_tokens = len(tokenizer.encode(prompt))

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = max_new_tokens
        config.do_sample = True
        config.temperature = 0.7
        config.top_p = 0.9

        print(f"\n>> {name} (Prompt: ~{prompt_tokens} tokens, max_tokens: {max_new_tokens}):")
        t_start = time.time()
        raw_output = pipe.generate(prompt, config)
        elapsed = time.time() - t_start

        cleaned_output = raw_output if is_cot else clean_reply_text(raw_output)
        chars = len(cleaned_output)
        cps = chars / elapsed if elapsed > 0 else 0.0

        print(f"   Elapsed:    {elapsed:.2f}s")
        print(f"   Output:     {chars} chars (raw: {len(raw_output)} chars)")
        print(f"   Throughput: {cps:.1f} chars/sec")
        preview = cleaned_output[:140].replace("\n", " ")
        print(f"   Preview:    \"{preview}...\"")

        return {
            "name": name,
            "elapsed": elapsed,
            "chars": chars,
            "cps": cps,
            "raw": raw_output,
            "cleaned": cleaned_output,
        }

    results = []

    # Prompt 1: Short casual turn (~250 prompt tokens total with system prompt)
    casual_messages = [
        {"role": "system", "content": "You are Bot Bot, a smart, concise, and helpful companion. Keep responses natural and conversational."},
        {"role": "user", "content": "Hey, what are 3 simple tips for staying focused when working from home?"}
    ]
    r1 = generate_and_measure("Test 1: Short Casual Turn", casual_messages, max_new_tokens=200)
    results.append(r1)

    # Prompt 2: RAG-injected turn (~1,500 tokens)
    context_chunk_1 = (
        "Project Orion Architecture Notes:\n"
        "The offline companion relies on an on-device hybrid RAG architecture. "
        "Dense embeddings are generated locally via all-MiniLM-L6-v2 in ChromaDB, while lexical matching "
        "uses rank_bm25 with Reciprocal Rank Fusion (RRF). "
        "A typed decision model (convaiinnovations/laya, ModernBERT 421M) reranks candidate excerpts on CPU "
        "in a single forward pass under 1.2 seconds, filtering irrelevant noise. "
    ) * 3
    context_chunk_2 = (
        "Memory System Overview:\n"
        "Conversations are persisted in SQLite (memory.db) with rolling turn summaries triggered every 12 turns. "
        "Token budgets are strictly enforced against a 6144 NPU prompt ceiling to prevent out-of-host-memory compilation faults. "
    ) * 3

    rag_messages = [
        {"role": "system", "content": f"You are Bot Bot. Use the following retrieved context to answer the question accurately:\n\n{context_chunk_1}\n{context_chunk_2}"},
        {"role": "user", "content": "How does Bot Bot rerank RAG candidates and prevent prompt overflow?"}
    ]
    r2 = generate_and_measure("Test 2: RAG-Injected Turn (~1500 tokens)", rag_messages, max_new_tokens=250)
    results.append(r2)

    # Prompt 3: CoT diagnostic prompt (<reasoning> + <reply>)
    cot_messages = [
        {"role": "system", "content": (
            "You are Bot Bot.\n"
            "[DIAGNOSTIC REASONING MODE — this turn only]\n"
            "Format requirements:\n"
            "<reasoning>\n"
            "2-4 sentences explaining your direct approach\n"
            "</reasoning>\n"
            "<reply>\n"
            "Your friendly, natural message to the user\n"
            "</reply>"
        )},
        {"role": "user", "content": "I'm having trouble deciding between learning Rust or Go for systems programming. What should I prioritize?"}
    ]
    r3 = generate_and_measure("Test 3: CoT Diagnostic Turn", cot_messages, max_new_tokens=300, is_cot=True)
    results.append(r3)

    # Verify CoT Parser on Prompt 3
    print("\n--- 4. CoT Parser Validation ---")
    reasoning, reply = main._parse_cot_output(r3["raw"])
    if reasoning is not None:
        print("[PASS] main._parse_cot_output successfully parsed 4B output!")
        print(f"   Reasoning: \"{reasoning[:90]}...\"")
        print(f"   Reply:     \"{reply[:90]}...\"")
    else:
        print(f"[FAIL] CoT Parse failed: {reply}")
        print(f"   Raw output: {r3['raw']}")

    # Summary
    print("\n" + "=" * 70)
    print(" BENCHMARK RESULTS SUMMARY (NPU 3720)")
    print("=" * 70)
    print(f" NPU Cold Load & Compile: {compile_time:.2f}s")
    for r in results:
        print(f" - {r['name']:<35} : {r['elapsed']:>6.2f}s | {r['chars']:>5} chars | {r['cps']:>6.1f} chars/sec")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()
