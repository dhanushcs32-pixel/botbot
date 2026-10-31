"""
Bot Bot — NPU stateful-pipeline leak test

Empirically checks whether npu_server.py's global LLMPipeline object leaks
KV-cache/context between separate /api/chat calls, despite code review
showing no start_chat()/finish_chat() calls anywhere (which per OpenVINO
GenAI docs should mean each generate() call is fully independent).

How it works:
  1. Send a first request containing a fictitious, highly distinctive
     planted detail (a nonsense codeword + a specific fake fact) that has
     no plausible way of appearing in a reply unless it leaked forward
     from this exact call.
  2. Send a second, completely unrelated request — a generic question
     with no mention of the planted detail, using a plain/minimal system
     prompt (not the bot's real one) so there's no legitimate channel for
     the codeword to appear via retrieval/memory.
  3. Check the second reply for any trace of the planted codeword or fact.
  4. Repeat for several trials with different planted details, since a
     single clean run doesn't rule out intermittent leakage.

Run directly against npu_server.py (bypasses main.py entirely) so results
aren't confounded by anything in the RAG/history/prompt-trim pipeline —
this isolates the pipeline object itself as the only variable.

Usage:
    python test_state_leak.py
    python test_state_leak.py --trials 5
    python test_state_leak.py --url http://localhost:11435/api/chat
"""

import argparse
import random
import string
import sys
import time

import httpx

DEFAULT_URL = "http://localhost:11435/api/chat"
MODEL_NAME = "botbot"
REQUEST_TIMEOUT = 180  # generous — NPU calls have been observed at 40-105s


def _random_codeword() -> str:
    """A nonsense token that could never appear in a real reply by chance."""
    letters = "".join(random.choices(string.ascii_uppercase, k=6))
    digits = "".join(random.choices(string.digits, k=4))
    return f"ZQPLANT-{letters}-{digits}"


def _call_npu(url: str, messages: list) -> tuple[str, float]:
    """Sends one /api/chat request, returns (reply_text, elapsed_seconds)."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
    }
    start = time.monotonic()
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
    elapsed = time.monotonic() - start
    return data["message"]["content"].strip(), elapsed


def run_trial(url: str, trial_num: int) -> bool:
    """
    Runs one plant-then-probe pair. Returns True if CLEAN (no leak),
    False if leakage was detected.
    """
    codeword = _random_codeword()
    fake_fact = f"the secret launch date is {random.randint(1, 28)}/13/2099"

    print(f"\n{'=' * 60}")
    print(f"TRIAL {trial_num}")
    print(f"{'=' * 60}")
    print(f"Planted codeword: {codeword}")
    print(f"Planted fake fact: {fake_fact}")

    # --- Call 1: plant the detail ---
    plant_messages = [
        {
            "role": "system",
            "content": (
                "You are a plain assistant. Just acknowledge what the user "
                "tells you briefly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Remember this exact codeword: {codeword}. Also remember "
                f"this: {fake_fact}. Just say OK once you've noted it."
            ),
        },
    ]
    plant_reply, plant_time = _call_npu(url, plant_messages)
    print(f"\n[Call 1 - plant] ({plant_time:.1f}s)")
    print(f"Reply: {plant_reply[:200]}")

    # --- Call 2: completely unrelated probe, fresh system prompt ---
    probe_messages = [
        {
            "role": "system",
            "content": (
                "You are a plain assistant. Answer the user's question "
                "directly and briefly."
            ),
        },
        {
            "role": "user",
            "content": "What's a good way to organize a small bookshelf?",
        },
    ]
    probe_reply, probe_time = _call_npu(url, probe_messages)
    print(f"\n[Call 2 - unrelated probe] ({probe_time:.1f}s)")
    print(f"Reply: {probe_reply[:300]}")

    # --- Check for leakage ---
    leaked_codeword = codeword in probe_reply
    leaked_fact_fragment = fake_fact.split(" is ")[-1] in probe_reply  # e.g. "13/13/2099"

    if leaked_codeword or leaked_fact_fragment:
        print(f"\n*** LEAK DETECTED ***")
        if leaked_codeword:
            print(f"  - Codeword '{codeword}' appeared in the unrelated reply.")
        if leaked_fact_fragment:
            print(f"  - Planted fake date fragment appeared in the unrelated reply.")
        return False

    print(f"\nClean — no trace of planted detail in the unrelated reply.")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default=DEFAULT_URL, help=f"npu_server.py endpoint (default: {DEFAULT_URL})"
    )
    parser.add_argument(
        "--trials", type=int, default=3, help="Number of plant-then-probe trials to run (default: 3)"
    )
    args = parser.parse_args()

    print(f"NPU state-leak diagnostic")
    print(f"Target: {args.url}")
    print(f"Trials: {args.trials}")

    results = []
    for i in range(1, args.trials + 1):
        try:
            clean = run_trial(args.url, i)
            results.append(clean)
        except httpx.HTTPError as e:
            print(f"\n[ERROR] Trial {i} failed to reach {args.url}: {e}")
            print("Is npu_server.py running? Check the URL and try again.")
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    clean_count = sum(results)
    print(f"{clean_count}/{len(results)} trials clean.")

    if all(results):
        print(
            "\nNo leakage detected across all trials. Consistent with code "
            "review + OpenVINO GenAI docs: generate() without start_chat()/"
            "finish_chat() should be fully stateless per call. This does "
            "NOT prove leakage is impossible under all conditions (e.g. "
            "concurrent/overlapping requests weren't tested here — only "
            "sequential calls), but it's a real empirical data point against "
            "the NPU pipeline being a root cause of the formatting issue."
        )
    else:
        print(
            "\nLEAKAGE DETECTED in at least one trial. This means the "
            "'stateless per call' assumption is wrong in practice for this "
            "build/hardware, regardless of what start_chat()/finish_chat() "
            "code review suggested. This should become the top-priority "
            "investigation — it could explain cross-session bleed-through "
            "independent of anything in main.py's prompt assembly."
        )


if __name__ == "__main__":
    main()
