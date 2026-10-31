"""
Bot Bot — Context Sufficiency Log Analyzer

Reads context_sufficiency_log.jsonl (written by main.py's generate_reply(),
one JSON record per turn) and reports whether
CONTEXT_SUFFICIENCY_INSTRUCTION (mode_router.py) is actually changing
behavior on real traffic, per PROJECT_STATE.txt item 1.

Usage:
    python analyze_sufficiency_log.py [path_to_log.jsonl]

Defaults to context_sufficiency_log.jsonl in the current directory (same
place main.py writes it, since it runs from C:\\botbot).

What this checks, and why:

1. Hedge rate by context_present group
   If the instruction is working, turns where context_present=False
   should hedge ("I don't know") noticeably more often than turns where
   context_present=True. If both groups hedge at a similar rate, the
   instruction likely isn't influencing behavior much either way.

2. Candidate confabulations
   context_present=False (or True but very little context: a lookup ran
   and came back thin) AND reply does NOT read as uncertain. These are
   the turns most worth manually reading — the model answered
   confidently with little or nothing to answer from.

3. Candidate over-caution / false abstention
   context_present=True with a LARGE amount of context AND the reply DOES
   read as uncertain. Possible sign the instruction is too conservative
   and is causing the model to hedge even when it had what it needed —
   costs usefulness even though it's not actively harmful.

4. Breakdown by mode
   Supportive/guidance/conversational turns can have very different
   baseline hedge rates; worth seeing separately rather than only as one
   pooled number.

This is a diagnostic tool, not a verdict-maker: candidate lists in
sections 2/3 need a human to actually read them and judge true/false
positive. The point is narrowing "read every reply" down to "read the
~10-30 replies that are actually suspicious".
"""

import json
import sys
from collections import defaultdict

# A record with more than this many personal+reference context chars
# counts as "well-resourced" for the false-abstention check (section 3).
# Rough heuristic, not tuned against anything yet — adjust once you've
# read a batch of real logs and have a sense of typical retrieval sizes.
WELL_RESOURCED_CHAR_THRESHOLD = 400

# A record with fewer than this many chars (but context_present=True —
# i.e. retrieval ran and found *something*, just very little) counts as
# "thin" for the candidate-confabulation check (section 2).
THIN_CONTEXT_CHAR_THRESHOLD = 60


def load_records(path: str) -> list[dict]:
    records = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
    if skipped:
        print(f"(skipped {skipped} malformed line(s) in log)")
    return records


def pct(n: int, d: int) -> str:
    if d == 0:
        return "n/a (0 turns)"
    return f"{n}/{d} ({100 * n / d:.1f}%)"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "context_sufficiency_log.jsonl"

    try:
        records = load_records(path)
    except FileNotFoundError:
        print(f"No log file found at '{path}'.")
        print("Run the server and have a few real conversations first —")
        print("this file is written automatically by generate_reply() in main.py.")
        return

    if not records:
        print(f"'{path}' exists but has no valid records yet.")
        return

    # Only turns where retrieval was actually attempted are meaningful for
    # this analysis — trivial messages ("ok", "lol") never call
    # search_personal_data at all, so including them would just dilute
    # every ratio below with irrelevant turns.
    attempted = [r for r in records if r.get("context_lookup_attempted")]
    skipped_trivial = len(records) - len(attempted)

    print(f"Total logged turns: {len(records)}")
    print(f"  Trivial (no retrieval attempted, excluded below): {skipped_trivial}")
    print(f"  Retrieval attempted (analyzed below): {len(attempted)}")
    print()

    if not attempted:
        print("No non-trivial turns logged yet — nothing to analyze.")
        return

    with_context = [r for r in attempted if r.get("context_present")]
    without_context = [r for r in attempted if not r.get("context_present")]

    hedged_with = sum(1 for r in with_context if r.get("reply_flagged_uncertainty"))
    hedged_without = sum(1 for r in without_context if r.get("reply_flagged_uncertainty"))

    print("=== 1. Hedge rate by context_present ===")
    print(f"  context_present=True  : hedged {pct(hedged_with, len(with_context))}")
    print(f"  context_present=False : hedged {pct(hedged_without, len(without_context))}")
    if with_context and without_context:
        rate_with = hedged_with / len(with_context)
        rate_without = hedged_without / len(without_context)
        if rate_without <= rate_with:
            print(
                "  ⚠ No-context turns are NOT hedging more than context turns — "
                "the instruction may not be doing much. Worth reading section 2."
            )
        else:
            print("  ✓ No-context turns hedge more often, as expected if the instruction is working.")
    print()

    candidates_confab = [
        r for r in attempted
        if not r.get("reply_flagged_uncertainty")
        and (
            not r.get("context_present")
            or (r.get("personal_context_chars", 0) + r.get("reference_context_chars", 0)) < THIN_CONTEXT_CHAR_THRESHOLD
        )
    ]

    print(f"=== 2. Candidate confabulations (read these) — {len(candidates_confab)} found ===")
    print("  (no/thin context, but reply did NOT read as uncertain — most worth a manual read)")
    for r in candidates_confab[:15]:
        print(f"  --- session '{r.get('session_id')}' [{r.get('timestamp')}] ---")
        print(f"  user: {r.get('prompt_message_preview')}")
        print(f"  bot:  {r.get('reply_text')}")
        print()
    if len(candidates_confab) > 15:
        print(f"  ...and {len(candidates_confab) - 15} more (showing first 15)")
    print()

    candidates_overcautious = [
        r for r in attempted
        if r.get("reply_flagged_uncertainty")
        and (r.get("personal_context_chars", 0) + r.get("reference_context_chars", 0)) >= WELL_RESOURCED_CHAR_THRESHOLD
    ]

    print(f"=== 3. Candidate over-caution / false abstention — {len(candidates_overcautious)} found ===")
    print("  (plenty of context available, but reply hedged anyway — possible false abstention)")
    for r in candidates_overcautious[:10]:
        print(f"  --- session '{r.get('session_id')}' [{r.get('timestamp')}] ---")
        print(f"  user: {r.get('prompt_message_preview')}")
        print(f"  bot:  {r.get('reply_text')}")
        print()
    if len(candidates_overcautious) > 10:
        print(f"  ...and {len(candidates_overcautious) - 10} more (showing first 10)")
    print()

    print("=== 4. Breakdown by mode ===")
    by_mode = defaultdict(lambda: {"total": 0, "hedged": 0})
    for r in attempted:
        m = r.get("mode", "unknown")
        by_mode[m]["total"] += 1
        if r.get("reply_flagged_uncertainty"):
            by_mode[m]["hedged"] += 1
    for m, stats in sorted(by_mode.items()):
        print(f"  {m:15s} hedged {pct(stats['hedged'], stats['total'])}")


if __name__ == "__main__":
    main()
