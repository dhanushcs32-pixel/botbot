"""
Verification harness for:
  1. Sequential size-bounded chunk storage (maybe_rollup_session / rolling_summaries)
  2. Person + activity hash index (save_session_summary tagging + get_recent_summaries query)

Uses a throwaway sqlite DB (memory.db in this folder) and a mocked
summarize_fn / npu_lock, so no real NPU/network calls happen.
"""
import asyncio
import os
import sqlite3
import sys
from pathlib import Path

DB_FILE = Path(__file__).parent / "memory.db"
if DB_FILE.exists():
    DB_FILE.unlink()

import memory  # noqa: E402  (import after cleaning DB so init_db() creates fresh)

# ISOLATION FIX (2026-09-01): this test always *claimed* to use a throwaway
# DB here, but memory.py's DB_PATH points at the REAL c:\botbot\memory.db —
# the delete above touched a file nothing reads, and the test silently ran
# against live data (leaving chunktest1 sessions behind and colliding with
# real usage on the next run). Point memory at the throwaway explicitly.
memory.DB_PATH = DB_FILE
memory.init_db()

PASS = []
FAIL = []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


def insert_turns(session_id, n, content_len=50):
    """Directly inserts n user+assistant turn pairs (2n rows) via save_turn."""
    for i in range(n):
        memory.save_turn(session_id, f"user msg {i} " + ("x" * content_len),
                          f"bot reply {i} " + ("y" * content_len), mode="conversational")


async def fake_summarize_short(existing_summary, batch_dicts, npu_lock):
    """Returns a short deterministic summary — chunk should NOT roll over."""
    return (existing_summary + f" [folded {len(batch_dicts)} turns]").strip()


async def fake_summarize_long(existing_summary, batch_dicts, npu_lock):
    """Returns a summary padded past ROLLUP_CHUNK_MAX_CHARS so the NEXT
    rollup call is forced to roll into a new chunk."""
    return existing_summary + ("z" * 6500)


# ============================================================
# TEST 1: chunk rollover — current chunk exceeds ROLLUP_CHUNK_MAX_CHARS,
# next maybe_rollup_session() call should start chunk_index=1
# ============================================================
async def test_chunk_rollover():
    print("\n=== TEST 1: chunk rollover ===")
    sid = "chunktest1"

    # First rollup: not enough turns yet (need ROLLUP_TRIGGER_TURNS=12 candidate rows)
    insert_turns(sid, 3)  # 6 rows
    fired = await memory.maybe_rollup_session(sid, fake_summarize_long, npu_lock=None)
    check("no rollup before threshold", fired is False)

    # Cross the threshold: need > 12 un-folded rows, KEEP_RECENT_RAW_TURNS=6 kept back
    insert_turns(sid, 6)  # now 18 rows total, well over 12
    fired = await memory.maybe_rollup_session(sid, fake_summarize_long, npu_lock=None)
    check("first rollup fires once threshold crossed", fired is True)

    latest = memory._get_latest_chunk(sid)
    check("chunk 0 exists after first rollup", latest is not None and latest[0] == 0)
    chunk0_len = len(latest[1]) if latest else 0
    check("chunk 0 summary is now oversized (fake returned 6500+ chars)",
          chunk0_len >= memory.ROLLUP_CHUNK_MAX_CHARS, detail=f"len={chunk0_len}")

    # Add enough new turns to trigger another rollup — should detect chunk 0
    # is oversized and roll into chunk_index=1 with a FRESH ("") existing summary.
    insert_turns(sid, 6)
    fired = await memory.maybe_rollup_session(sid, fake_summarize_short, npu_lock=None)
    check("second rollup fires", fired is True)

    latest2 = memory._get_latest_chunk(sid)
    check("rolled into chunk_index=1", latest2 is not None and latest2[0] == 1,
          detail=f"got chunk_index={latest2[0] if latest2 else None}")
    check("chunk 1 started fresh (short, not carrying chunk 0's bulk forward)",
          latest2 is not None and len(latest2[1]) < 1000, detail=f"len={len(latest2[1]) if latest2 else -1}")

    # chunk 0 should be UNTOUCHED (still the oversized one) since we only ever
    # update the latest chunk in place, never rewrite an old one.
    conn = sqlite3.connect(memory.DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT chunk_index, summary FROM rolling_summaries WHERE session_id = ? ORDER BY chunk_index", (sid,))
    all_chunks = cur.fetchall()
    conn.close()
    check("exactly 2 chunk rows exist (0 and 1)", [c[0] for c in all_chunks] == [0, 1],
          detail=str([c[0] for c in all_chunks]))
    check("chunk 0 preserved, not merged/overwritten",
          len(all_chunks[0][1]) >= memory.ROLLUP_CHUNK_MAX_CHARS)

    return all_chunks


# ============================================================
# TEST 2: get_current_session_summary joins chunks in order
# ============================================================
def test_reconstruction():
    print("\n=== TEST 2: chunk reconstruction (get_current_session_summary) ===")
    sid = "chunktest1"  # reuse session from test 1, now has chunk 0 + chunk 1
    full = memory.get_current_session_summary(sid)
    check("reconstruction returns non-None", full is not None)
    check("reconstruction contains both chunks joined",
          full is not None and "z" in full and "[folded" in full,
          detail=f"len={len(full) if full else 0}")

    # Order check: chunk 0's content should appear before chunk 1's in the joined string
    if full:
        idx_chunk0_marker = full.find("z")
        idx_chunk1_marker = full.find("[folded")
        check("chunk 0 content appears before chunk 1 content (correct order)",
              idx_chunk0_marker != -1 and idx_chunk1_marker != -1 and idx_chunk0_marker < idx_chunk1_marker)

    # Session with no chunks at all -> None
    none_result = memory.get_current_session_summary("nonexistent_session_xyz")
    check("session with no chunks returns None", none_result is None)


# ============================================================
# TEST 3: idempotency still holds within chunk system (same batch = no-op)
# ============================================================
async def test_idempotency():
    print("\n=== TEST 3: idempotent rollup (repeated call, no new turns) ===")
    sid = "chunktest1"
    fired = await memory.maybe_rollup_session(sid, fake_summarize_short, npu_lock=None)
    check("repeated rollup call with no new turns is a no-op", fired is False)


# ============================================================
# TEST 4: person + activity tagging on save_session_summary
# ============================================================
def test_tagging():
    print("\n=== TEST 4: person + activity tagging ===")
    # NOTE: session ids here deliberately do NOT contain the substring
    # "test" (e.g. "tagcheck_*" not "tagtest_*") — is_test_session()
    # treats any session_id containing "test" as an isolated sandbox and
    # get_recent_summaries() excludes those from its results by design.
    # Using "test"-named ids here would make Test 5 fail for a totally
    # different reason (isolation filtering) than what Test 5 is actually
    # checking (query-aware retrieval).
    known_people = {"sooryan", "sharadhi", "sampada"}

    memory.save_session_summary(
        "tagcheck_sooryan",
        "User discussed feeling anxious about an upcoming trip with Sooryan, "
        "and mentioned wanting to go hiking next weekend.",
        known_people=known_people,
    )
    memory.save_session_summary(
        "tagcheck_sharadhi",
        "User talked about a disagreement with Sharadhi regarding weekend plans.",
        known_people=known_people,
    )
    memory.save_session_summary(
        "tagcheck_neither",
        "User asked general questions about cooking pasta and grocery shopping.",
        known_people=known_people,
    )

    conn = sqlite3.connect(memory.DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT tag_type, tag_value FROM session_summary_tags WHERE session_id = ?", ("tagcheck_sooryan",))
    tags = cur.fetchall()
    conn.close()
    tag_dict = {}
    for t, v in tags:
        tag_dict.setdefault(t, set()).add(v)

    check("sooryan session tagged with person:sooryan",
          "sooryan" in tag_dict.get("person", set()), detail=str(tag_dict.get("person")))
    check("sooryan session NOT tagged with unrelated people",
          "sharadhi" not in tag_dict.get("person", set()))
    check("sooryan session has activity tags (e.g. 'hiking' or 'trip')",
          bool(tag_dict.get("activity")) and any(
              w in tag_dict["activity"] for w in ("hiking", "trip", "weekend", "anxious")),
          detail=str(tag_dict.get("activity")))
    # person name should be excluded from activity tags (name shouldn't double as activity keyword)
    check("person name excluded from activity tags",
          "sooryan" not in tag_dict.get("activity", set()))

    # wholesale replace check: re-saving should replace, not accumulate duplicates.
    # Text below deliberately avoids the literal word "Sooryan" (even in
    # negation) so a stale person:sooryan tag can't reappear via an
    # accidental substring match — that would give a false pass/fail
    # unrelated to whether replace-vs-accumulate logic is actually correct.
    memory.save_session_summary(
        "tagcheck_sooryan",
        "User now only talks about cooking and grocery shopping, nothing else today.",
        known_people=known_people,
    )
    conn = sqlite3.connect(memory.DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT tag_type, tag_value FROM session_summary_tags WHERE session_id = ?", ("tagcheck_sooryan",))
    tags2 = cur.fetchall()
    conn.close()
    tag_dict2 = {}
    for t, v in tags2:
        tag_dict2.setdefault(t, set()).add(v)
    check("re-save replaces old tags (sooryan/trip tags gone after re-save)",
          "sooryan" not in tag_dict2.get("person", set()) and "trip" not in tag_dict2.get("activity", set()),
          detail=str(tag_dict2))


# ============================================================
# TEST 5: get_recent_summaries query-based reordering
# ============================================================
def test_query_retrieval():
    print("\n=== TEST 5: get_recent_summaries query-aware retrieval ===")
    # tagcheck_sharadhi should be findable via a query mentioning "sharadhi"
    # even though tagcheck_neither was saved more recently in real time... but
    # since we save in insertion order and ORDER BY id DESC LIMIT 200 governs
    # the eligible pool, let's just confirm query-match ordering vs no-query.

    no_query = memory.get_recent_summaries(exclude_session_id=None, limit=5, query=None)
    check("no-query call returns some summaries", len(no_query) > 0, detail=str(len(no_query)))

    with_query = memory.get_recent_summaries(exclude_session_id=None, limit=5, query="what about sharadhi and weekend plans")
    check("query call returns some summaries", len(with_query) > 0)

    # The sharadhi summary should be the FIRST result when querying "sharadhi",
    # even if it wasn't the most recently-saved session.
    sharadhi_summary = memory.get_summary_for_session("tagcheck_sharadhi")
    check("sharadhi summary retrievable directly (sanity)", sharadhi_summary is not None)
    check("sharadhi summary is first (or present) in query-matched results",
          sharadhi_summary in with_query,
          detail=f"with_query={with_query}")
    if sharadhi_summary in with_query:
        check("sharadhi summary is FIRST in query-matched ordering (matched sessions surface first)",
              with_query[0] == sharadhi_summary,
              detail=f"first={with_query[0]!r}")


# ============================================================
# TEST 6: is_test_session isolation still respected by get_recent_summaries
# ============================================================
def test_isolation_still_respected():
    print("\n=== TEST 6: test-session isolation unaffected by query matching ===")
    memory.save_session_summary("faketest_session", "User discussed sooryan and hiking a lot here.",
                                 known_people={"sooryan"})
    results = memory.get_recent_summaries(exclude_session_id=None, limit=20, query="sooryan hiking")
    faketest_summary = memory.get_summary_for_session("faketest_session")
    check("test-named session excluded from results even when query matches its tags",
          faketest_summary not in results)


async def main():
    await test_chunk_rollover()
    test_reconstruction()
    await test_idempotency()
    test_tagging()
    test_query_retrieval()
    test_isolation_still_respected()

    print(f"\n{'='*50}\nRESULTS: {len(PASS)} passed, {len(FAIL)} failed\n{'='*50}")
    if FAIL:
        print("FAILED CHECKS:")
        for f in FAIL:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
