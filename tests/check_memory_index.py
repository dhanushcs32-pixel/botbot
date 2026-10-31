"""
Diagnostic script for memory.db chunk storage + hash-index health.
Matches the actual schema in memory.py:
  - rolling_summaries: chunk storage (chunk_index, upto_turn_id, last_hash)
  - session_summaries: one row per session, end-of-session summary
  - session_summary_tags: person/activity tags keyed off session_summaries

Run from C:\\botbot with:  python check_memory_index.py
"""
import sqlite3
import sys

DB_PATH = "memory.db"

def main():
    try:
        conn = sqlite3.connect(DB_PATH)
    except Exception as e:
        print(f"Could not open {DB_PATH}: {e}")
        sys.exit(1)

    cur = conn.cursor()

    print("=" * 60)
    print("1. Chunk storage (rolling_summaries): chunk_index gaps")
    print("=" * 60)
    try:
        cur.execute("""
            SELECT session_id, COUNT(*), MIN(chunk_index), MAX(chunk_index)
            FROM rolling_summaries
            GROUP BY session_id
        """)
        rows = cur.fetchall()
        if not rows:
            print("No rows in rolling_summaries - either no session has crossed "
                  "ROLLUP_TRIGGER_TURNS (12) yet, or rollups aren't firing/persisting.")
        else:
            gap_found = False
            for session_id, count, min_idx, max_idx in rows:
                expected = max_idx - min_idx + 1
                status = "OK" if expected == count else "GAP DETECTED"
                if status == "GAP DETECTED":
                    gap_found = True
                print(f"  session={session_id!r:30} chunks={count:3} "
                      f"min_idx={min_idx} max_idx={max_idx}  [{status}]")
            if gap_found:
                print("\n  -> Missing chunk_index values for some session. Rollup may be "
                      "skipping a chunk_index increment, or a UNIQUE(session_id, chunk_index) "
                      "upsert landed on the wrong index.")
            else:
                print("\n  -> All sessions have contiguous chunk_index ranges. Good.")

        cur.execute("""
            SELECT session_id, chunk_index, LENGTH(summary)
            FROM rolling_summaries
            WHERE LENGTH(summary) >= 6000
        """)
        oversized = cur.fetchall()
        if oversized:
            print("\n  Chunks at/over the 6000-char rollover threshold:")
            for sid, idx, length in oversized:
                print(f"    session={sid!r:30} chunk_index={idx} chars={length}")
            print("  (Expected for a session's LATEST chunk right before it rolls "
                  "to the next index. Concerning only if a NON-latest chunk is over "
                  "this size, or a session's latest chunk stays over this size across "
                  "repeated rollups without ever incrementing chunk_index.)")
    except sqlite3.OperationalError as e:
        print(f"  Query failed: {e}")

    print()
    print("=" * 60)
    print("2. session_summaries: does the source table have rows?")
    print("=" * 60)
    try:
        cur.execute("SELECT COUNT(*) FROM session_summaries")
        (count,) = cur.fetchone()
        print(f"  session_summaries row count: {count}")
        if count == 0:
            print("  -> No end-of-session summaries saved yet. Tags and "
                  "get_recent_summaries() will have nothing to work with "
                  "regardless of index health.")
    except sqlite3.OperationalError as e:
        print(f"  Query failed: {e}")

    print()
    print("=" * 60)
    print("3. Hash index (session_summary_tags): row count + type breakdown")
    print("=" * 60)
    try:
        cur.execute("SELECT COUNT(*) FROM session_summary_tags")
        (tag_count,) = cur.fetchone()
        print(f"  session_summary_tags row count: {tag_count}")
        if tag_count == 0:
            print("  -> Empty. If session_summaries has rows (check #2 above), "
                  "this means summaries are saved but never tagged - check that "
                  "save_session_summary()/backfill_summary_tags() is actually "
                  "being called with a non-empty known_people set.")
        else:
            cur.execute("""
                SELECT tag_type, COUNT(*) FROM session_summary_tags
                GROUP BY tag_type
            """)
            for tag_type, count in cur.fetchall():
                print(f"    tag_type={tag_type!r:12} count={count}")

            cur.execute("""
                SELECT tag_type, tag_value, COUNT(*) FROM session_summary_tags
                GROUP BY tag_type, tag_value
                ORDER BY COUNT(*) DESC LIMIT 15
            """)
            print("\n  Top tag values:")
            for tag_type, tag_value, count in cur.fetchall():
                print(f"    {tag_type:10} {tag_value!r:25} x{count}")
    except sqlite3.OperationalError as e:
        print(f"  Query failed: {e}")

    print()
    print("=" * 60)
    print("4. Orphan check: session_summaries rows with NO tags at all")
    print("=" * 60)
    try:
        cur.execute("""
            SELECT s.session_id FROM session_summaries s
            LEFT JOIN session_summary_tags t ON s.session_id = t.session_id
            WHERE t.session_id IS NULL
        """)
        orphans = [r[0] for r in cur.fetchall()]
        if orphans:
            print(f"  {len(orphans)} session(s) have a summary but zero tags:")
            for sid in orphans[:20]:
                print(f"    {sid}")
            if len(orphans) > 20:
                print(f"    ... and {len(orphans) - 20} more")
            print("  -> Expected only if their summary text genuinely produced no "
                  "person/activity tokens (short/generic summary) - otherwise these "
                  "are candidates for backfill_summary_tags().")
        else:
            print("  None - every session_summaries row has at least one tag.")
    except sqlite3.OperationalError as e:
        print(f"  Query failed: {e}")

    conn.close()

    print()
    print("=" * 60)
    print("5. Functional check: get_recent_summaries() against real data")
    print("=" * 60)
    try:
        from memory import get_recent_summaries
        no_query = get_recent_summaries()
        print(f"  get_recent_summaries() [no query] -> {len(no_query)} result(s)")

        conn2 = sqlite3.connect(DB_PATH)
        cur2 = conn2.cursor()
        cur2.execute("SELECT tag_value FROM session_summary_tags LIMIT 1")
        row = cur2.fetchone()
        conn2.close()
        sample_term = row[0] if row else "cooking"

        with_query = get_recent_summaries(query=sample_term)
        print(f"  get_recent_summaries(query={sample_term!r}) -> {len(with_query)} result(s)")
        if no_query and not with_query:
            print("  -> No-query works but query-based returns nothing even with a "
                  "real tag_value as the query. Points at a bug in "
                  "_session_ids_matching_tags() or the tokenize/filter step in "
                  "get_recent_summaries(), not at missing data.")
    except ImportError as e:
        print(f"  Could not import memory.py from this directory: {e}")
        print("  -> Make sure you're running this from C:\\botbot")
    except Exception as e:
        print(f"  get_recent_summaries() raised an exception: {e}")

    print()
    print("Done.")

if __name__ == "__main__":
    main()
