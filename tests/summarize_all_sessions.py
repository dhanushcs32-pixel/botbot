"""
One-off backfill: calls POST /summarize for every session that has
conversation turns but no row in session_summaries yet, then polls
GET /summarize/status/{session_id} until each one finishes.

/summarize now returns immediately with "queued" rather than blocking
until the summary is done — a single large session (415 turns / ~165K
chars observed in practice) can take upwards of an hour across ~25-30
chained NPU calls, which no single HTTP request timeout survives
reliably. Polling avoids needing one giant timeout at all.

Run this from the botbot project directory (same folder as main.py,
memory.py, secret_key.py) with the server already running.

Usage:
    python summarize_all_sessions.py
    python summarize_all_sessions.py --base-url http://localhost:8000
    python summarize_all_sessions.py --include-test   # also summarize
                                                        # sessions whose
                                                        # id contains
                                                        # "test"
    python summarize_all_sessions.py --force           # re-summarize
                                                        # sessions that
                                                        # already have a
                                                        # summary too
"""

import argparse
import time

import httpx

from secret_key import API_KEYS
from memory import get_db_connection, is_test_session

API_KEY = API_KEYS.get("primary", "")


def get_sessions_needing_summary(include_test: bool, force: bool) -> list[str]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if force:
            cursor.execute("SELECT DISTINCT session_id FROM conversations")
            session_ids = [row[0] for row in cursor.fetchall()]
        else:
            cursor.execute(
                """
                SELECT DISTINCT c.session_id
                FROM conversations c
                LEFT JOIN session_summaries s ON c.session_id = s.session_id
                WHERE s.session_id IS NULL
                """
            )
            session_ids = [row[0] for row in cursor.fetchall()]

    if not include_test:
        session_ids = [sid for sid in session_ids if not is_test_session(sid)]

    return sorted(session_ids)


def summarize_and_wait(
    client: httpx.Client, base_url: str, session_id: str, force: bool,
    poll_interval: float, max_wait: float,
) -> tuple[bool, str]:
    """Queues a session for summarization, then polls status until it's
    done, errors, or max_wait is exceeded. Returns (success, reason)."""
    resp = client.post(
        f"{base_url}/summarize",
        headers={"X-API-Key": API_KEY},
        json={"session_id": session_id, "force": force},
    )
    resp.raise_for_status()
    queue_status = resp.json().get("status")

    if queue_status == "skipped":
        return True, "already summarized"
    if queue_status not in ("queued", "already_in_progress"):
        return False, f"unexpected queue response: {queue_status}"

    start = time.time()
    while True:
        resp = client.get(
            f"{base_url}/summarize/status/{session_id}",
            headers={"X-API-Key": API_KEY},
        )
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")

        if status == "done":
            return True, "done"
        if status == "error":
            return False, data.get("detail", "unknown error")

        if time.time() - start > max_wait:
            return False, f"gave up after {int(max_wait)}s waiting (last status: {status})"

        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--delay", type=float, default=1.0,
                         help="seconds to wait between sessions (default 1.0)")
    parser.add_argument("--poll-interval", type=float, default=5.0,
                         help="seconds between status polls while a "
                              "session is summarizing (default 5.0)")
    parser.add_argument("--max-wait", type=float, default=7200.0,
                         help="max seconds to wait for any single "
                              "session to finish before giving up and "
                              "moving on (default 7200 = 2 hours)")
    args = parser.parse_args()

    session_ids = get_sessions_needing_summary(args.include_test, args.force)

    if not session_ids:
        print("Nothing to do — every session already has a summary.")
        return

    print(f"Found {len(session_ids)} session(s) to summarize:")
    for sid in session_ids:
        print(f"  - {sid}")
    print()

    ok, failed = [], []

    # Short timeout is fine now — each individual HTTP call (queue or
    # poll) is quick; the long wait happens in the polling loop, not
    # inside any single request.
    with httpx.Client(timeout=30) as client:
        for i, sid in enumerate(session_ids, 1):
            print(f"[{i}/{len(session_ids)}] summarizing '{sid}' ... ", end="", flush=True)
            try:
                success, reason = summarize_and_wait(
                    client, args.base_url, sid, args.force,
                    args.poll_interval, args.max_wait,
                )
                print(reason)
                (ok if success else failed).append((sid, reason))
            except Exception as e:
                print(f"ERROR: {e}")
                failed.append((sid, str(e)))

            if i < len(session_ids):
                time.sleep(args.delay)

    print()
    print(f"Done. {len(ok)} summarized, {len(failed)} skipped/failed.")
    if failed:
        print("Skipped/failed sessions:")
        for sid, reason in failed:
            print(f"  - {sid}: {reason}")


if __name__ == "__main__":
    main()

