"""
test_followup.py — standalone script to manually test the "followup"
mode generation path (system prompt -> model call -> scrub) end to end,
without needing a live websocket connection or waiting on the 30-minute
timer.

IMPORTANT: this deliberately calls followup._ask_model_for_followup()
directly rather than followup.check_and_send_followups(). The latter
wraps its call in `async with npu_lock`, and the call_model function it
passes down (main._call_npu_once) ALSO acquires npu_lock internally.
npu_lock is an asyncio.Lock (non-reentrant), so that nested acquire
deadlocks forever with no exception, no log line — which is almost
certainly why followups appear to "not trigger" at all. This script
sidesteps that so you can verify the rest of the pipeline (mode
registration, system prompt build, model call, scrub) actually works,
separately from that locking bug.

Usage:
    python test_followup.py                # picks most-recently-active
                                             # real (non-test) session
    python test_followup.py <session_id>    # test a specific session
"""

import asyncio
import sys

from memory import get_sessions_with_last_activity, is_test_session
from followup import _ask_model_for_followup
from main import _call_npu_once, scrub_reply


def _pick_session_id() -> str | None:
    for session_id, _last_active in get_sessions_with_last_activity():
        if not is_test_session(session_id):
            return session_id
    return None


async def main():
    session_id = sys.argv[1] if len(sys.argv) > 1 else _pick_session_id()
    if not session_id:
        print("No eligible session found in the DB (and none given on the "
              "command line). Have a real chat turn saved first.")
        return

    print(f"[test_followup] using session_id={session_id!r}")

    message = await _ask_model_for_followup(session_id, _call_npu_once, scrub_reply)

    if message is None:
        print(
            "[test_followup] got None back. This means either:\n"
            "  - get_recent_history(session_id) returned no turns for this "
            "session (nothing to follow up on), or\n"
            "  - the model call itself returned an empty/falsy reply.\n"
            "Check console output above for any [followup]/npu errors."
        )
    else:
        print("[test_followup] generated followup message:\n")
        print(message)


if __name__ == "__main__":
    asyncio.run(main())
