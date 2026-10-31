"""Regression test: get_sessions_with_last_activity() type contract AND
the npu_lock nested-acquire deadlock in the followup path.

Run from anywhere with the project venv:
    cd C:\\botbot && .\\venv\\Scripts\\python.exe tests\\verify_followup_runtime.py

Background (see PROJECT_STATE.txt item A): get_sessions_with_last_activity()
originally returned raw ISO TEXT timestamps while followup.py's timing math
did float arithmetic on them — TypeError every eligible tick, swallowed by
_followup_loop's blanket try/except, so followups silently never fired.

Separately, check_and_send_followups() used to wrap its generation call in
`async with npu_lock` while the injected call_model (main._call_npu_once)
acquires that SAME lock internally — a non-reentrant nested acquire that
deadlocked forever on the first due followup, holding the lock and stalling
every /chat request behind it. This test now exercises the REAL
_call_npu_once (httpx transport monkeypatched — no NPU server needed) so a
regression to either bug fails loudly:

  - every async phase runs under asyncio.wait_for(); under the old deadlock
    the phase times out and reports FAIL instead of hanging forever;
  - a post-tick direct _call_npu_once call proves npu_lock was released
    and chat traffic still gets through.

The only DB access is read-only; followup/main in-memory state is cleared
around the run. The test-session filter is neutralized in-process only,
because a fresh/personal DB may contain nothing but test sessions.
main.strip_semantic_violations is passed-through (scrub's regex paths stay
real) purely to avoid loading the embedding model mid-test.
"""
import asyncio
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

import memory
import main
from followup import (
    _get_latest_eligible_session,
    check_and_send_followups,
)

PHASE_TIMEOUT_SECONDS = 25  # generous over the ~5s NPUThrottle cooldown; a
                            # genuine deadlock blows straight past this
from followup import (
    _get_latest_eligible_session,
    check_and_send_followups,
)


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    return cond


class _FakeResponse:
    """Minimal stand-in for the httpx response surface _call_npu_once uses."""
    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": "real-npu-path followup reply"}}


class _FakeAsyncClient(httpx.AsyncClient):
    """Real AsyncClient lifecycle, fake transport — _call_npu_once runs its
    actual code path (npu_lock acquisition, timeout wrapper, JSON parse)."""
    async def post(self, *args, **kwargs):
        return _FakeResponse()


def main_sync_checks():
    ok = True
    sessions = memory.get_sessions_with_last_activity()
    ok &= check("function returns rows", len(sessions) > 0,
                f"{len(sessions)} sessions")
    if not sessions:
        return ok

    ok &= check("all timestamps are float",
                all(isinstance(ts, float) for _, ts in sessions))
    ok &= check("ordered most-recent-first",
                all(sessions[i][1] >= sessions[i + 1][1]
                    for i in range(len(sessions) - 1)))

    # Cross-check row 0 against an independent manual conversion of the
    # raw stored string, including the UTC anchoring.
    conn = sqlite3.connect(memory.DB_PATH)
    raw_sid, raw_ts = conn.execute(
        "SELECT session_id, MAX(timestamp) FROM conversations GROUP BY session_id "
        "ORDER BY 2 DESC LIMIT 1"
    ).fetchone()
    conn.close()
    expected = datetime.fromisoformat(raw_ts).replace(tzinfo=timezone.utc).timestamp()
    got_sid, got_ts = sessions[0]
    ok &= check("row 0 matches independent UTC-aware conversion",
                got_sid == raw_sid and abs(got_ts - expected) < 1e-6,
                f"{got_sid} {got_ts} vs {expected}")
    age_hours = (time.time() - got_ts) / 3600
    print(f"       newest session '{got_sid}' last active {age_hours:.2f}h ago")
    return ok


async def main_async_checks():
    """Full check_and_send_followups path through the REAL _call_npu_once."""
    ok = True
    # NOTE: we deliberately do NOT filter out test sessions here. The goal
    # is to exercise the real pipeline; nothing is actually sent anywhere.
    sessions = memory.get_sessions_with_last_activity()
    if not sessions:
        print("[SKIP] no sessions in DB")
        return True
    sid = sessions[0][0]

    import followup
    # Sessions in this DB may all be test sessions; neutralize just the
    # test-session filter so the full flow still gets exercised (nothing
    # is actually sent — the HTTP transport and push are both faked).
    followup.is_test_session = lambda sid: False

    conns = {sid: object()}  # fake live websocket registry
    result = _get_latest_eligible_session(conns)
    ok &= check("_get_latest_eligible_session returns (str, float)",
                result is not None and isinstance(result[0], str)
                and isinstance(result[1], float),
                f"{result}")

    pushed = []

    async def push_fn(s, msg):
        pushed.append((s, msg))

    # Keep scrub_reply fully real EXCEPT the semantic-embedding step, which
    # would otherwise lazy-load the MiniLM model mid-test.
    real_strip = main.strip_semantic_violations
    main.strip_semantic_violations = lambda text, session_id="": (text, [])
    # Route the REAL _call_npu_once's HTTP call through the fake transport.
    real_client = httpx.AsyncClient
    httpx.AsyncClient = _FakeAsyncClient
    try:
        followup._followup_state.clear()

        # Tick 1: creates state (former type-crash site). Depending on how
        # old the newest session's message is, this tick may legitimately
        # already be due and push (inactive first-delay is 30min) — so
        # record the baseline instead of assuming zero pushes.
        await asyncio.wait_for(
            check_and_send_followups(push_fn, conns, main._call_npu_once,
                                     main.scrub_reply),
            timeout=PHASE_TIMEOUT_SECONDS)
        st = followup._followup_state.get(sid)
        ok &= check("tick 1: completed without deadlock or TypeError", st is not None)
        baseline_pushes = len(pushed)

        # Force due, then tick 2: exercises due-check arithmetic AND the
        # generation path through the real _call_npu_once + npu_lock.
        # Under the old nested-acquire code this phase deadlocked forever.
        st["last_event_at"] -= 24 * 3600
        await asyncio.wait_for(
            check_and_send_followups(push_fn, conns, main._call_npu_once,
                                     main.scrub_reply),
            timeout=PHASE_TIMEOUT_SECONDS)
        ok &= check("tick 2: followup generated via real _call_npu_once and pushed",
                    len(pushed) == baseline_pushes + 1
                    and all(m == "real-npu-path followup reply" for _, m in pushed),
                    f"{pushed}")
        ok &= check("tick 2: stage advanced", st["stage"] >= 1)
        ok &= check("tick 2: failure state clean", st["fail_count"] == 0)

        # Tick 3 immediately: fresh backoff window -> must NOT re-send.
        await asyncio.wait_for(
            check_and_send_followups(push_fn, conns, main._call_npu_once,
                                     main.scrub_reply),
            timeout=PHASE_TIMEOUT_SECONDS)
        ok &= check("tick 3: no premature re-send (stage backoff respected)",
                    len(pushed) == baseline_pushes + 1)

        # The deadlock-proof punchline: normal chat traffic must still get
        # through on npu_lock right after the followup tick released it.
        reply = await asyncio.wait_for(
            main._call_npu_once([{"role": "user", "content": "lock liveness probe"}]),
            timeout=PHASE_TIMEOUT_SECONDS)
        ok &= check("post-tick: npu_lock free — real _call_npu_once completes",
                    reply == "real-npu-path followup reply", f"reply={reply!r}")
    finally:
        httpx.AsyncClient = real_client
        main.strip_semantic_violations = real_strip
        followup._followup_state.clear()

    return ok


def main_skip_log_checks():
    """Corrupted-timestamp branch: must log AND skip, never raise, and
    must not lose the good rows. Uses its own throwaway DB so this works
    even if the real memory.db contains no bad data."""
    import contextlib
    import io
    import tempfile

    ok = True
    orig_db_path = memory.DB_PATH
    tmp_dir = tempfile.mkdtemp(prefix="botbot_regress_")
    tmp_db = Path(tmp_dir) / "regress.db"
    conn = sqlite3.connect(tmp_db)
    conn.execute("CREATE TABLE conversations (session_id TEXT, timestamp TEXT)")
    conn.execute("INSERT INTO conversations VALUES ('good_session', '2026-08-23T10:00:00')")
    conn.execute("INSERT INTO conversations VALUES ('bad_row', 'not-a-timestamp')")
    conn.commit()
    conn.close()

    memory.DB_PATH = str(tmp_db)
    try:
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            sessions = dict(memory.get_sessions_with_last_activity())
        out = captured.getvalue()
    finally:
        memory.DB_PATH = orig_db_path

    ok &= check("corrupt row skipped without raising",
                "bad_row" not in sessions and "good_session" in sessions,
                f"{sessions}")
    ok &= check("skip was logged, not silent",
                "[memory]" in out and "unparseable timestamp" in out,
                out.strip().splitlines()[-1] if out.strip() else "<no output>")
    return ok


if __name__ == "__main__":
    ok1 = main_sync_checks()
    ok2 = asyncio.run(main_async_checks())
    ok3 = main_skip_log_checks()
    all_ok = ok1 and ok2 and ok3
    print("\nRESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    sys.exit(0 if all_ok else 1)
