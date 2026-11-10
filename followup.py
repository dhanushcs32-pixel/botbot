"""
followup.py — Proactive followup message mechanism.

Picks the most-recently-active session that is CURRENTLY CONNECTED
(has a live WebSocket in main.py's active_connections registry) and,
if eligible AND due per the timing schedule below, generates a short
lightweight followup message and sends it via the provided push
callback.

Deliberately lightweight generation: system prompt + style profile +
recent history only. No RAG/personal-data search, no session-summary
retrieval — this keeps followups cheap and fast, unlike the full
generate_reply() pipeline in main.py.

Timing schedule
----------------
Each connected session is tracked with a small in-memory state
machine (_followup_state), keyed by session_id:

  - When a session's last-message timestamp changes (a new user
    message arrived, or this is the first time we've seen the
    session), the state resets to stage 0 and the session is
    classified:
      * "active"     — the message was already recent (within
                        ACTIVE_RECENCY_THRESHOLD_SECONDS) at the
                        moment we noticed it. This is the normal
                        case: the followup loop ticks frequently, so
                        a message is typically noticed within
                        seconds of being sent.
      * "not active"  — the message was already older than that
                        threshold when first noticed (e.g. state was
                        lost on a process restart, or the session
                        reconnected with stale history and no fresh
                        message).
  - Stage 0 -> 1 (first followup): fires ACTIVE_FIRST_DELAY_SECONDS
    (15 min) after the last message if classified active, or
    INACTIVE_FIRST_DELAY_SECONDS (30 min) if classified not active.
  - Stage 1 -> 2: fires 30 min after the previous followup was sent.
  - Stage 2 -> 3: fires 1h after the previous followup was sent.
  - Stage 3+ : fires every 2h after the previous followup was sent,
    indefinitely, until a new message resets the state.

Only the single most-recently-active connected session is ever
considered per tick (unchanged from prior behavior).

Failure backoff
----------------
If generation or the push callback raises, that's tracked
separately from the stage schedule above (fail_count /
last_fail_at). While a session has outstanding failures, retries
are gated by FAILURE_BACKOFF_SECONDS (an escalating ladder, capped
at its last value) measured from the last failure — this prevents
hammering the model API or a dead socket every single tick during a
persistent failure. A successful send clears the failure state. The
stage schedule itself is untouched by failures: it only advances on
confirmed success, exactly as before.
"""

import time
import datetime as _dt
from typing import Callable, Awaitable, Dict, Optional, Tuple

from memory import get_recent_history, is_test_session, get_sessions_with_last_activity
from mode_router import get_system_prompt
from rag import get_style_profile

import logging

# Persistent logging (media.py's pattern): routes to botbot_debug.log via
# main.py's botbot.debug handler — process-global logger, no import of
# main.py. Logging ONLY; scheduling/send logic untouched.
_logger = logging.getLogger("botbot.debug")


def _log(message: str) -> None:
    print(message)
    if _logger.handlers:
        _logger.info(message)


# Injected by main.py at call time so this module never has to import
# main.py directly (main.py already imports from followup.py, and a
# circular import would break startup).
PushFn = Callable[[str, str], Awaitable[None]]
CallModelFn = Callable[[list], Awaitable[str]]
ScrubFn = Callable[[str, str], str]

FOLLOWUP_MODE = "followup"

# --- Timing constants (all in seconds) ---------------------------------
ACTIVE_RECENCY_THRESHOLD_SECONDS = 5 * 60       # "a few minutes" for active/not-active classification
ACTIVE_FIRST_DELAY_SECONDS = 15 * 60            # stage 0 -> 1, active convo
INACTIVE_FIRST_DELAY_SECONDS = 30 * 60          # stage 0 -> 1, not-active convo
FOLLOWUP_BACKOFF_SECONDS = [30 * 60, 60 * 60, 2 * 60 * 60]  # stage1->2, stage2->3, stage3+->stage4+ (repeats)

# Escalating retry delay after a generation/push failure, independent of
# the stage schedule above. Capped at its last value (repeats forever
# until a success clears it or a new message resets state).
FAILURE_BACKOFF_SECONDS = [60, 2 * 60, 5 * 60, 15 * 60, 30 * 60]

# Per-session followup timing state. Not persisted across restarts —
# on restart, a session's next check will treat it as "first noticed"
# and classify active/not-active fresh based on how old its last
# message is at that point.
_followup_state: Dict[str, dict] = {}


def _prune_followup_state(active_connections: Dict[str, object], keep_session_id: Optional[str]) -> None:
    """
    Drop tracked state for sessions that are no longer connected, so
    _followup_state can't grow unbounded across the lifetime of the
    process (same class of leak as pending_messages/ws_tickets
    elsewhere in this codebase).
    """
    connected_ids = {sid for sid, ws in active_connections.items() if ws is not None}
    stale = [
        sid for sid in _followup_state
        if sid != keep_session_id and sid not in connected_ids
    ]
    for sid in stale:
        del _followup_state[sid]


def _get_latest_eligible_session(active_connections: Dict[str, object]) -> Optional[Tuple[str, float]]:
    """
    Of all sessions that currently have a live WebSocket connection,
    return (session_id, last_active_timestamp) for the one with the
    most recent activity. Returns None if no connected session is
    eligible.
    """
    connected_ids = {
        sid for sid, ws in active_connections.items() if ws is not None
    }
    if not connected_ids:
        return None

    # get_sessions_with_last_activity() returns sessions ordered by
    # last activity, most recent first.
    for session_id, last_active in get_sessions_with_last_activity():
        if session_id not in connected_ids:
            continue
        if is_test_session(session_id):
            continue
        return session_id, last_active

    return None


def _required_delay_seconds(state: dict) -> float:
    """
    Given a session's tracked followup state, return how long must
    elapse since state['last_event_at'] before the next followup is due.
    """
    stage = state["stage"]
    if stage == 0:
        return ACTIVE_FIRST_DELAY_SECONDS if state["classified_active"] else INACTIVE_FIRST_DELAY_SECONDS
    # stage 1 -> index 0 (30min), stage 2 -> index 1 (1h), stage 3+ -> index 2 (2h, repeats)
    idx = min(stage - 1, len(FOLLOWUP_BACKOFF_SECONDS) - 1)
    return FOLLOWUP_BACKOFF_SECONDS[idx]


def _required_failure_delay_seconds(fail_count: int) -> float:
    """
    Given a session's consecutive-failure count, return how long must
    elapse since state['last_fail_at'] before another attempt is made.
    Ladder is 1-indexed against fail_count (1st failure -> first entry),
    clamped to the last entry for repeated failures.
    """
    idx = min(fail_count - 1, len(FAILURE_BACKOFF_SECONDS) - 1)
    return FAILURE_BACKOFF_SECONDS[idx]


def _get_or_reset_state(session_id: str, last_active: float, now: float) -> dict:
    """
    Return this session's tracked followup state, resetting it to
    stage 0 (and re-classifying active/not-active) if this is the
    first time we've seen it, or if its last-message timestamp has
    moved forward since we last checked (i.e. a new message arrived).
    """
    state = _followup_state.get(session_id)
    if state is None or state["last_msg_at"] != last_active:
        time_since_msg = now - last_active
        classified_active = time_since_msg <= ACTIVE_RECENCY_THRESHOLD_SECONDS
        state = {
            "last_msg_at": last_active,
            "classified_active": classified_active,
            "stage": 0,
            "last_event_at": last_active,
            "fail_count": 0,
            "last_fail_at": None,
        }
        _followup_state[session_id] = state
    return state


async def _ask_model_for_followup(
    session_id: str,
    call_model: CallModelFn,
    scrub: ScrubFn,
) -> str | None:
    """
    Generate a short proactive followup message for session_id using a
    lightweight prompt: system prompt (followup mode) + style profile +
    recent history. No RAG/personal-data search, no cross-session
    summaries — keep this cheap.
    """
    history = get_recent_history(session_id)
    if not history:
        # Nothing to follow up on — no prior turns for this session.
        return None

    style_profile = get_style_profile()

    system_prompt = get_system_prompt(
        FOLLOWUP_MODE,
        personal_context="",
        style_profile=style_profile,
        session_summaries="",
        reference_context="",
        referenced_session_context="",
    )

    messages = [{"role": "system", "content": system_prompt}] + history

    raw_reply = await call_model(messages)
    if not raw_reply:
        return None

    return scrub(raw_reply, session_id)


def _is_quiet_hours(now_ts: Optional[float] = None) -> bool:
    """True between 23:00 (11 PM) and 08:00 (8 AM) in local time."""
    local_hour = _dt.datetime.fromtimestamp(now_ts or time.time()).hour
    return local_hour >= 23 or local_hour < 8


async def check_and_send_followups(
    push_fn: PushFn,
    active_connections: Dict[str, object],
    call_model: CallModelFn,
    scrub: ScrubFn,
) -> None:
    """
    Entry point called on a timer from main.py's _followup_loop().

    Only ever considers the single most-recently-active CONNECTED
    session (see _get_latest_eligible_session) — never sends to a
    session with no live WebSocket, and never sends to more than one
    session per call. Within that session, only actually sends if the
    timing schedule (module docstring) says a followup is due.

    Locking ownership: this function deliberately does NOT acquire
    npu_lock. Locking around NPU access belongs entirely to the
    injected call_model (main._call_npu_once acquires it internally).
    An earlier version wrapped the generation call in `async with
    npu_lock` as well — a nested acquire of the same non-reentrant
    asyncio.Lock inside main.NPUThrottle, which deadlocked forever on
    the first due followup while HOLDING the lock, stalling every
    /chat request behind it too. See tests/verify_followup_runtime.py,
    which exercises the real _call_npu_once so this can't regress.
    """
    result = _get_latest_eligible_session(active_connections)
    if result is None:
        _prune_followup_state(active_connections, keep_session_id=None)
        return

    session_id, last_active = result
    _prune_followup_state(active_connections, keep_session_id=session_id)

    now = time.time()

    # Enforce quiet hours: 11:00 PM to 8:00 AM local time
    if _is_quiet_hours(now):
        return

    state = _get_or_reset_state(session_id, last_active, now)

    # Cap at 1 unanswered proactive followup per session: do not spam the user
    if state["stage"] >= 1:
        return

    delay = _required_delay_seconds(state)
    if now - state["last_event_at"] < delay:
        # Not due yet.
        return

    # If prior attempts for this session have failed, don't hammer the
    # model API or a dead socket every tick — wait out the failure
    # backoff before trying again. This is independent of (and doesn't
    # advance or reset) the stage schedule above.
    if state["fail_count"] > 0:
        fail_delay = _required_failure_delay_seconds(state["fail_count"])
        if now - state["last_fail_at"] < fail_delay:
            return

    try:
        message = await _ask_model_for_followup(session_id, call_model, scrub)
    except Exception as e:
        _log(f"[followup] error generating followup for session '{session_id}': {e}")
        state["fail_count"] += 1
        state["last_fail_at"] = now
        return

    if not message:
        # Avoid tight infinite loop every 60s when generation/scrub returns empty
        state["last_event_at"] = now
        _log(f"[followup] generated empty reply for '{session_id}', advancing event window")
        return

    try:
        await push_fn(session_id, message)
    except Exception as e:
        _log(f"[followup] error pushing followup for session '{session_id}': {e}")
        state["fail_count"] += 1
        state["last_fail_at"] = now
        return

    # Only advance the schedule on a confirmed successful send — a
    # failed generation or push should retry (per the failure backoff
    # above) rather than silently skipping ahead in the stage schedule.
    state["fail_count"] = 0
    state["last_fail_at"] = None
    state["stage"] += 1
    state["last_event_at"] = now
    # Persistent record of the successful send — the job audit found
    # historical followup sends were unverifiable because nothing was
    # ever written to botbot_debug.log.
    _log(f"[followup] sent to session '{session_id}' "
         f"({len(message)} chars, stage now {state['stage']})")
