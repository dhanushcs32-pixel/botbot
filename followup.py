import httpx
from datetime import datetime, timedelta

from memory import get_sessions_with_last_activity, get_recent_history, get_summary_for_session, save_bot_initiated_message

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "botbot"
MAX_STALE_DAYS = 3

# Instead of just the last sent time, we track the count and timestamp
# format: { session_id: {"count": int, "last_sent": datetime} }
_followup_state: dict[str, dict] = {}


def _get_required_gap(followup_count: int) -> timedelta:
    """Returns the escalating time gap based on how many follow-ups have been sent."""
    if followup_count == 0:
        return timedelta(minutes=30)
    elif followup_count == 1:
        return timedelta(hours=1)
    elif followup_count == 2:
        return timedelta(hours=2)
    else:
        return timedelta(hours=5)


def _is_test_session(session_id: str) -> bool:
    return "test" in session_id.lower()


def _get_latest_eligible_session() -> str | None:
    """
    Finds the SINGLE most recently active session overall. 
    Returns the session_id if it's eligible for a tiered follow-up, or None.
    """
    now = datetime.utcnow()
    all_sessions = get_sessions_with_last_activity()
    
    if not all_sessions:
        return None

    # Parse and filter out test/invalid sessions
    valid_sessions = []
    for sid, ts in all_sessions:
        if _is_test_session(sid):
            continue
        try:
            valid_sessions.append((sid, datetime.fromisoformat(ts)))
        except (TypeError, ValueError):
            continue

    if not valid_sessions:
        return None

    # Sort by timestamp descending to grab the absolute latest session
    valid_sessions.sort(key=lambda x: x[1], reverse=True)
    latest_session_id, last_active = valid_sessions[0]

    # If it's completely abandoned, let it rest
    if now - last_active > timedelta(days=MAX_STALE_DAYS):
        return None

    # Get our tracking state for this specific session
    state = _followup_state.get(latest_session_id, {"count": 0, "last_sent": None})
    last_sent = state["last_sent"]
    
    # Clever trick: If the user has sent a message SINCE our last follow-up, 
    # the session is active again. Reset the follow-up count to 0!
    if last_sent and last_active > last_sent:
        state = {"count": 0, "last_sent": None}
        last_sent = None

    count = state["count"]
    required_gap = _get_required_gap(count)

    if count == 0:
        # First follow-up: Wait for 30 minutes of quiet time from the user's last message
        if (now - last_active) < required_gap:
            return None
    else:
        # Subsequent follow-ups: Wait for the required gap since the BOT'S last message
        if not last_sent or (now - last_sent) < required_gap:
            return None

    return latest_session_id

# _ask_model_for_followup remains exactly the same as your original code
# ... [Keep your existing _ask_model_for_followup function here] ...

async def check_and_send_followups(push_callback, ollama_lock):
    """
    Checks ONLY the single most recently active session and sends a follow-up if warranted.
    """
    session_id = _get_latest_eligible_session()
    
    if not session_id:
        return  # Nothing to do right now

    try:
        message = await _ask_model_for_followup(session_id, ollama_lock)
    except Exception as e:
        print(f"[followup] Error checking session '{session_id}': {e}")
        return

    if not message:
        return

    save_bot_initiated_message(session_id, message)
    
    # Update our tiered state tracker
    current_state = _followup_state.get(session_id, {"count": 0, "last_sent": None})
    _followup_state[session_id] = {
        "count": current_state["count"] + 1,
        "last_sent": datetime.utcnow()
    }
    
    await push_callback(session_id, message)
    print(f"[followup] Sent proactive check-in to '{session_id}' (Follow-up #{_followup_state[session_id]['count']}): {message[:60]}")