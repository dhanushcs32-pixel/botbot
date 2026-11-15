"""
time_utils.py — Code-Computed Time Awareness & Relative Date Resolution

Computes all date/time facts in Python (Asia/Kolkata timezone) using an injectable
clock (`now: datetime | None = None`) for deterministic testing.
Never asks the LLM to do date or time arithmetic.
"""

import os
import re
from datetime import datetime, timezone, timedelta
from typing import Optional, Union

# Robust Asia/Kolkata timezone resolution (handles Windows environments without tzdata)
try:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo("Asia/Kolkata")
except Exception:
    import dateutil.tz
    LOCAL_TZ = dateutil.tz.gettz("Asia/Kolkata") or timezone(timedelta(hours=5, minutes=30), name="Asia/Kolkata")


def get_current_time(now: Optional[datetime] = None) -> datetime:
    """Returns now in Asia/Kolkata if provided, else current wall clock in Asia/Kolkata."""
    if now is not None:
        if now.tzinfo is None:
            # Assume UTC if naive, then convert to Asia/Kolkata
            now = now.replace(tzinfo=timezone.utc)
        return now.astimezone(LOCAL_TZ)
    return datetime.now(LOCAL_TZ)


def parse_timestamp_safe(ts: Union[datetime, str, float, int, None]) -> Optional[datetime]:
    """Safely parses any ISO string, epoch float, or datetime into an aware UTC datetime."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    if isinstance(ts, str):
        s = ts.strip()
        if not s:
            return None
        try:
            # Handle trailing Z or offsets
            s_clean = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s_clean)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None
    return None


def human_gap(sec: float) -> str:
    """
    Computes human-friendly gap string:
      < 120s: 'just now'
      < 3600s: 'X min ago'
      < 86400s: 'Xh ago'
      < 172800s (1 day): 'yesterday'
      >= 172800s: 'X days ago'
    """
    if sec < 0:
        sec = 0.0
    if sec < 120:
        return "just now"
    if sec < 3600:
        return f"{int(sec // 60)} min ago"
    if sec < 86400:
        return f"{int(sec // 3600)}h ago"
    if sec < 172800:
        return "yesterday"
    return f"{int(sec // 86400)} days ago"


def get_time_bucket(hr: int) -> str:
    """
    Categorizes local hour into time buckets:
      0..4: 'late night'
      5..11: 'morning'
      12..16: 'afternoon'
      17..21: 'evening'
      22..23: 'night'
    """
    if 0 <= hr <= 4:
        return "late night"
    if 5 <= hr <= 11:
        return "morning"
    if 12 <= hr <= 16:
        return "afternoon"
    if 17 <= hr <= 21:
        return "evening"
    return "night"


def time_note(last_user_ts: Union[datetime, str, float, None], now: Optional[datetime] = None) -> str:
    """
    Builds compact deterministic time note:
    e.g. '[now: Sun 15 Nov, 04:20PM (afternoon) | you last spoke: 40 min ago]'
    Includes behavior cues for 24h+ gaps or late night hours.
    """
    cur = get_current_time(now)
    bucket = get_time_bucket(cur.hour)
    # Format: Sun 15 Nov, 04:20PM
    formatted_now = cur.strftime("%a %d %b, %I:%M%p")

    parsed_last = parse_timestamp_safe(last_user_ts)
    if parsed_last is None:
        gap_str = "unknown"
        gap_sec = 0.0
    else:
        cur_utc = cur.astimezone(timezone.utc)
        gap_sec = (cur_utc - parsed_last).total_seconds()
        gap_str = human_gap(gap_sec)

    base = f"[now: {formatted_now} ({bucket}) | you last spoke: {gap_str}"

    # Behavioral cues
    cues = []
    if gap_sec >= 86400:
        cues.append("24h+ gap, naturally acknowledge time gap ('been a minute', 'hey stranger') when appropriate")
    if bucket == "late night":
        cues.append("late night hours, prefer softer, shorter replies")

    if cues:
        base += f" | note: {'; '.join(cues)}]"
    else:
        base += "]"

    return base


def is_new_session(last_user_ts: Union[datetime, str, float, None], now: Optional[datetime] = None, threshold_hours: float = 6.0) -> bool:
    """Detects whether elapsed time exceeds the session boundary threshold."""
    if last_user_ts is None:
        return True
    parsed_last = parse_timestamp_safe(last_user_ts)
    if parsed_last is None:
        return True
    cur = get_current_time(now).astimezone(timezone.utc)
    gap_hours = (cur - parsed_last).total_seconds() / 3600.0
    return gap_hours >= threshold_hours


# ----------------------------------------------------------------
# Relative Date Resolution (Storage & Retrieval)
# ----------------------------------------------------------------

RELATIVE_STORAGE_REGEX = re.compile(
    r"\b(today|tomorrow|yesterday|in \d+ days?|\d+ days? ago|next (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
    re.IGNORECASE
)

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def resolve_relative_dates_for_storage(text: str, ref_now: Optional[datetime] = None) -> str:
    """
    Resolves relative expressions ('tomorrow', 'yesterday', 'next Friday', 'in 3 days')
    into explicit ISO dates (e.g. 'tomorrow (2026-11-16)') anchored to ref_now.
    """
    if not text:
        return ""
    cur = get_current_time(ref_now)
    base_date = cur.date()

    def repl(m: re.Match) -> str:
        full = m.group(0)
        # Avoid double-resolving if already followed by an ISO date
        after = text[m.end():m.end()+15]
        if re.match(r"\s*\(\d{4}-\d{2}-\d{2}\)", after):
            return full

        lower = full.lower()
        target_date = None
        if "today" in lower:
            target_date = base_date
        elif "tomorrow" in lower:
            target_date = base_date + timedelta(days=1)
        elif "yesterday" in lower:
            target_date = base_date - timedelta(days=1)
        elif "in " in lower and "day" in lower:
            num_match = re.search(r"\d+", lower)
            if num_match:
                target_date = base_date + timedelta(days=int(num_match.group(0)))
        elif "day" in lower and "ago" in lower:
            num_match = re.search(r"\d+", lower)
            if num_match:
                target_date = base_date - timedelta(days=int(num_match.group(0)))
        elif "next " in lower:
            for i, wd in enumerate(WEEKDAYS):
                if wd in lower:
                    days_ahead = (i - base_date.weekday()) % 7
                    if days_ahead == 0:
                        days_ahead = 7
                    target_date = base_date + timedelta(days=days_ahead)
                    break

        if target_date:
            return f"{full} ({target_date.isoformat()})"
        return full

    return RELATIVE_STORAGE_REGEX.sub(repl, text)


RETRIEVAL_DATE_REGEX = re.compile(
    r"(?:(tomorrow|yesterday|today|in \d+ days?|\d+ days? ago|next \w+)\s*)?\(?(\d{4}-\d{2}-\d{2})\)?",
    re.IGNORECASE
)


def format_memory_dates_for_retrieval(text: str, ref_now: Optional[datetime] = None) -> str:
    """
    Converts explicit ISO dates (YYYY-MM-DD) in retrieved memories into human-relative
    labels ('today', 'tomorrow', 'yesterday', '3 days ago'), and tags past upcoming
    events with '[past event]'.
    """
    if not text:
        return ""
    cur = get_current_time(ref_now)
    ref_date = cur.date()

    def repl(m: re.Match) -> str:
        prefix_word = m.group(1) or ""
        iso_str = m.group(2)
        try:
            target_date = datetime.strptime(iso_str, "%Y-%m-%d").date()
        except ValueError:
            return m.group(0)

        delta_days = (target_date - ref_date).days
        if delta_days == 0:
            rel_label = "today"
        elif delta_days == 1:
            rel_label = "tomorrow"
        elif delta_days == -1:
            rel_label = "yesterday"
        elif delta_days > 1:
            rel_label = f"in {delta_days} days"
        else: # delta_days < -1
            days_ago = abs(delta_days)
            rel_label = f"{days_ago} days ago [past event]"

        # If prefix indicated future but date is in the past:
        if delta_days < 0 and prefix_word and any(w in prefix_word.lower() for w in ["tomorrow", "next", "in "]):
            return f"{rel_label}"
        if prefix_word:
            return f"{rel_label}"
        return f"{rel_label}"

    return RETRIEVAL_DATE_REGEX.sub(repl, text)

