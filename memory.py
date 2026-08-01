"""
Bot Bot — Conversation Memory (Part 5)

Saves every conversation turn to a local SQLite database and
retrieves recent history so the bot remembers what was said
earlier in the same session.

Database file: memory.db (created automatically in C:\\mybot)
"""

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "memory.db"
LINKS_FILE_PATH = Path(__file__).parent / "personal_data" / "journals" / "Important links.txt"

# Rolling-summarization tuning: fold older turns into an updating summary
# once a session accumulates ROLLUP_TRIGGER_TURNS un-folded rows, keeping
# the most recent KEEP_RECENT_RAW_TURNS raw so recent exchanges stay
# verbatim in the prompt rather than immediately paraphrased.
ROLLUP_TRIGGER_TURNS = 12
KEEP_RECENT_RAW_TURNS = 6


def get_important_links() -> str:
    """Reads and returns the contents of the Important links file directly."""
    if not LINKS_FILE_PATH.exists():
        return ""
    
    try:
        with open(LINKS_FILE_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                # Add a clear header so the bot knows what this text is
                return f"### IMPORTANT USER LINKS ###\n{content}\n"
            return ""
    except Exception as e:
        print(f"[warning] Failed to read Important links.txt: {e}")
        return ""


def init_db():
    """Creates the conversations and session_summaries tables if they don't exist yet."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            mode TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS session_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL UNIQUE,
            summary TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    # One row per session, updated in place as the rolling summary grows.
    # Distinct from session_summaries, which is the end-of-session summary.
    # upto_turn_id is the highest conversations.id already folded into
    # `summary` — get_recent_history() uses it to skip re-sending turns
    # that are already represented in the summary. last_hash lets
    # maybe_rollup_session() no-op on a repeated/racing call with the same
    # batch instead of re-calling the LLM.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rolling_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL UNIQUE,
            summary TEXT NOT NULL,
            upto_turn_id INTEGER NOT NULL DEFAULT 0,
            last_hash TEXT,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_turn(session_id: str, user_message: str, bot_reply: str, mode: str):
    """Saves one full turn (user message + bot reply) to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()

    cursor.execute(
        "INSERT INTO conversations (session_id, role, content, mode, timestamp) VALUES (?, ?, ?, ?, ?)",
        (session_id, "user", user_message, mode, now)
    )
    cursor.execute(
        "INSERT INTO conversations (session_id, role, content, mode, timestamp) VALUES (?, ?, ?, ?, ?)",
        (session_id, "assistant", bot_reply, mode, now)
    )

    conn.commit()
    conn.close()


def save_bot_initiated_message(session_id: str, bot_message: str, mode: str = "proactive"):
    """
    Saves a message the bot sent WITHOUT the user prompting it first.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()

    cursor.execute(
        "INSERT INTO conversations (session_id, role, content, mode, timestamp) VALUES (?, ?, ?, ?, ?)",
        (session_id, "assistant", bot_message, mode, now)
    )

    conn.commit()
    conn.close()


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return max(1, len(text) // 4)


def get_recent_history(session_id: str, limit: int = 40, max_tokens: int = 2200) -> list[dict]:
    """
    Returns recent turns formatted for direct use in the Ollama messages array.
    Rows already folded into the rolling summary are excluded.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT upto_turn_id FROM rolling_summaries WHERE session_id = ?",
        (session_id,)
    )
    row = cursor.fetchone()
    upto_turn_id = row[0] if row else 0

    cursor.execute(
        """
        SELECT role, content FROM conversations
        WHERE session_id = ? AND id > ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (session_id, upto_turn_id, 200)
    )
    rows = cursor.fetchall()
    conn.close()

    selected = []
    token_total = 0
    for role, content in rows:
        if len(selected) >= limit:
            break
        turn_tokens = _estimate_tokens(content)
        if selected and token_total + turn_tokens > max_tokens:
            break
        selected.append((role, content))
        token_total += turn_tokens

    selected.reverse()  # chronological order
    return [{"role": role, "content": content} for role, content in selected]


def get_all_turns(session_id: str) -> list[dict]:
    """Returns every turn for a session, in chronological order."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT role, content FROM conversations
        WHERE session_id = ?
        ORDER BY id ASC
        """,
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()

    return [{"role": role, "content": content} for role, content in rows]


def session_has_summary(session_id: str) -> bool:
    """Checks whether a session already has a saved summary."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM session_summaries WHERE session_id = ?", (session_id,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


_summaries_cache = {"data": None}


def save_session_summary(session_id: str, summary: str):
    """Saves or updates the summary for a given session."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()

    cursor.execute(
        """
        INSERT INTO session_summaries (session_id, summary, created_at)
        VALUES (?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET summary = excluded.summary, created_at = excluded.created_at
        """,
        (session_id, summary, now)
    )
    conn.commit()
    conn.close()

    _summaries_cache["data"] = None  # invalidate cache


def _hash_turns(existing_summary: str, batch: list[tuple]) -> str:
    """Hashes the existing rolling summary plus the batch of turns for idempotency."""
    hasher = hashlib.sha256()
    hasher.update(existing_summary.encode("utf-8"))
    for turn_id, role, content in batch:
        hasher.update(str(turn_id).encode("utf-8"))
        hasher.update(role.encode("utf-8"))
        hasher.update(content.encode("utf-8"))
    return hasher.hexdigest()


def get_current_session_summary(session_id: str) -> str | None:
    """Returns the session's live rolling summary directly from the DB."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT summary FROM rolling_summaries WHERE session_id = ?",
        (session_id,)
    )
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def _save_rolling_summary(session_id: str, summary: str, upto_turn_id: int, turns_hash: str):
    """Inserts or updates the single rolling_summaries row for a session."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()

    cursor.execute(
        """
        INSERT INTO rolling_summaries (session_id, summary, upto_turn_id, last_hash, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            summary = excluded.summary,
            upto_turn_id = excluded.upto_turn_id,
            last_hash = excluded.last_hash,
            updated_at = excluded.updated_at
        """,
        (session_id, summary, upto_turn_id, turns_hash, now)
    )
    conn.commit()
    conn.close()


async def maybe_rollup_session(session_id: str, summarize_fn, npu_lock) -> bool:
    """
    Cheap no-op unless the session has crossed ROLLUP_TRIGGER_TURNS
    un-folded rows since the last rollup.

    Async function that awaits `summarize_fn`, transforming the DB tuples
    into dictionary shapes `[{"role": ..., "content": ...}]` as expected
    by the summarizer batch process.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT summary, upto_turn_id, last_hash FROM rolling_summaries WHERE session_id = ?",
        (session_id,)
    )
    row = cursor.fetchone()
    existing_summary = row[0] if row else ""
    upto_turn_id = row[1] if row else 0
    last_hash = row[2] if row else None

    cursor.execute(
        """
        SELECT id, role, content FROM conversations
        WHERE session_id = ? AND id > ?
        ORDER BY id ASC
        """,
        (session_id, upto_turn_id)
    )
    candidates = cursor.fetchall()
    conn.close()

    if len(candidates) < ROLLUP_TRIGGER_TURNS:
        return False  # not enough un-folded turns yet

    fold_count = len(candidates) - KEEP_RECENT_RAW_TURNS
    if fold_count <= 0:
        return False  # threshold crossed, but nothing old enough to fold yet

    batch = candidates[:fold_count]
    turns_hash = _hash_turns(existing_summary, batch)

    if turns_hash == last_hash:
        return False  # idempotent skip

    # Convert the (id, role, content) tuple to the dict shape summarizer expects
    batch_dicts = [{"role": role, "content": content} for _, role, content in batch]

    # Call the async summarize function
    new_summary = await summarize_fn(existing_summary, batch_dicts, npu_lock)
    new_upto_turn_id = batch[-1][0]  # id of the last folded row

    _save_rolling_summary(session_id, new_summary, new_upto_turn_id, turns_hash)
    return True


ALWAYS_INCLUDE_SESSIONS = {"therapytest"}


def is_test_session(session_id: str) -> bool:
    """
    True for any session_id containing "test" (case-insensitive), except
    the explicit ALWAYS_INCLUDE_SESSIONS allowlist (e.g. "therapytest",
    which is intentionally a real, globally-visible session despite its
    name). Test sessions like "testtest" are meant to be fully isolated
    sandboxes — this is the single source of truth both get_recent_summaries()
    and main.py's cross-session lookup use to decide what counts as
    isolated, so the two stay consistent.
    """
    return session_id not in ALWAYS_INCLUDE_SESSIONS and "test" in session_id.lower()


def get_recent_summaries(exclude_session_id: str = None, limit: int = 5) -> list[str]:
    """Returns the most recent session summaries across all past sessions."""
    if _summaries_cache["data"] is None:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT session_id, summary FROM session_summaries ORDER BY id DESC LIMIT 200"
        )
        _summaries_cache["data"] = cursor.fetchall()
        conn.close()

    all_summaries = _summaries_cache["data"]

    priority = [
        summary for sid, summary in all_summaries
        if sid in ALWAYS_INCLUDE_SESSIONS and sid != exclude_session_id
    ]

    remaining_slots = max(limit - len(priority), 0)
    others = [
        summary for sid, summary in all_summaries
        if sid not in ALWAYS_INCLUDE_SESSIONS and sid != exclude_session_id and not is_test_session(sid)
    ][:remaining_slots]

    # ALWAYS_INCLUDE_SESSIONS (e.g. "therapytest") are intentionally global:
    # they're meant to be accessible from every session regardless of
    # mode/limit, so `priority` bypasses `limit` on purpose here. Only
    # `others` is capped by remaining_slots. Do NOT "fix" this to
    # `(priority + others)[:limit]` — that was tried once and reverted;
    # it broke the intended always-visible behavior for pinned sessions.
    return priority + others


def get_summary_for_session(session_id: str) -> str | None:
    """Returns the saved summary for a specific session_id."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT summary FROM session_summaries WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def get_sessions_with_last_activity() -> list[tuple[str, str]]:
    """Returns (session_id, last_timestamp) for every session with at least one turn."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT session_id, MAX(timestamp) as last_ts
        FROM conversations
        GROUP BY session_id
        ORDER BY last_ts DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_all_session_ids() -> list[str]:
    """Returns every distinct session_id that has conversation turns."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT session_id FROM conversations")
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]


# Initialize the database as soon as this module is imported
init_db()