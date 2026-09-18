"""
Bot Bot — Conversation Memory (Part 5)

Saves every conversation turn to a local SQLite database and
retrieves recent history so the bot remembers what was said
earlier in the same session.

Database file: memory.db (created automatically in C:\\mybot)
"""

import hashlib
import re
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "memory.db"
LINKS_FILE_PATH = Path(__file__).parent / "personal_data" / "journals" / "Important links.txt"

# Rolling-summarization tuning: fold older turns into an updating summary
# once a session accumulates ROLLUP_TRIGGER_TURNS un-folded rows, keeping
# the most recent KEEP_RECENT_RAW_TURNS raw so recent exchanges stay
# verbatim in the prompt rather than immediately paraphrased.
ROLLUP_TRIGGER_TURNS = 12
KEEP_RECENT_RAW_TURNS = 6

# Sequential size-bounded chunk storage: once a session's current rolling
# chunk's summary text reaches this many characters, the NEXT rollup starts
# a fresh chunk (chunk_index + 1) instead of folding onto it again. This
# mirrors summarizer.py's MAX_CHARS_PER_SUMMARY_BATCH=6000 pattern for the
# new-turns side, applied here to the existing-summary side, which was the
# unbounded-growth gap. Not imported from summarizer.py on purpose — kept
# as a local constant to avoid a circular import (same reason summarize_fn
# is passed into maybe_rollup_session() as DI rather than imported).
ROLLUP_CHUNK_MAX_CHARS = 6000

# ----------------------------------------------------------------
# Person + activity tagging (Step 2)
#
# session_summaries rows get tagged on save with:
#   - "person" tags: which of rag.py's known_people (extract_person_key()
#     keys) are mentioned in the summary text. Reuses the exact substring
#     match rag.py's detect_mentioned_person() uses (`person in text_lower`)
#     rather than calling that function directly, since it only returns
#     the FIRST match — tagging needs every person mentioned, not one.
#   - "activity" tags: plain keyword extraction from the summary text
#     (tokenize, drop stopwords/short words, keep the rest verbatim).
#     Deliberately exact-match only for now, no stemming/normalization
#     ("went for a run" / "running" / "jog" won't collide yet) — that's a
#     known, intentionally deferred follow-up per the decision to ship the
#     simplest version first and tune once real query patterns are
#     visible, rather than guessing at a fixed vocabulary upfront.
#
# get_recent_summaries()'s `query` param tokenizes the same way and looks
# up overlapping tag_values, so this is symmetric: whatever a summary got
# tagged with is exactly what a query can match against.
# ----------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9']+")

_MIN_TAG_WORD_LEN = 3

_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "been", "being", "this",
    "that", "these", "those", "with", "about", "into", "onto", "from",
    "they", "them", "their", "theirs", "there", "then", "than", "when",
    "what", "which", "while", "would", "could", "should", "will", "shall",
    "can", "cant", "cannot", "not", "did", "does", "doing", "done", "have",
    "has", "had", "having", "his", "her", "hers", "him", "she", "you",
    "your", "yours", "our", "ours", "its", "itself", "himself", "herself",
    "themselves", "who", "whom", "whose", "how", "why", "where", "some",
    "any", "all", "each", "few", "more", "most", "other", "such", "only",
    "own", "same", "here", "over", "under", "again", "once", "also", "just",
    "still", "even", "ever", "never", "always", "user", "bot", "session",
    "conversation", "discussed", "mentioned", "talked", "including",
    "involving", "upcoming", "described", "chat",
}


def _tokenize(text: str) -> set[str]:
    """Lowercases and splits text into word tokens, no stopword/length
    filtering yet — callers apply that themselves since save-side and
    query-side sometimes want slightly different filtering."""
    return set(_WORD_RE.findall(text.lower()))


def _filter_tag_words(words: set[str], exclude: set[str] = frozenset()) -> set[str]:
    return {
        w for w in words
        if len(w) >= _MIN_TAG_WORD_LEN and w not in _STOPWORDS and w not in exclude
    }


def _extract_person_tags(summary: str, known_people: set[str] | None) -> set[str]:
    """Same substring-match approach as rag.py's detect_mentioned_person(),
    but collects every match instead of stopping at the first."""
    if not known_people:
        return set()
    text_lower = summary.lower()
    return {p for p in known_people if p and p in text_lower}


def _extract_activity_tags(summary: str, exclude: set[str] = frozenset()) -> set[str]:
    """Exact-keyword activity tagging: tokenize the summary, drop
    stopwords/short words, exclude anything already tagged as a person so
    a name doesn't also show up as an activity keyword."""
    return _filter_tag_words(_tokenize(summary), exclude=exclude)


def _save_summary_tags(session_id: str, person_tags: set[str], activity_tags: set[str]):
    """Replaces this session's tag rows wholesale — safe to call
    repeatedly (save_session_summary() updates in place, backfill re-runs
    over everything), avoids stale tags accumulating from an old summary
    text after the session's summary changes."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM session_summary_tags WHERE session_id = ?", (session_id,))
    rows = (
        [(session_id, "person", v) for v in person_tags]
        + [(session_id, "activity", v) for v in activity_tags]
    )
    if rows:
        cursor.executemany(
            "INSERT OR IGNORE INTO session_summary_tags (session_id, tag_type, tag_value) VALUES (?, ?, ?)",
            rows,
        )
    conn.commit()
    conn.close()


def _session_ids_matching_tags(session_ids: list[str], tokens: set[str]) -> set[str]:
    """Returns the subset of session_ids that have at least one
    session_summary_tags row whose tag_value is in tokens (person or
    activity, doesn't matter which for matching purposes)."""
    if not session_ids or not tokens:
        return set()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    sid_placeholders = ",".join("?" for _ in session_ids)
    tok_placeholders = ",".join("?" for _ in tokens)
    cursor.execute(
        f"""
        SELECT DISTINCT session_id FROM session_summary_tags
        WHERE session_id IN ({sid_placeholders}) AND tag_value IN ({tok_placeholders})
        """,
        (*session_ids, *tokens),
    )
    rows = cursor.fetchall()
    conn.close()
    return {r[0] for r in rows}


def backfill_summary_tags(known_people: set[str]) -> int:
    """
    Re-tags every existing session_summaries row against the given
    known_people set. Needed for rows saved before tagging existed, or
    saved with known_people=None/empty (e.g. get_known_people() returned
    nothing because ChromaDB wasn't ready yet at save time). Safe to
    re-run — _save_summary_tags() replaces each session's tags wholesale
    rather than appending. Returns the number of rows processed.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, summary FROM session_summaries")
    rows = cursor.fetchall()
    conn.close()

    for session_id, summary in rows:
        person_tags = _extract_person_tags(summary, known_people)
        activity_tags = _extract_activity_tags(summary, exclude=person_tags)
        _save_summary_tags(session_id, person_tags, activity_tags)

    print(f"[memory] backfilled tags for {len(rows)} session_summaries row(s)")
    return len(rows)


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


def _migrate_rolling_summaries_if_needed(conn: sqlite3.Connection):
    """
    One-time migration for DBs created before chunk storage existed.

    The original schema had `rolling_summaries.session_id` as UNIQUE (one
    ever-growing row per session). If that shape is still on disk, rename
    it, create the new chunked shape, and copy each session's existing row
    in as chunk_index=0 so no history is lost. If that migrated chunk is
    already over ROLLUP_CHUNK_MAX_CHARS (likely, since it's the row that
    had the unbounded-growth bug), the next rollup call will detect that
    and roll forward into chunk_index=1 on its own — no special-casing
    needed here beyond getting the row into the new table.
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(rolling_summaries)")
    cols = {row[1] for row in cursor.fetchall()}

    if not cols or "chunk_index" in cols:
        return  # fresh DB (table doesn't exist yet) or already migrated

    cursor.execute("ALTER TABLE rolling_summaries RENAME TO rolling_summaries_old")
    cursor.execute("""
        CREATE TABLE rolling_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            summary TEXT NOT NULL,
            upto_turn_id INTEGER NOT NULL DEFAULT 0,
            last_hash TEXT,
            updated_at TEXT NOT NULL,
            UNIQUE(session_id, chunk_index)
        )
    """)
    cursor.execute("""
        INSERT INTO rolling_summaries
            (session_id, chunk_index, summary, upto_turn_id, last_hash, updated_at)
        SELECT session_id, 0, summary, upto_turn_id, last_hash, updated_at
        FROM rolling_summaries_old
    """)
    cursor.execute("DROP TABLE rolling_summaries_old")
    conn.commit()
    print("[memory] migrated rolling_summaries to chunked schema "
          "(existing rows preserved as chunk_index=0)")


# Tables/columns that hold turn/summary timestamps written by this module.
# Used by the one-time naive->UTC-aware migration below. Internal constants,
# never user input — safe to interpolate into SQL here.
_TIMESTAMP_COLUMNS = [
    ("conversations", "timestamp"),
    ("session_summaries", "created_at"),
    ("rolling_summaries", "updated_at"),
]


def _count_naive_timestamp_rows(conn) -> int:
    """How many stored timestamp values are still naive ISO strings (no
    UTC offset suffix). 0 on a fresh DB, an already-migrated DB, or when
    optional tables don't exist yet."""
    cursor = conn.cursor()
    total = 0
    for table, col in _TIMESTAMP_COLUMNS:
        try:
            cursor.execute(
                f"SELECT COUNT(*) FROM {table} "
                f"WHERE {col} IS NOT NULL "
                f"AND {col} NOT LIKE '%+00:00' AND {col} NOT LIKE '%Z'",
                (),
            )
            total += cursor.fetchone()[0]
        except sqlite3.OperationalError:
            continue  # table doesn't exist yet — nothing to migrate
    return total


def _migrate_naive_timestamps_to_utc(conn) -> int:
    """One-time rewrite of datetime.utcnow().isoformat()-era values to
    tz-aware UTC ('...+00:00'). Idempotent: aware rows are skipped, so
    re-running is a no-op. Unparseable/corrupt values are logged and left
    untouched rather than raised over (same policy as
    get_sessions_with_last_activity's skip-and-log). Returns rows rewritten.

    Ordering note: lexicographic MAX()/ORDER BY stays correct after the
    migration because ALL legacy rows convert in one pass and every new
    write is aware — no mixed-format comparisons occur at rest.
    """
    cursor = conn.cursor()
    converted = 0
    for table, col in _TIMESTAMP_COLUMNS:
        try:
            cursor.execute(f"SELECT rowid, {col} FROM {table}")
        except sqlite3.OperationalError:
            continue
        updates = []
        for rowid, val in cursor.fetchall():
            if not isinstance(val, str) or val.endswith("+00:00") or val.endswith("Z"):
                continue
            try:
                dt = datetime.fromisoformat(val)
            except ValueError:
                print(f"[memory] tz migration: skipping unparseable "
                      f"{table}.{col} rowid={rowid}: {val!r}")
                continue
            if dt.tzinfo is None:
                updates.append((dt.replace(tzinfo=timezone.utc).isoformat(), rowid))
        if updates:
            cursor.executemany(
                f"UPDATE {table} SET {col} = ? WHERE rowid = ?", updates
            )
            converted += len(updates)
    return converted


def init_db():
    """Creates the conversations, session_summaries, and rolling_summaries
    tables if they don't exist yet, migrating rolling_summaries in place
    if an older single-row-per-session copy is found."""
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

    # Secondary index over session_summaries for person/activity-aware
    # retrieval (Step 2). UNIQUE(session_id, tag_type, tag_value) makes
    # _save_summary_tags()'s INSERT OR IGNORE safe against duplicate tags
    # within one save. Not a foreign key to session_summaries on purpose —
    # rows are wholesale-replaced by session_id, not cascade-managed.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS session_summary_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            tag_type TEXT NOT NULL,
            tag_value TEXT NOT NULL,
            UNIQUE(session_id, tag_type, tag_value)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_summary_tags_value
        ON session_summary_tags(tag_value)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_summary_tags_session
        ON session_summary_tags(session_id)
    """)

    # Covering index for get_sessions_with_last_activity(): lets its
    # GROUP BY session_id / MAX(timestamp) walk the index instead of
    # full-scanning conversations every followup tick (60s). One-time
    # build cost; keeps the per-tick query cheap as turn count grows.
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversations_session_ts
        ON conversations(session_id, timestamp)
    """)

    _migrate_rolling_summaries_if_needed(conn)

    # Sequential size-bounded chunks: multiple rows per session instead of
    # one ever-growing row. Distinct from session_summaries, which is the
    # end-of-session summary. upto_turn_id is the highest conversations.id
    # already folded into ANY of this session's chunks — get_recent_history()
    # uses it to skip re-sending turns already represented in a summary.
    # last_hash lets maybe_rollup_session() no-op on a repeated/racing call
    # with the same batch instead of re-calling the LLM. chunk_index orders
    # chunks so get_current_session_summary() can reconstruct the full
    # session by joining them in sequence.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rolling_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            summary TEXT NOT NULL,
            upto_turn_id INTEGER NOT NULL DEFAULT 0,
            last_hash TEXT,
            updated_at TEXT NOT NULL,
            UNIQUE(session_id, chunk_index)
        )
    """)
    # One-time naive->UTC-aware timestamp migration (see the plan in
    # PROJECT_STATE-era discussions: datetime.utcnow() is deprecated and
    # its naive output was the root of the float-vs-str bug class). A
    # fresh file backup is taken BEFORE any rewrite, only on the run
    # where rows actually need converting, so normal startups don't
    # accumulate backups.
    if _count_naive_timestamp_rows(conn) > 0:
        conn.commit()
        conn.close()
        backup_path = DB_PATH.with_name(
            f"{DB_PATH.name}.bak-{datetime.now(timezone.utc).date().isoformat()}-pre-tz"
        )
        if not backup_path.exists():
            shutil.copyfile(DB_PATH, backup_path)
            print(f"[memory] pre-migration backup written: {backup_path.name}")
        else:
            print(f"[memory] backup {backup_path.name} already exists, not overwriting")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        converted = _migrate_naive_timestamps_to_utc(conn)
        if converted:
            print(f"[memory] tz migration: rewrote {converted} naive timestamp(s) "
                  f"to UTC-aware ISO")
    # else: nothing to migrate (fresh or already migrated) — stay silent,
    # this runs on every import.

    conn.commit()
    conn.close()


def save_turn(session_id: str, user_message: str, bot_reply: str, mode: str):
    """Saves one full turn (user message + bot reply) to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()

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
    now = datetime.now(timezone.utc).isoformat()

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
    Rows already folded into any rolling-summary chunk are excluded.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT MAX(upto_turn_id) FROM rolling_summaries WHERE session_id = ?",
        (session_id,)
    )
    row = cursor.fetchone()
    upto_turn_id = row[0] if row and row[0] is not None else 0

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


def count_turns(session_id: str) -> int:
    """Cheap row count for a session (no content read). Used by main.py's
    summary-idle loop to skip trivial sessions without fetching turns."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM conversations WHERE session_id = ?", (session_id,))
    n = cursor.fetchone()[0]
    conn.close()
    return n


def get_replay_turn(session_id: str, turn_rowid: int, history_turns: int = 12):
    """CoT-replay loader: returns (message, history) for one historical USER
    turn, with the N turns that preceded it at the time. Returns (None, None)
    if the rowid doesn't exist in that session or isn't a user row."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM conversations WHERE id = ? AND session_id = ?",
        (turn_rowid, session_id))
    row = cursor.fetchone()
    if row is None or row[0] != "user":
        conn.close()
        return None, None
    message = row[1]
    cursor.execute(
        "SELECT role, content FROM conversations "
        "WHERE session_id = ? AND id < ? ORDER BY id DESC LIMIT ?",
        (session_id, turn_rowid, history_turns * 2))
    hist = list(reversed(cursor.fetchall()))
    conn.close()
    history = [{"role": r, "content": c} for r, c in hist]
    return message, history


_summaries_cache = {"data": None}


def save_session_summary(session_id: str, summary: str, known_people: set[str] | None = None):
    """
    Saves or updates the summary for a given session, then tags it for
    person/activity retrieval (Step 2).

    known_people - rag.py's get_known_people() set, passed in by the
                    caller (summarizer.py, via main.py's /summarize call
                    site) rather than imported here, same reasoning as
                    npu_lock being DI'd elsewhere in this file: avoids a
                    circular import (memory.py -> rag.py -> ... ) and
                    keeps this module's only responsibility as storage.
                    If omitted or empty (e.g. get_known_people() failed
                    or ChromaDB had no data yet), the summary is saved
                    with activity tags only — backfill_summary_tags() can
                    re-tag it with person tags later once known_people is
                    available.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()

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

    person_tags = _extract_person_tags(summary, known_people)
    activity_tags = _extract_activity_tags(summary, exclude=person_tags)
    _save_summary_tags(session_id, person_tags, activity_tags)


def _hash_turns(existing_summary: str, batch: list[tuple]) -> str:
    """Hashes the existing chunk summary plus the batch of turns being
    folded, for idempotency. `existing_summary` is the CURRENT CHUNK's
    text only (may be "" if a new chunk is starting), not the full
    reconstructed session summary — that's what keeps this bounded."""
    hasher = hashlib.sha256()
    hasher.update(existing_summary.encode("utf-8"))
    for turn_id, role, content in batch:
        hasher.update(str(turn_id).encode("utf-8"))
        hasher.update(role.encode("utf-8"))
        hasher.update(content.encode("utf-8"))
    return hasher.hexdigest()


def get_current_session_summary(session_id: str) -> str | None:
    """
    Returns the session's live rolling summary for 'summarize this chat'
    requests, reconstructed by joining all size-bounded chunks in
    chunk_index order. Same interface as before chunking (a single string
    or None) — this path needs completeness/order, not search, so it just
    concatenates rather than going through any indexing layer.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT summary FROM rolling_summaries WHERE session_id = ? ORDER BY chunk_index ASC",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return None
    return "\n\n".join(r[0] for r in rows)


def _get_latest_chunk(session_id: str):
    """Returns (chunk_index, summary, upto_turn_id, last_hash) for a
    session's highest-index chunk, or None if the session has no chunks
    yet."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT chunk_index, summary, upto_turn_id, last_hash
        FROM rolling_summaries
        WHERE session_id = ?
        ORDER BY chunk_index DESC
        LIMIT 1
        """,
        (session_id,)
    )
    row = cursor.fetchone()
    conn.close()
    return row


def _save_rolling_summary(session_id: str, chunk_index: int, summary: str, upto_turn_id: int, turns_hash: str):
    """Inserts a new chunk row or updates the existing (session_id,
    chunk_index) row — chunk_index is only incremented by the caller
    (maybe_rollup_session) once the current chunk is at/over the size
    budget, so most calls update the same latest chunk in place."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()

    cursor.execute(
        """
        INSERT INTO rolling_summaries (session_id, chunk_index, summary, upto_turn_id, last_hash, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id, chunk_index) DO UPDATE SET
            summary = excluded.summary,
            upto_turn_id = excluded.upto_turn_id,
            last_hash = excluded.last_hash,
            updated_at = excluded.updated_at
        """,
        (session_id, chunk_index, summary, upto_turn_id, turns_hash, now)
    )
    conn.commit()
    conn.close()


# In-flight guard: at most one rollup per session at a time. Without this,
# rapid messaging spawns overlapping _background_rollup tasks that each read
# the same upto_turn_id, fold overlapping batches through separate NPU calls,
# and write duplicated content into the rolling summary (last_hash only
# dedupes identical batches, not overlapping ones — measured ~4.6% of user
# turns land within the rollup NPU-call window). Same pattern as main.py's
# _summarize_jobs, simplified to a set since rollups have no status endpoint.
_rollups_in_flight: set[str] = set()


async def maybe_rollup_session(session_id: str, summarize_fn, npu_lock) -> bool:
    """
    Public entry: enforces one in-flight rollup per session (see
    _rollups_in_flight above) and delegates to the inner implementation,
    so the fold/chunk logic below stays untouched. Returns False both for
    "not enough turns yet" and "rollup already running" — the caller
    (main._background_rollup) only logs, so the two are indistinguishable
    there, which is fine.
    """
    if session_id in _rollups_in_flight:
        return False
    _rollups_in_flight.add(session_id)
    try:
        return await _maybe_rollup_session_inner(session_id, summarize_fn, npu_lock)
    finally:
        _rollups_in_flight.discard(session_id)


async def _maybe_rollup_session_inner(session_id: str, summarize_fn, npu_lock) -> bool:
    """
    Cheap no-op unless the session has crossed ROLLUP_TRIGGER_TURNS
    un-folded rows since the last rollup.

    Folds onto the session's current (highest-index) chunk as long as that
    chunk's summary is under ROLLUP_CHUNK_MAX_CHARS. Once it's at/over that
    budget, starts a fresh chunk (chunk_index + 1, empty starting summary)
    instead — so no single NPU call ever has to re-carry an unbounded
    existing-summary; each call's existing-summary side is capped the same
    way the new-turns side already was.

    Async function that awaits `summarize_fn`, transforming the DB tuples
    into dictionary shapes `[{"role": ..., "content": ...}]` as expected
    by the summarizer batch process.
    """
    latest = _get_latest_chunk(session_id)
    if latest:
        chunk_index, chunk_summary, upto_turn_id, last_hash = latest
    else:
        chunk_index, chunk_summary, upto_turn_id, last_hash = 0, "", 0, None

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
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

    # Current chunk is full: start a new one rather than folding onto it
    # again. This is the actual overflow fix.
    if len(chunk_summary) >= ROLLUP_CHUNK_MAX_CHARS:
        chunk_index += 1
        chunk_summary = ""
        last_hash = None  # new chunk has no prior hash to compare against

    turns_hash = _hash_turns(chunk_summary, batch)

    if turns_hash == last_hash:
        return False  # idempotent skip

    # Convert the (id, role, content) tuple to the dict shape summarizer expects
    batch_dicts = [{"role": role, "content": content} for _, role, content in batch]

    # Call the async summarize function
    new_summary = await summarize_fn(chunk_summary, batch_dicts, npu_lock)
    new_upto_turn_id = batch[-1][0]  # id of the last folded row

    _save_rolling_summary(session_id, chunk_index, new_summary, new_upto_turn_id, turns_hash)
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


def get_recent_summaries(exclude_session_id: str = None, limit: int = 5, query: str = None) -> list[str]:
    """
    Returns the most recent session summaries across all past sessions.

    query - optional (Step 2). When given, sessions among the eligible
            pool whose person/activity tags overlap with query's tokens
            are moved to the front (most-recent-among-matches first),
            with any remaining slots back-filled by plain recency —
            same tokenizing/stopword rules as tagging on save, so a
            question like "did we ever talk about a trip with sooryan?"
            can surface an older session about Sooryan even if it's
            outside the most recent `limit` sessions. Matching is
            restricted to the same 200-most-recent cached pool this
            function already worked from before Step 2 — not a full
            history scan. Omitting query (main.py skips this call
            entirely for trivial messages) reproduces the exact
            pre-Step-2 pure-recency behavior.
    """
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

    eligible = [
        (sid, summary) for sid, summary in all_summaries
        if sid not in ALWAYS_INCLUDE_SESSIONS and sid != exclude_session_id and not is_test_session(sid)
    ]

    remaining_slots = max(limit - len(priority), 0)

    query_tokens = _filter_tag_words(_tokenize(query)) if query else set()

    if query_tokens:
        eligible_ids = [sid for sid, _ in eligible]
        matched_ids = _session_ids_matching_tags(eligible_ids, query_tokens)
        # Preserve eligible's existing recency order within each group —
        # matches first (most recent match first), then the rest.
        ordered = (
            [(sid, s) for sid, s in eligible if sid in matched_ids]
            + [(sid, s) for sid, s in eligible if sid not in matched_ids]
        )
    else:
        ordered = eligible

    others = [summary for _, summary in ordered][:remaining_slots]

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


def get_sessions_with_last_activity() -> list[tuple[str, float]]:
    """Returns (session_id, last_activity_epoch_seconds) for every session
    with at least one turn, ordered most recent first.

    timestamps are STORED as naive-UTC ISO strings (save_turn writes
    datetime.utcnow().isoformat()), but RETURNED as epoch floats so
    callers (followup.py's timing math) can compare directly against
    time.time(). The naive value is explicitly anchored to UTC before
    conversion — calling .timestamp() on a naive datetime would assume
    LOCAL time and skew every timestamp by the machine's UTC offset.
    Malformed/unparseable rows are skipped rather than raised, so one
    bad row can't take down the followup loop."""
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

    result = []
    for session_id, last_ts in rows:
        try:
            dt = datetime.fromisoformat(last_ts)
        except (TypeError, ValueError):
            # Corrupted timestamp data must not fail silently (that's how
            # the original float-vs-str bug hid) — but one bad row also
            # shouldn't take down the followup loop. Log and skip.
            print(f"[memory] get_sessions_with_last_activity(): skipping session "
                  f"{session_id!r} — unparseable timestamp {last_ts!r}")
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        result.append((session_id, dt.timestamp()))
    return result


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
