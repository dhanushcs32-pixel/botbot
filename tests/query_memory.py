"""
Ad-hoc query runner for memory.db - avoids REPL copy-paste issues.
Edit the QUERY and PARAMS below, then run:
    python query_memory.py
"""
import sqlite3

QUERY = """
SELECT s.session_id, s.summary, s.created_at, t.tag_type, t.tag_value
FROM session_summaries s
JOIN session_summary_tags t ON s.session_id = t.session_id
WHERE t.tag_value = ?
ORDER BY s.created_at;
"""
PARAMS = ("sharadhi",)

conn = sqlite3.connect("memory.db")
cur = conn.cursor()
cur.execute(QUERY, PARAMS)
rows = cur.fetchall()
conn.close()

print(f"{len(rows)} row(s):")
for row in rows:
    print(row)
