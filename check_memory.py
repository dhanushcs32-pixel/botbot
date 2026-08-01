import sqlite3

conn = sqlite3.connect("memory.db")
cur = conn.cursor()
cur.execute(
    "SELECT role, content, mode, timestamp FROM conversations "
    "WHERE session_id = 'therapytest' ORDER BY id DESC LIMIT 5"
)
for row in cur.fetchall():
    print(row)

conn.close()
