import sqlite3

conn = sqlite3.connect("memory.db")
c = conn.cursor()

sid = "session-7uyhn2917"

c.execute("SELECT COUNT(*) FROM conversations WHERE session_id = ?", (sid,))
count = c.fetchone()[0]

c.execute("SELECT SUM(LENGTH(content)) FROM conversations WHERE session_id = ?", (sid,))
total_chars = c.fetchone()[0]

print(f"turns: {count}")
print(f"total chars: {total_chars}")

conn.close()
