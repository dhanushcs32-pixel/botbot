import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memory import DB_PATH

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

sid = "session-7uyhn2917"

c.execute("SELECT COUNT(*) FROM conversations WHERE session_id = ?", (sid,))
count = c.fetchone()[0]

c.execute("SELECT SUM(LENGTH(content)) FROM conversations WHERE session_id = ?", (sid,))
total_chars = c.fetchone()[0]

print(f"turns: {count}")
print(f"total chars: {total_chars}")

conn.close()
