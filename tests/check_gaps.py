import sqlite3
from datetime import datetime
from collections import defaultdict

conn = sqlite3.connect('memory.db')
rows = conn.execute(
    "SELECT session_id, timestamp FROM conversations WHERE role='user' ORDER BY session_id, id"
).fetchall()
conn.close()

by_session = defaultdict(list)
for session_id, ts in rows:
    try:
        by_session[session_id].append(datetime.fromisoformat(ts))
    except (TypeError, ValueError):
        continue

rapid_pairs = 0
total_pairs = 0
for session_id, timestamps in by_session.items():
    for i in range(1, len(timestamps)):
        gap = (timestamps[i] - timestamps[i - 1]).total_seconds()
        total_pairs += 1
        if 0 < gap < 130:
            rapid_pairs += 1
            print(f"[rapid] session={session_id} gap={gap:.1f}s at {timestamps[i]}")

print(f"\nTotal consecutive user-message pairs: {total_pairs}")
print(f"Pairs under 130s apart (rollup race risk window): {rapid_pairs}")
