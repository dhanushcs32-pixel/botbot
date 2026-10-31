"""
One-off: force-regenerate a specific session's summary using the fixed
summarizer.py prompts. Overwrites session_summaries.summary for that
session_id and re-tags it. Run after replacing summarizer.py + restarting
the bot at least once (so npu_server.py is up).

Usage:
    python regenerate_summary.py
"""
import asyncio

from summarizer import summarize_session
from rag import get_known_people  # adjust import if get_known_people lives elsewhere

# Edit this: session_id(s) to force-regenerate
SESSION_IDS = ["testtest"]


async def main():
    known_people = get_known_people()
    for session_id in SESSION_IDS:
        print(f"Regenerating summary for session_id={session_id!r} ...")
        summary = await summarize_session(
            session_id,
            force=True,
            npu_lock=None,
            known_people=known_people,
        )
        print(f"  -> {summary}")


if __name__ == "__main__":
    asyncio.run(main())
