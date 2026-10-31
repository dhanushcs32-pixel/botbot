"""
Run this from your project folder (same folder as rag.py) with:
    python diagnose_retrieval.py "your exact question here"

Prints, in order:
  1. known_people currently in the collection (from real indexed metadata)
  2. what detect_mentioned_person() resolves your question to
  3. the RAW text search_personal_data() would hand to the model

This isolates two different possible bugs:
  - If (2) is None or the wrong person -> the bug is in matching your
    question's wording to the person key (retrieval-side).
  - If (2) is correct but (3) doesn't contain the real answer -> the bug
    is in what gets chunked/ranked/returned (retrieval-side, different
    mechanism).
  - If (3) DOES contain the real answer -> retrieval is fine and the
    fabrication is happening on the model side (prompt not being
    followed) -- different fix needed, not a rag.py issue.
"""
import sys
import rag

def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_retrieval.py \"your question here\"")
        sys.exit(1)

    query = sys.argv[1]

    print("=" * 60)
    print("STEP 1: known_people currently in the collection")
    print("=" * 60)
    people = rag.get_known_people()
    print(sorted(people) if people else "(none found — is the collection populated?)")

    print()
    print("=" * 60)
    print("STEP 2: detect_mentioned_person() for your query")
    print("=" * 60)
    mentioned = rag.detect_mentioned_person(query, people)
    print(f"Query: {query!r}")
    print(f"Resolved person key: {mentioned!r}")
    if mentioned is None:
        print(">>> No person matched. Search will fall through to unfiltered")
        print(">>> similarity search across ALL personal data, not just this person.")

    print()
    print("=" * 60)
    print("STEP 3: is_meta_memory_question() check")
    print("=" * 60)
    is_meta = rag.is_meta_memory_question(query)
    print(f"is_meta_memory_question: {is_meta}")
    if is_meta and not mentioned:
        print(">>> This routes to the bot-conversation-ONLY branch —")
        print(">>> personal chat files (like the one in question) are")
        print(">>> excluded entirely regardless of top_k.")

    print()
    print("=" * 60)
    print("STEP 4: raw search_personal_data() output (top_k=8, max_chars=8000)")
    print("=" * 60)
    result = rag.search_personal_data(query, top_k=8, max_result_chars=8000)
    if not result:
        print("(EMPTY — nothing was retrieved at all for this query)")
    else:
        print(result)

if __name__ == "__main__":
    main()
