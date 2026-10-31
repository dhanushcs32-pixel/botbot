"""
Indexes ONLY personal_data/reference (e.g. therapy_reference.txt) into the
separate "reference_data" ChromaDB collection, via rag.py's
index_reference_data(). This is a distinct collection from the main
personal_data collection (60k+ chunks) — running this does NOT touch or
re-index personal_data at all.

Prints the reference_data chunk count before and after indexing, then runs
a live search_reference_data() call so you can immediately eyeball whether
retrieval is returning clean, chapter-labeled excerpts (post chapter-aware
chunking + worksheet-line-stripping fix) rather than raw/noisy/unlabeled
chunks.

Usage:
    python index_reference_only.py
    python index_reference_only.py "how do I manage distress in the moment"
"""
import sys

from rag import get_reference_collection, index_reference_data, search_reference_data

DEFAULT_TEST_QUERY = "how do I manage distress in the moment"


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEST_QUERY

    try:
        count_before = get_reference_collection().count()
    except Exception as e:
        print(f"[warning] Couldn't read reference_data count before indexing: {e}")
        count_before = None

    print(f"reference_data collection count BEFORE indexing: {count_before}")
    print()

    index_reference_data()

    try:
        count_after = get_reference_collection().count()
    except Exception as e:
        print(f"[warning] Couldn't read reference_data count after indexing: {e}")
        count_after = None

    print()
    print(f"reference_data collection count AFTER indexing: {count_after}")
    if count_before is not None and count_after is not None:
        print(f"Delta: {count_after - count_before:+d} chunks")

    print()
    print("=== Live search_reference_data() test ===")
    print(f"Query: {query!r}")
    print()

    result = search_reference_data(query)
    if not result:
        print("No results returned. (Collection may be empty, or nothing matched the query.)")
    else:
        print(result)


if __name__ == "__main__":
    main()
