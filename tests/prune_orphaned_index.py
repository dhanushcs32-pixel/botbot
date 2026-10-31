"""
prune_orphaned_index.py

Deletes chunks from the 'personal_data' ChromaDB collection whose source
file no longer exists on disk under PERSONAL_DATA_DIR (renamed/deleted/
reorganized files that index_personal_data() never pruned, since it only
upserts). Same diff logic as check_rag_index.py's ORPHANED check, but
actually deletes instead of just reporting.

Usage (from inside C:\\botbot, PowerShell, alongside rag.py):

    python prune_orphaned_index.py
        Dry run (default). Lists every orphaned source and how many
        chunks it has, and how many chunks WOULD be deleted. Deletes
        NOTHING.

    python prune_orphaned_index.py --confirm
        Actually deletes the orphaned chunks. Prints the same report
        first, then a final list of orphaned sources to delete, then
        requires typing "yes" at an interactive prompt before touching
        the collection.

    python prune_orphaned_index.py --confirm --yes
        Same as --confirm but skips the interactive prompt too (for
        scripted/non-interactive use). Use with care.

Safe by construction:
  - Read-only unless --confirm is passed.
  - Even with --confirm, requires typing "yes" unless --yes is also passed.
  - Only ever deletes chunks whose "source" metadata does not match any
    .txt file currently on disk under PERSONAL_DATA_DIR -- never touches
    chunks for files that exist.
  - Deletes are batched (same SQL-variable-limit reasoning as the
    pagination fix in rag.py's get_known_people()) so this doesn't
    reintroduce the same crash it's cleaning up after.
"""

import sys
import argparse
from collections import defaultdict

try:
    import rag
except ImportError as e:
    print(f"[FAIL] Could not import rag.py: {e}")
    print("       Make sure this script is in the SAME folder as rag.py.")
    sys.exit(1)


PAGE_SIZE = 1000
DELETE_BATCH_SIZE = 500  # keep well under ChromaDB/SQLite's variable limit


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true",
                         help="Actually delete orphaned chunks (default is dry-run/report-only).")
    parser.add_argument("--yes", action="store_true",
                         help="Skip the interactive confirmation prompt (only used with --confirm).")
    args = parser.parse_args()

    section("1. PATH CHECK")
    if not rag.PERSONAL_DATA_DIR.exists():
        print(f"[FAIL] {rag.PERSONAL_DATA_DIR} does not exist. Aborting.")
        sys.exit(1)
    print(f"[OK] PERSONAL_DATA_DIR: {rag.PERSONAL_DATA_DIR}")

    txt_files = [
        f for f in rag.PERSONAL_DATA_DIR.rglob("*.txt")
        if f != rag.MY_STYLE_PATH
    ]
    disk_sources = {
        str(f.relative_to(rag.PERSONAL_DATA_DIR)) for f in txt_files
    }
    print(f"Found {len(disk_sources)} indexable .txt file(s) on disk (excludes my_style_template.txt).")

    section("2. SCANNING LIVE INDEX")
    try:
        collection = rag.get_collection()
        count = collection.count()
    except Exception as e:
        print(f"[FAIL] Could not open ChromaDB collection: {e}")
        sys.exit(1)

    if count == 0:
        print("Collection is empty. Nothing to prune.")
        return

    print(f"Collection 'personal_data' has {count} chunk(s) total. Paginating metadata + ids...")

    # source -> list of chunk ids currently indexed under it
    ids_by_source = defaultdict(list)
    offset = 0
    while offset < count:
        try:
            page = collection.get(
                include=["metadatas"], limit=PAGE_SIZE, offset=offset
            )
        except Exception as e:
            print(f"[FAIL] Paginated get() failed at offset {offset}: {e}")
            sys.exit(1)

        page_ids = page.get("ids", [])
        page_metas = page.get("metadatas", [])
        if not page_ids:
            break

        for chunk_id, meta in zip(page_ids, page_metas):
            src = (meta or {}).get("source", "")
            ids_by_source[src].append(chunk_id)

        offset += PAGE_SIZE

    print(f"[OK] Retrieved {sum(len(v) for v in ids_by_source.values())} chunk record(s) "
          f"across {len(ids_by_source)} distinct source(s).")

    section("3. ORPHAN DIFF")
    orphaned_sources = sorted(set(ids_by_source) - disk_sources)
    orphaned_chunk_count = sum(len(ids_by_source[src]) for src in orphaned_sources)

    print(f"Indexed sources: {len(ids_by_source)}")
    print(f"On-disk sources: {len(disk_sources)}")
    print(f"Orphaned sources (indexed, no matching file on disk): {len(orphaned_sources)}")
    print(f"Orphaned chunks (would be deleted): {orphaned_chunk_count}")

    if not orphaned_sources:
        print("\nNothing to prune. Index is clean.")
        return

    print("\n--- ORPHANED SOURCES ---")
    for src in orphaned_sources:
        n = len(ids_by_source[src])
        print(f"  {src}  ({n} chunk(s))")

    section("4. ACTION")
    if not args.confirm:
        print("Dry run only -- nothing deleted. Re-run with --confirm to actually delete")
        print("the chunks listed above.")
        return

    print(f"About to permanently delete {orphaned_chunk_count} chunk(s) across "
          f"{len(orphaned_sources)} orphaned source(s).")
    print("This cannot be undone. If any of these sources were renamed rather than")
    print("truly deleted, re-index the new filename with `python rag.py` FIRST so")
    print("the content isn't lost entirely.")

    if not args.yes:
        answer = input("\nType 'yes' to proceed: ").strip().lower()
        if answer != "yes":
            print("Aborted. Nothing deleted.")
            return

    all_orphaned_ids = [
        chunk_id for src in orphaned_sources for chunk_id in ids_by_source[src]
    ]

    deleted = 0
    for i in range(0, len(all_orphaned_ids), DELETE_BATCH_SIZE):
        batch = all_orphaned_ids[i:i + DELETE_BATCH_SIZE]
        try:
            collection.delete(ids=batch)
            deleted += len(batch)
        except Exception as e:
            print(f"[FAIL] Delete failed on batch starting at index {i}: {e}")
            print(f"       {deleted} chunk(s) deleted before this failure.")
            sys.exit(1)

    print(f"\n[OK] Deleted {deleted} chunk(s) across {len(orphaned_sources)} orphaned source(s).")
    new_count = collection.count()
    print(f"Collection 'personal_data' now has {new_count} chunk(s) (was {count}).")


if __name__ == "__main__":
    main()
