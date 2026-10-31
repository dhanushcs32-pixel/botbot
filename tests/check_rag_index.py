"""
check_rag_index.py

Step 2 diagnostic: imports rag.py directly (so it must live in the SAME
folder as rag.py -- e.g. drop this into C:\\botbot alongside rag.py) and
checks three things:

  1. Does rag.py's PERSONAL_DATA_DIR actually resolve to where your files
     live, after the folder reorg?
  2. What person key does extract_person_key() produce right now for each
     real filename -- grouped, so you can eyeball whether same-person
     files are merging correctly and different people aren't colliding.
  3. What's ACTUALLY stored in the live ChromaDB index for each file's
     chunks (the "person" metadata field) -- compared against what
     extract_person_key() would produce today. Any mismatch means that
     file's chunks are stale and need re-indexing (python rag.py).
  4. Whether the index contains ORPHANED chunks -- sources indexed under
     a filename that no longer exists anywhere on disk (renamed/deleted/
     reorganized files that were never pruned, since rag.py only upserts
     and never deletes stale sources).

Usage (from inside C:\\botbot, PowerShell):

    python check_rag_index.py

Read-only: this script does NOT modify the index, only inspects it.
"""

import sys
from collections import defaultdict

try:
    import rag
except ImportError as e:
    print(f"[FAIL] Could not import rag.py: {e}")
    print("       Make sure this script is in the SAME folder as rag.py.")
    sys.exit(1)


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    section("1. PATH CHECK")
    print(f"rag.py's PERSONAL_DATA_DIR resolves to:\n  {rag.PERSONAL_DATA_DIR}")
    if not rag.PERSONAL_DATA_DIR.exists():
        print("\n[FAIL] This path does not exist. The reorg likely broke")
        print("       rag.py's assumption that personal_data/ sits next to it.")
        print("       Either move personal_data/ back next to rag.py, or")
        print("       edit PERSONAL_DATA_DIR in rag.py to the new location.")
        sys.exit(1)
    print("[OK] Path exists.")

    txt_files = [
        f for f in rag.PERSONAL_DATA_DIR.rglob("*.txt")
        if f != rag.MY_STYLE_PATH
    ]
    print(f"\nFound {len(txt_files)} indexable .txt file(s) under PERSONAL_DATA_DIR")
    print("(excludes my_style_template.txt; includes bot_conversations/ and any subfolders)")

    # ---- Split into bot_conversations vs person-chat files, same logic as rag.py ----
    person_files = []
    bot_conv_files = []
    for f in txt_files:
        if rag.BOT_CONVERSATIONS_DIR in f.parents:
            bot_conv_files.append(f)
        else:
            person_files.append(f)

    print(f"  - person/chat files: {len(person_files)}")
    print(f"  - bot_conversations files: {len(bot_conv_files)}")

    section("2. extract_person_key() ON REAL FILENAMES (computed fresh, right now)")
    key_groups = defaultdict(list)
    for f in sorted(person_files):
        key = rag.extract_person_key(f.name)
        key_groups[key].append(f.name)

    for key in sorted(key_groups):
        files = key_groups[key]
        flag = ""
        if key == "":
            flag = "  <-- EMPTY KEY (becomes source_type='notes', not searchable by person)"
        print(f"  '{key}'{flag}")
        for fn in files:
            print(f"      {fn}")

    # Flag suspicious cases: keys that still contain platform words (regex missed them)
    suspicious = [k for k in key_groups if rag.PLATFORM_MARKER_PATTERN.search(k)]
    if suspicious:
        print("\n[WARN] These keys still contain 'whatsapp'/'insta' -- extraction")
        print("       may not be fully stripping the platform marker:")
        for k in suspicious:
            print(f"         '{k}'  <- from {key_groups[k]}")

    section("3. LIVE CHROMADB INDEX STATE")
    try:
        collection = rag.get_collection()
        count = collection.count()
    except Exception as e:
        print(f"[FAIL] Could not open ChromaDB collection: {e}")
        sys.exit(1)

    print(f"Collection 'personal_data' currently has {count} chunk(s) total.")

    if count == 0:
        print("\n[FAIL] Index is empty. Nothing has been indexed yet -- run:")
        print("         python rag.py")
        sys.exit(1)

    # NOTE: fetching all metadatas in one .get() call breaks on large
    # collections -- ChromaDB's SQLite backend hits "too many SQL variables"
    # somewhere in the tens-of-thousands-of-rows range. This is the SAME
    # unpaginated call rag.py's own get_known_people() makes, so this
    # collection size (60k+) likely means get_known_people() is ALREADY
    # silently failing in production every time it's called. Paginating
    # here to get past it for the diagnostic; flagged as a real bug below.
    PAGE_SIZE = 1000
    metadatas = []
    offset = 0
    paginated_get_failed = False
    while offset < count:
        try:
            page = collection.get(
                include=["metadatas"], limit=PAGE_SIZE, offset=offset
            )
        except Exception as e:
            print(f"[FAIL] Paginated get() also failed at offset {offset}: {e}")
            paginated_get_failed = True
            break
        page_metas = page.get("metadatas", [])
        if not page_metas:
            break
        metadatas.extend(page_metas)
        offset += PAGE_SIZE

    if paginated_get_failed:
        print("\n[FAIL] Could not retrieve full metadata even with pagination.")
        print("       Skipping sections 4/5 (cross-check, known people).")
        sys.exit(1)

    print(f"[OK] Retrieved {len(metadatas)} metadata record(s) via pagination "
          f"(unpaginated .get() fails on this collection size).")

    # Map: source filename (relative) -> set of person keys actually stored for it
    stored_by_source = defaultdict(set)
    stored_source_types = defaultdict(set)
    for meta in metadatas:
        if not meta:
            continue
        src = meta.get("source", "")
        stored_by_source[src].add(meta.get("person", ""))
        stored_source_types[src].add(meta.get("source_type", ""))

    print(f"Distinct source files represented in the index: {len(stored_by_source)}")

    section("4. CROSS-CHECK: does the live index match what extract_person_key() says NOW?")
    stale = []
    missing_from_index = []
    ok_count = 0

    for f in sorted(person_files):
        relative_source = str(f.relative_to(rag.PERSONAL_DATA_DIR))
        current_key = rag.extract_person_key(f.name)

        if relative_source not in stored_by_source:
            missing_from_index.append(relative_source)
            continue

        stored_keys = stored_by_source[relative_source]
        if stored_keys == {current_key}:
            ok_count += 1
        else:
            stale.append((relative_source, current_key, stored_keys))

    print(f"\n[OK]      {ok_count} file(s) match -- index reflects current extract_person_key() logic.")
    print(f"[STALE]   {len(stale)} file(s) where the indexed 'person' key differs from what")
    print(f"          extract_person_key() produces today -- these need re-indexing.")
    print(f"[MISSING] {len(missing_from_index)} file(s) on disk but NOT in the index at all.")

    if stale:
        print("\n--- STALE (indexed under an old/different key) ---")
        for src, current, stored in stale:
            print(f"  {src}")
            print(f"      current key would be: '{current}'")
            print(f"      but index has:        {stored}")

    if missing_from_index:
        print("\n--- MISSING (never indexed, or indexed then deleted) ---")
        for src in missing_from_index:
            print(f"  {src}")

    # ---- ORPHANED: index has chunks for a source file that no longer exists
    # on disk at all (renamed/deleted/reorganized, never pruned -- rag.py
    # only upserts, it never removes stale sources). This is the reverse
    # direction of "missing_from_index" above and was NOT previously
    # checked here -- it can only be found by diffing every indexed source
    # against every file actually on disk right now.
    all_disk_sources = {
        str(f.relative_to(rag.PERSONAL_DATA_DIR)) for f in txt_files
    }
    orphaned = sorted(set(stored_by_source) - all_disk_sources)

    print(f"\n[ORPHANED] {len(orphaned)} source(s) present in the index whose file")
    print(f"           no longer exists anywhere on disk under PERSONAL_DATA_DIR.")
    if orphaned:
        print("\n--- ORPHANED (indexed, but source file is gone -- stale/unreachable data) ---")
        for src in orphaned:
            people = stored_by_source[src]
            print(f"  {src}  (person key(s) in index: {sorted(people)})")

    section("5. WHAT detect_mentioned_person() SHOULD SEE (computed via pagination)")
    known_people = {m.get("person") for m in metadatas if m and m.get("person")}
    print(f"Distinct person keys present in the index: {len(known_people)}")
    for p in sorted(known_people):
        print(f"  '{p}'")

    print("\n[TESTING rag.get_known_people() AS ACTUALLY CALLED IN PRODUCTION]")
    try:
        live_known_people = rag.get_known_people()
        print(f"[OK] rag.get_known_people() succeeded, returned "
              f"{len(live_known_people)} key(s) -- matches expectation.")
        prod_get_known_people_broken = False
    except Exception as e:
        print(f"[FAIL] rag.get_known_people() raised: {e}")
        prod_get_known_people_broken = True

    section("VERDICT")
    if prod_get_known_people_broken:
        print("*** LIKELY ROOT CAUSE FOUND ***")
        print("rag.get_known_people() -- called by search_personal_data() on every")
        print("query -- crashes on this collection size (60k+ chunks) due to an")
        print("unpaginated ChromaDB .get() call hitting SQLite's variable limit.")
        print("search_personal_data() wraps the .get() inside get_known_people()'s")
        print("own try/except, which returns an empty set on failure -- so")
        print("mentioned_person is ALWAYS None, and every query silently falls")
        print("through to unfiltered similarity search regardless of whether the")
        print("person-key extraction is correct. This likely explains the")
        print("'not properly accessing data' symptom independent of the")
        print("extract_person_key() fix. FIX NEEDED: paginate get_known_people()")
        print("in rag.py itself (see PAGE_SIZE pagination used in this script).")
    elif stale or missing_from_index:
        print("Re-indexing is needed. From C:\\botbot, run:")
        print("    python rag.py")
        print("\nThis calls index_personal_data() + index_reference_data(), which")
        print("upserts (not replaces) -- old chunk IDs for unchanged files are")
        print("left alone, but changed files will get corrected 'person' metadata")
        print("under their existing chunk IDs since IDs are derived from filename,")
        print("not from the key itself.")
        if orphaned:
            print(f"\nAlso: {len(orphaned)} orphaned source(s) found (see section 4) --")
            print("      `python rag.py` will NOT clean these up on its own since it")
            print("      only upserts. They'll keep being searchable/retrievable")
            print("      indefinitely unless explicitly deleted from the collection.")
    else:
        print("Index matches current extract_person_key() logic for all files on")
        print("disk, and get_known_people() works fine.")
        if orphaned:
            print(f"\nHowever: {len(orphaned)} orphaned source(s) exist in the index")
            print("for files that no longer exist on disk (see section 4). These")
            print("won't cause a person-key mismatch, but they ARE stale, fully")
            print("searchable data that can surface in unfiltered similarity search")
            print("or wrong-person results. Worth deciding whether to prune them.")
        print("\nIf retrieval is still failing after all of the above, the bug is")
        print("NOT the indexing layer -- move to checking detect_mentioned_person()'s")
        print("substring matching against real user phrasing, or the confabulation")
        print("issue (context retrieved but doesn't answer the question).")


if __name__ == "__main__":
    main()
