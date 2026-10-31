"""
check_data_access.py

Step 1 diagnostic: confirms the personal_data\\conversations\\ folder is
reachable from wherever you run this script, and lists what's actually in it.

This does NOT test rag.py's retrieval logic or ChromaDB's index state yet --
that needs rag.py itself (extract_person_key, collection name, etc.) which
we haven't seen. This script only answers: "can Python even see the files?"

Usage (from PowerShell, inside the botbot folder or wherever this script
lives relative to personal_data\\conversations\\):

    python check_data_access.py

If your personal_data folder isn't a sibling of this script, edit
DATA_DIR below to the correct absolute path.
"""

import os
import sys
from pathlib import Path

# ---- EDIT THIS if personal_data isn't next to this script ----
DATA_DIR = Path("personal_data") / "conversations"
# ----------------------------------------------------------------


def main():
    print("=" * 60)
    print("RAG DATA ACCESS CHECK")
    print("=" * 60)

    cwd = Path.cwd()
    print(f"\nRunning from: {cwd}")

    resolved = (cwd / DATA_DIR).resolve()
    print(f"Looking for:  {resolved}")

    if not resolved.exists():
        print("\n[FAIL] Path does not exist.")
        print("       Either the folder moved during your reorg, or this")
        print("       script isn't sitting where rag.py expects to be.")
        print("       Edit DATA_DIR at the top of this script and re-run.")
        sys.exit(1)

    if not resolved.is_dir():
        print("\n[FAIL] Path exists but is not a directory.")
        sys.exit(1)

    print("[OK] Directory found.\n")

    # List all files recursively, with size and basic sanity checks
    all_files = sorted(resolved.rglob("*"))
    files_only = [f for f in all_files if f.is_file()]

    if not files_only:
        print("[WARN] Directory exists but contains no files.")
        sys.exit(0)

    print(f"Found {len(files_only)} file(s):\n")

    empty_files = []
    unreadable_files = []
    ext_counts = {}

    for f in files_only:
        rel = f.relative_to(resolved)
        try:
            size = f.stat().st_size
        except OSError as e:
            unreadable_files.append((rel, str(e)))
            print(f"  [UNREADABLE] {rel}  ({e})")
            continue

        ext = f.suffix.lower() or "(no extension)"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

        flag = ""
        if size == 0:
            empty_files.append(rel)
            flag = "  <-- EMPTY FILE"

        print(f"  {rel}  ({size} bytes){flag}")

    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"Total files:      {len(files_only)}")
    print(f"By extension:     {ext_counts}")
    print(f"Empty files:      {len(empty_files)}")
    print(f"Unreadable files: {len(unreadable_files)}")

    if empty_files:
        print("\n[WARN] Empty files found -- these will silently contribute")
        print("       nothing to retrieval even if correctly indexed:")
        for f in empty_files:
            print(f"         {f}")

    if unreadable_files:
        print("\n[WARN] Unreadable files -- check permissions / encoding:")
        for f, err in unreadable_files:
            print(f"         {f}: {err}")

    print("\nNext step: once rag.py is available, we extend this check to")
    print("run extract_person_key() against these exact filenames and")
    print("compare the result against what's actually stored in the")
    print("ChromaDB index -- that's what confirms or rules out the")
    print("person-key bug from your notes.")


if __name__ == "__main__":
    main()
