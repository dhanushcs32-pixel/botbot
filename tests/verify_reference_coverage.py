"""
verify_reference_coverage.py

Confirms the therapy_reference.txt indexing covers the ENTIRE file
(all 12 chapters, not just a subset), and quantifies exactly how much
text was intentionally dropped (worksheet blank-lines) vs. everything
else, which should all be indexed.

Run this from C:\\botbot (same place you run index_reference_only.py),
after rag.py has already been re-indexed.

What it checks:
  1. Re-runs the same chunking logic rag.py uses (imported directly from
     rag.py, not reimplemented) against the real file on disk, and
     reports a per-chapter breakdown: chapter number/title, paragraph
     count, chunk count, total characters.
  2. Compares total characters across all chunks vs. the length of the
     worksheet-stripped source text, to show how much (if any) text is
     unaccounted for beyond the intentional strip.
  3. Cross-checks against the LIVE ChromaDB reference_data collection
     (not just the in-memory chunking result) so you're verifying what
     actually got indexed, not just what the code would produce.
"""

import sys
from pathlib import Path

# Make sure we're importing the real rag.py from the current directory.
sys.path.insert(0, str(Path.cwd()))

import rag  # noqa: E402


def main():
    ref_dir = rag.REFERENCE_DATA_DIR
    txt_files = list(ref_dir.rglob("*.txt"))
    if not txt_files:
        print(f"No .txt files found under {ref_dir}")
        return

    for file_path in txt_files:
        print(f"\n{'=' * 70}")
        print(f"FILE: {file_path}")
        print(f"{'=' * 70}")

        try:
            raw_text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw_text = file_path.read_text(encoding="latin1")

        original_chars = len(raw_text)

        stripped_text = rag._strip_worksheet_lines(raw_text)
        stripped_chars = len(stripped_text)
        removed_by_strip = original_chars - stripped_chars

        chapters = rag._split_into_chapters(stripped_text)
        chunk_dicts = rag._chunk_reference_text(raw_text)

        print(f"\nOriginal file length:              {original_chars:,} chars")
        print(f"After worksheet-line stripping:    {stripped_chars:,} chars")
        print(f"Removed by worksheet strip:         {removed_by_strip:,} chars "
              f"({removed_by_strip / original_chars * 100:.2f}% of file)")

        print(f"\nChapters found: {len(chapters)}")
        print(f"{'Ch#':<5}{'Title':<45}{'Paragraphs':<12}{'Chars'}")
        print("-" * 75)

        total_chapter_chars = 0
        untitled_chars = 0
        for ch in chapters:
            ch_text = "\n\n".join(ch["paragraphs"])
            ch_chars = len(ch_text)
            total_chapter_chars += ch_chars
            label = (f"{ch['chapter_number']}" if ch["chapter_number"] is not None
                     else "—")
            title = ch["chapter_title"] or "(untitled / pre-chapter-1 text)"
            if ch["chapter_number"] is None:
                untitled_chars += ch_chars
            print(f"{label:<5}{title[:44]:<45}{len(ch['paragraphs']):<12}{ch_chars:,}")

        print("-" * 75)
        print(f"Total chapter-body characters:      {total_chapter_chars:,} chars")
        if untitled_chars:
            print(f"  (of which untitled/pre-chapter-1):  {untitled_chars:,} chars")

        # Chunk-level totals (post oversized-paragraph splitting)
        total_chunk_chars = sum(len(c["text"]) for c in chunk_dicts)
        n_chapter_tagged = sum(1 for c in chunk_dicts if c["chapter_number"] is not None)
        print(f"\nTotal chunks produced:              {len(chunk_dicts)}")
        print(f"Chunks with chapter tag:            {n_chapter_tagged}")
        print(f"Total characters across all chunks: {total_chunk_chars:,} chars")

        # Coverage check: chunk chars should be very close to stripped_chars
        # (small differences expected from strip()/whitespace normalization
        # at paragraph joins, not real content loss)
        coverage_pct = total_chunk_chars / stripped_chars * 100
        print(f"\nCoverage: chunked text is {coverage_pct:.2f}% of the "
              f"worksheet-stripped source.")
        if coverage_pct < 95:
            print("  ⚠ SIGNIFICANT GAP — investigate: some content may be "
                  "getting dropped beyond the intentional worksheet strip.")
        else:
            print("  ✓ Coverage looks complete (remaining delta is expected "
                  "whitespace/blank-line normalization, not lost content).")

        # Live collection cross-check
        print(f"\n{'-' * 70}")
        print("LIVE ChromaDB reference_data collection cross-check:")
        try:
            collection = rag.get_reference_collection()
            all_results = collection.get(
                where={"source": str(file_path.relative_to(ref_dir))},
                include=["metadatas"]
            )
            metas = all_results.get("metadatas", [])
            print(f"Chunks actually stored in ChromaDB for this file: {len(metas)}")

            chapter_counts = {}
            for m in metas:
                key = m.get("chapter_number", "untagged")
                chapter_counts[key] = chapter_counts.get(key, 0) + 1

            print("Chunks per chapter (as stored in ChromaDB):")
            for key in sorted(chapter_counts.keys(), key=lambda k: (k == "untagged", k)):
                print(f"  Chapter {key}: {chapter_counts[key]} chunks")

            distinct_chapters = sorted(
                k for k in chapter_counts if k != "untagged"
            )
            print(f"\nDistinct chapters represented in the live index: "
                  f"{len(distinct_chapters)} -> {distinct_chapters}")
        except Exception as e:
            print(f"Could not query live collection: {e}")


if __name__ == "__main__":
    main()
