"""
Scans personal_data/reference/therapy_reference.txt to check:
  1. How consistently "CHAPTER <N>:" style headings appear (and any
     variants), so chapter-tagging logic can be built around a pattern
     that actually holds throughout the whole file.
  2. How many paragraphs look like worksheet/blank-numbered-list pages
     (mostly short numbered lines with little real text), so we can
     gauge how much noise filtering would remove.

Read-only — doesn't touch the DB or ChromaDB. Safe to run anytime.

Usage:
    python check_reference_formatting.py
"""
import re
from pathlib import Path

FILE_PATH = Path(__file__).parent / "personal_data" / "reference" / "therapy_reference.txt"

# Broad net first: any line starting with "chapter" (case-insensitive),
# optionally followed by a number/word and punctuation. We'll print what
# we actually find rather than assume the exact format up front.
CHAPTER_LINE_RE = re.compile(r"^\s*chapter\b", re.IGNORECASE)

# A worksheet-style line: starts with a number + period, then mostly
# whitespace (a blank line for the reader to fill in).
WORKSHEET_LINE_RE = re.compile(r"^\s*\d{1,2}\.\s*$")


def main():
    if not FILE_PATH.exists():
        print(f"ERROR: {FILE_PATH} not found.")
        return

    try:
        text = FILE_PATH.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = FILE_PATH.read_text(encoding="latin1")

    lines = text.splitlines()
    print(f"Total lines: {len(lines)}")
    print(f"Total chars: {len(text)}")
    print()

    print("=== 1. Lines starting with 'chapter' (case-insensitive) ===")
    chapter_lines = [(i, line) for i, line in enumerate(lines) if CHAPTER_LINE_RE.match(line)]
    print(f"Found {len(chapter_lines)} matching line(s):")
    for i, line in chapter_lines[:60]:
        print(f"  line {i}: {line!r}")
    if len(chapter_lines) > 60:
        print(f"  ... and {len(chapter_lines) - 60} more")
    print()

    print("=== 2. Paragraph-level split check (using rag.py's \\n\\n split) ===")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    print(f"Total paragraphs (blank-line-separated): {len(paragraphs)}")
    chapter_start_paragraphs = [p for p in paragraphs if CHAPTER_LINE_RE.match(p)]
    print(f"Paragraphs that START with 'chapter': {len(chapter_start_paragraphs)}")
    for p in chapter_start_paragraphs[:20]:
        print(f"  - {p[:80]!r}")
    print()

    print("=== 3. Worksheet-style paragraph detection ===")
    def worksheet_score(p: str) -> float:
        """Fraction of lines in this paragraph that look like a blank
        numbered worksheet line."""
        plines = p.splitlines()
        if not plines:
            return 0.0
        numbered = sum(1 for l in plines if WORKSHEET_LINE_RE.match(l))
        return numbered / len(plines)

    worksheet_like = [p for p in paragraphs if worksheet_score(p) >= 0.4]
    print(f"Paragraphs that look worksheet-like (>=40% blank numbered lines): {len(worksheet_like)}")
    for p in worksheet_like[:10]:
        preview = p[:120].replace("\n", " | ")
        print(f"  - {preview!r}")
    if len(worksheet_like) > 10:
        print(f"  ... and {len(worksheet_like) - 10} more")


if __name__ == "__main__":
    main()
