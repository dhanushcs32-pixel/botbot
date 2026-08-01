"""
Bot Bot — PDF to TXT Converter (for reference/therapy material)

Extracts text from a PDF and saves it as a .txt file in
personal_data/reference/, ready to be indexed by rag.py.

Usage:
  1. Put your PDF somewhere, e.g. C:\\mybot\\personal_data\\raw_pdfs\\therapy.pdf
  2. Edit PDF_PATH below
  3. Run: python pdf_to_reference.py
"""

from pathlib import Path
from pypdf import PdfReader

# ----------------------------------------------------------------
# CONFIGURE THIS
# ----------------------------------------------------------------
PDF_PATH = Path(r"C:\mybot\personal_data\raw_pdfs\therapy.pdf")
OUTPUT_PATH = Path(r"C:\mybot\personal_data\reference\therapy_reference.txt")

# Most books have several pages of endorsements, title page, copyright,
# dedication, table of contents before real content starts. Adjust this
# after a first run once you see where the actual content begins.
SKIP_FIRST_PAGES = 15


def fix_mojibake(text: str) -> str:
    """Fixes common UTF-8-read-as-Latin-1 garbling (smart quotes, em dashes, etc)."""
    try:
        return text.encode("latin1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def main():
    if not PDF_PATH.exists():
        print(f"ERROR: {PDF_PATH} not found. Update PDF_PATH in this script.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(PDF_PATH))
    total_pages = len(reader.pages)
    print(f"Found {total_pages} pages. Skipping first {SKIP_FIRST_PAGES} (front matter). Extracting...")

    all_text = []
    for i, page in enumerate(reader.pages):
        if i < SKIP_FIRST_PAGES:
            continue

        text = page.extract_text()
        if text and text.strip():
            all_text.append(fix_mojibake(text.strip()))
        if (i + 1) % 20 == 0:
            print(f"  ...processed {i + 1}/{total_pages} pages")

    full_text = "\n\n".join(all_text)
    OUTPUT_PATH.write_text(full_text, encoding="utf-8")

    print(f"\nDone. Extracted {len(full_text)} characters to:")
    print(f"  {OUTPUT_PATH}")
    print("\nNext: run 'python rag.py' to index this into the reference collection.")


if __name__ == "__main__":
    main()
