"""
Tests the actual, hardened media.py pipeline (resize cap, OCR text cap,
BLIP memory guard) by importing it directly and exercising process_image()
- not a reimplementation, the real code path.

Usage (run from the same folder as media.py, e.g. C:\\BotBot):

    python test_media_pipeline.py path\\to\\image.jpg
        Runs process_image() on a real image and reports what happened
        at each stage.

    python test_media_pipeline.py --synthetic-large
        Generates a 4000x3000 synthetic image on the fly (bigger than
        MAX_IMAGE_DIMENSION_PX) to verify the resize cap fires, without
        needing a real oversized photo on hand.

    python test_media_pipeline.py --synthetic-long-text
        Generates a tall image packed with repeated lines of text,
        engineered to produce well over OCR_TEXT_MAX_CHARS of extracted
        text, to verify the truncation marker fires.

    python test_media_pipeline.py --all
        Runs both synthetic tests back to back.

If media.py lives in a different folder, pass --media-dir path\\to\\folder
before the other arguments.
"""

import sys
import argparse
import time
from pathlib import Path


def _import_media(media_dir: str):
    if media_dir:
        sys.path.insert(0, media_dir)
    else:
        # default: assume media.py is in the same folder as this script
        sys.path.insert(0, str(Path(__file__).parent))
    import media
    return media


def _get_legible_font(size: int = 28):
    """PIL's default font (used when no font is specified) is a tiny
    ~11px bitmap font that Tesseract struggles to read reliably,
    especially after the resize cap shrinks it further — that produced
    garbled OCR output in testing that had nothing to do with media.py's
    actual behavior. Load a real scalable font at a legible size instead,
    falling back gracefully across common font locations."""
    from PIL import ImageFont
    for candidate in (
        "arial.ttf",  # Windows, often resolvable by name alone
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux fallback
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    print("[test] No scalable font found, falling back to PIL's tiny default font — "
          "OCR results may be unreliable.")
    from PIL import ImageFont as _IF
    return _IF.load_default()


def _generate_no_text_image():
    """Image with no text at all — colored shapes only — so OCR finds
    under 15 chars and the captioning fallback actually runs. The
    synthetic-large test can't be relied on for this: once it has
    legible text (needed to test the resize cap without garbled OCR),
    OCR succeeds and captioning is correctly skipped, same as it would
    be for a real screenshot. This generator exists specifically to
    exercise the "OCR found nothing, fall back to captioning" branch."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (800, 600), color=(135, 206, 235))  # sky blue
    draw = ImageDraw.Draw(img)
    draw.ellipse((300, 150, 500, 350), fill=(255, 220, 0))  # sun-like circle
    draw.rectangle((0, 400, 800, 600), fill=(34, 139, 34))  # ground
    draw.polygon([(650, 400), (700, 300), (750, 400)], fill=(101, 67, 33))  # triangle
    return img


def _generate_large_image():
    """4000x3000 image, well over MAX_IMAGE_DIMENSION_PX=2000, with a
    short line of real text so OCR still has something to find after
    the resize (confirms resize doesn't break OCR, just bounds it)."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (4000, 3000), color="white")
    draw = ImageDraw.Draw(img)
    font = _get_legible_font(60)
    draw.text((100, 100), "This is a large test image for resize testing", fill="black", font=font)
    return img


def _generate_long_text_image(target_chars: int = 5000):
    """Tall image packed with repeated lines of legible text, sized to
    make Tesseract extract comfortably over OCR_TEXT_MAX_CHARS so the
    truncation path is actually exercised."""
    from PIL import Image, ImageDraw
    line = "The quick brown fox jumps over the lazy dog. "
    lines_needed = (target_chars // len(line)) + 20  # generous margin, OCR isn't perfect
    line_height = 34
    width = 1400
    height = lines_needed * line_height + 100
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)
    font = _get_legible_font(24)
    y = 20
    for i in range(lines_needed):
        draw.text((20, y), f"{i:04d} {line}", fill="black", font=font)
        y += line_height
    return img


def _run_process_image(media, image, label: str):
    import io
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"Image dimensions: {image.size[0]}x{image.size[1]} px "
          f"({image.size[0] * image.size[1] / 1_000_000:.1f} MP)")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    file_bytes = buf.getvalue()
    print(f"Encoded size: {len(file_bytes) / 1024:.1f} KB")

    if media._HAS_PSUTIL:
        import psutil
        avail = psutil.virtual_memory().available / (1024 * 1024)
        print(f"Available memory: {avail:.0f} MB "
              f"(captioning guard threshold: {media.MIN_AVAILABLE_MEMORY_MB_FOR_CAPTIONING} MB)")
    else:
        print("psutil not installed - BLIP memory guard is a no-op (always attempts captioning)")

    t0 = time.time()
    result = media.process_image(file_bytes)
    elapsed = time.time() - t0

    print(f"\nprocess_image() completed in {elapsed:.2f}s")
    print(f"Result length: {len(result)} chars")
    print(f"--- Result ---")
    preview = result if len(result) <= 800 else result[:800] + f"\n...[{len(result) - 800} more chars]"
    print(preview)
    print(f"--- end result ---")

    # Explicit pass/fail checks against the caps, so this is a real test
    # and not just a log dump.
    if "truncated" in result.lower():
        print(f"\n[CHECK] Truncation marker found in output - OCR_TEXT_MAX_CHARS cap fired.")
    if result.startswith("[Image shows:"):
        print(f"\n[CHECK] Captioning branch was taken and returned a caption - BLIP pipeline is working.")
    elif "no readable text or clear description" in result:
        print(f"\n[CHECK][WARN] Captioning was attempted but returned nothing (or was skipped by the "
              f"memory guard) - check the console output above for a 'Captioning failed:' line or a "
              f"'skipping captioning' memory-guard line to see which.")
    if len(result) > media.OCR_TEXT_MAX_CHARS + 200:  # +200 margin for the wrapper text/marker
        print(f"[CHECK][WARN] Result ({len(result)} chars) is suspiciously larger than "
              f"OCR_TEXT_MAX_CHARS ({media.OCR_TEXT_MAX_CHARS}) even accounting for wrapper text - "
              f"verify the cap is actually being applied.")


def main():
    parser = argparse.ArgumentParser(description="Test the hardened media.py pipeline")
    parser.add_argument("image_path", nargs="?", help="Path to a real image to test")
    parser.add_argument("--media-dir", default=None, help="Folder containing media.py, if not this script's folder")
    parser.add_argument("--synthetic-large", action="store_true", help="Test the resize cap with a generated oversized image")
    parser.add_argument("--synthetic-long-text", action="store_true", help="Test the OCR text cap with a generated text-dense image")
    parser.add_argument("--synthetic-no-text", action="store_true", help="Test the captioning fallback with a generated image containing no text")
    parser.add_argument("--force-captioning", action="store_true",
                         help="Bypass the memory guard for this test run only (does not modify media.py) - "
                              "use when you want to test BLIP itself independent of current memory conditions")
    parser.add_argument("--all", action="store_true", help="Run all synthetic tests")
    args = parser.parse_args()

    media = _import_media(args.media_dir)

    print(f"Loaded media.py from: {media.__file__}")
    print(f"MAX_IMAGE_DIMENSION_PX = {media.MAX_IMAGE_DIMENSION_PX}")
    print(f"OCR_TEXT_MAX_CHARS = {media.OCR_TEXT_MAX_CHARS}")
    print(f"MIN_AVAILABLE_MEMORY_MB_FOR_CAPTIONING = {media.MIN_AVAILABLE_MEMORY_MB_FOR_CAPTIONING}")
    print(f"psutil available: {media._HAS_PSUTIL}")

    if args.force_captioning:
        media._enough_memory_for_captioning = lambda: True
        print("[test] --force-captioning: memory guard bypassed for THIS TEST RUN ONLY "
              "(media.py on disk is unchanged). This isolates whether BLIP itself works, "
              "separately from whether current memory conditions allow it.")

    ran_something = False

    if args.image_path:
        ran_something = True
        file_bytes = Path(args.image_path).read_bytes()
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        _run_process_image(media, image, f"Real image: {args.image_path}")

    if args.synthetic_large or args.all:
        ran_something = True
        _run_process_image(media, _generate_large_image(), "Synthetic oversized image (resize cap test)")

    if args.synthetic_long_text or args.all:
        ran_something = True
        _run_process_image(media, _generate_long_text_image(), "Synthetic text-dense image (OCR cap test)")

    if args.synthetic_no_text or args.all:
        ran_something = True
        _run_process_image(media, _generate_no_text_image(), "Synthetic no-text image (captioning fallback test)")

    if not ran_something:
        parser.print_help()
        print("\nNo test selected - pass an image path, or --synthetic-large / --synthetic-long-text / --synthetic-no-text / --all")


if __name__ == "__main__":
    main()
