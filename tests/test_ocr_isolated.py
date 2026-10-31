"""
Isolated OCR/captioning test — mirrors media.py's process_image() exactly,
but run standalone (no main.py, no npu_server.py) so you can see what OCR
actually extracts from a specific screenshot, whether it falls through to
BLIP captioning, and how much memory/time each step takes.

Usage:
    python test_ocr_isolated.py path\to\screenshot.png

Requires the same packages media.py needs: pillow, pytesseract, and
(only if the captioning fallback triggers) transformers + torch.
Install psutil for the memory readings (optional, script still runs without it):
    pip install psutil --break-system-packages
"""

import io
import sys
import time
from pathlib import Path

from PIL import Image
import pytesseract

# Same path media.py hardcodes — must match your actual Tesseract install.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def mem_mb() -> str:
    if not _HAS_PSUTIL:
        return "unknown (pip install psutil for this)"
    rss = psutil.Process().memory_info().rss / (1024 * 1024)
    return f"{rss:.1f} MB"


def system_mem() -> str:
    if not _HAS_PSUTIL:
        return "unknown"
    vm = psutil.virtual_memory()
    return f"{vm.available / (1024**3):.2f} GB available / {vm.total / (1024**3):.2f} GB total"


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_ocr_isolated.py path\\to\\screenshot.png")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    print("=" * 70)
    print(f"Testing: {image_path}")
    print(f"File size: {image_path.stat().st_size / 1024:.1f} KB")
    print(f"Process RSS before anything: {mem_mb()}")
    print(f"System memory before anything: {system_mem()}")
    print("=" * 70)

    # --- Step 1: open image, exactly like media.py's process_image() ---
    file_bytes = image_path.read_bytes()
    t0 = time.time()
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        print(f"[FAIL] Could not open image: {e}")
        sys.exit(1)
    print(f"\n[Step 1] Opened image in {time.time() - t0:.3f}s")
    print(f"  Dimensions: {image.size[0]}x{image.size[1]} px  "
          f"({image.size[0] * image.size[1] / 1_000_000:.1f} MP)")
    print(f"  Process RSS after open: {mem_mb()}")

    # --- Step 2: OCR, exactly like media.py's _ocr_image() ---
    t0 = time.time()
    try:
        ocr_text = pytesseract.image_to_string(image).strip()
        ocr_error = None
    except Exception as e:
        ocr_text = ""
        ocr_error = str(e)
    ocr_time = time.time() - t0

    print(f"\n[Step 2] OCR completed in {ocr_time:.3f}s")
    if ocr_error:
        print(f"  [FAIL] OCR raised: {ocr_error}")
    print(f"  Extracted length: {len(ocr_text)} chars")
    print(f"  Process RSS after OCR: {mem_mb()}")
    print(f"  --- OCR text (raw) ---")
    print(f"  {ocr_text!r}")
    print(f"  --- end OCR text ---")

    # --- Step 3: replicate media.py's exact branch decision ---
    print(f"\n[Step 3] Routing decision (media.py's threshold is len(ocr_text) >= 15):")
    if len(ocr_text) >= 15:
        print(f"  -> OCR path taken. {len(ocr_text)} chars would be injected into the prompt as:")
        print(f"     '[Image contains the following text, extracted via OCR:]\\n{ocr_text[:200]}"
              f"{'...' if len(ocr_text) > 200 else ''}'")
        print("  -> Captioning (BLIP) would NOT run for this image.")
        print("\nDone. This image does not trigger the BLIP captioning fallback.")
        return
    else:
        print(f"  -> OCR found only {len(ocr_text)} chars (< 15) -> falls through to BLIP captioning.")
        print("     This is the path suspected of causing the npu_server.py crash")
        print("     (BLIP model load competing for RAM with the resident NPU model).")

    # --- Step 4: captioning fallback, exactly like media.py's _caption_image() ---
    print(f"\n[Step 4] Loading BLIP captioning pipeline (first use only, this is the suspected spike)...")
    print(f"  Process RSS before BLIP load: {mem_mb()}")
    print(f"  System memory before BLIP load: {system_mem()}")

    t0 = time.time()
    try:
        from transformers import pipeline
        caption_pipeline = pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base"
        )
        load_time = time.time() - t0
        print(f"  BLIP model loaded in {load_time:.2f}s")
        print(f"  Process RSS after BLIP load: {mem_mb()}")
        print(f"  System memory after BLIP load: {system_mem()}")
    except Exception as e:
        print(f"  [FAIL] BLIP load raised: {e}")
        sys.exit(1)

    t0 = time.time()
    try:
        result = caption_pipeline(image)
        caption_time = time.time() - t0
        caption_text = ""
        if result and isinstance(result, list):
            caption_text = result[0].get("generated_text", "").strip()
        print(f"\n[Step 5] Captioning inference completed in {caption_time:.2f}s")
        print(f"  Process RSS after inference: {mem_mb()}")
        print(f"  Caption: {caption_text!r}")
    except Exception as e:
        print(f"  [FAIL] Captioning inference raised: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Done. Compare 'System memory before/after BLIP load' above against")
    print("what's available while npu_server.py is also running (it holds the")
    print("compiled Qwen 7B model resident) — if available memory drops close")
    print("to zero here, that's strong evidence for the RAM-contention hypothesis.")
    print("=" * 70)


if __name__ == "__main__":
    main()
