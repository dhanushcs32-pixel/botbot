"""
Bot Bot — Media Processor

Converts uploaded images and documents into plain text BEFORE they
ever reach the LLM. This lets Bot Bot "read" photos and files without
needing a vision-capable model — everything downstream (RAG, Ollama)
still only ever sees text, exactly like a normal typed message.

Two strategies for images, combined:
  - OCR (pytesseract) — pulls out any printed/handwritten text
  - Captioning (BLIP, local model) — describes what the image actually
    shows (objects, scene, people), used only when OCR finds little
    or no text, since captioning is the slower of the two

Documents (PDF, DOCX, TXT) are read directly for their text content.

Everything runs 100% locally — no cloud calls, no API keys.
"""

import os
import io
import logging
import base64
import shutil
from pathlib import Path

from PIL import Image
import pytesseract

# Windows doesn't add Tesseract to PATH by default — point pytesseract
# at the configured/installed binary so it can find it portably.
_default_tess = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
_tess_cmd = os.getenv("TESSERACT_CMD")
if _tess_cmd and Path(_tess_cmd).exists():
    pytesseract.pytesseract.tesseract_cmd = _tess_cmd
elif Path(_default_tess).exists():
    pytesseract.pytesseract.tesseract_cmd = _default_tess
elif shutil.which("tesseract"):
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")

# Persistent failure logging: same destination as main.py's debug_log
# (botbot.debug logger -> botbot_debug.log). media.py can't import main
# (circular), but logging loggers are process-global — main.py configures
# the "botbot.debug" handler at import, and media is only ever imported
# by main, so the handler exists by the time anything here logs. The
# print() keeps standalone/test usage visible.
_logger = logging.getLogger("botbot.debug")


def _log(message: str) -> None:
    """Console + persistent debug log under main.py; console-only otherwise."""
    print(message)
    if _logger.handlers:
        _logger.info(message)

# psutil is optional — only used for the BLIP memory guard below. If it's
# not installed, the guard degrades to "always attempt captioning" (the
# pre-existing behavior), same non-fatal-degrade pattern used elsewhere
# in this codebase (e.g. npu_server.py's adapter loading).
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# --- Resource-bounding constants (see PROJECT_STATE.txt: npu_server.py
# crash incident) ---
#
# Largest dimension (px) an uploaded image is allowed to be before OCR/
# captioning. Uploads are downscaled (never upscaled) to fit, preserving
# aspect ratio. Bounds both Tesseract's and BLIP's memory/CPU cost against
# unbounded upload size — a full-resolution photo costs meaningfully more
# than a screenshot needs to for either OCR or captioning quality.
MAX_IMAGE_DIMENSION_PX = 2000

# Hard cap on how much OCR-extracted text gets injected into the prompt.
# main.py's char-based prompt_trim is the last line of defense against an
# oversized prompt, but that's defense-in-depth, not a guarantee at the
# source — a single OCR result this large would already be an unusually
# dense/long document photo, and truncating here means the trim logic
# downstream never has to deal with a single field this disproportionate.
OCR_TEXT_MAX_CHARS = 4000

# Hard cap on how much document text (PDF, DOCX, TXT) gets extracted.
# Bounds context length to avoid blowing through context budget (~6144 tokens).
MAX_EXTRACTED_DOC_CHARS = 8000


def _truncate_doc_text(text: str) -> str:
    """Caps extracted document text at MAX_EXTRACTED_DOC_CHARS with a clear notice."""
    if len(text) <= MAX_EXTRACTED_DOC_CHARS:
        return text
    return text[:MAX_EXTRACTED_DOC_CHARS].rstrip() + "\n\n[Document truncated to fit context window]"

# Minimum available system RAM (MB) required before attempting to load/run
# the BLIP captioning model. This machine has been observed to run with as
# little as ~2.35 GB available while npu_server.py is resident (see
# PROJECT_STATE.txt) — BLIP's model load is a genuine, uncoordinated
# memory spike that happens in THIS process while a separate process
# (npu_server.py) already holds a large model resident. Skipping
# captioning under memory pressure trades "no caption for this image" for
# "don't risk destabilizing the whole machine" — the same kind of
# kill-switch reasoning applied to npu_server.py's GPU fallback after that
# incident. If psutil isn't installed, this check is skipped entirely
# (captioning always attempted, matching pre-existing behavior).
MIN_AVAILABLE_MEMORY_MB_FOR_CAPTIONING = 2000

_caption_processor = None  # lazy-loaded — only load the model if an image actually needs captioning
_caption_model = None


def _resize_if_oversized(image: Image.Image) -> Image.Image:
    """Downscales an image so its longest side is at most
    MAX_IMAGE_DIMENSION_PX, preserving aspect ratio. No-op if the image
    is already within bounds — never upscales."""
    if max(image.size) <= MAX_IMAGE_DIMENSION_PX:
        return image
    original_size = image.size
    image = image.copy()
    image.thumbnail((MAX_IMAGE_DIMENSION_PX, MAX_IMAGE_DIMENSION_PX), Image.LANCZOS)
    print(f"[media] resized oversized image from {original_size} to {image.size} "
          f"(cap: {MAX_IMAGE_DIMENSION_PX}px longest side)")
    return image


def _truncate_ocr_text(ocr_text: str) -> str:
    """Caps OCR-extracted text at OCR_TEXT_MAX_CHARS, appending a marker
    if truncated so it's visible in the prompt (and in logs) that this
    happened rather than silently dropping content."""
    if len(ocr_text) <= OCR_TEXT_MAX_CHARS:
        return ocr_text
    omitted = len(ocr_text) - OCR_TEXT_MAX_CHARS
    print(f"[media] OCR text truncated: {len(ocr_text)} chars -> {OCR_TEXT_MAX_CHARS} chars ({omitted} omitted)")
    return ocr_text[:OCR_TEXT_MAX_CHARS] + f"\n[...truncated, {omitted} characters omitted]"


def _enough_memory_for_captioning() -> bool:
    """Returns True if it's safe to attempt loading/running BLIP right
    now. Always True if psutil isn't installed (can't check, so preserve
    pre-existing behavior rather than blocking captioning outright)."""
    if not _HAS_PSUTIL:
        return True
    available_mb = psutil.virtual_memory().available / (1024 * 1024)
    if available_mb < MIN_AVAILABLE_MEMORY_MB_FOR_CAPTIONING:
        print(f"[media] skipping captioning: only {available_mb:.0f} MB available, "
              f"need >= {MIN_AVAILABLE_MEMORY_MB_FOR_CAPTIONING} MB "
              f"(guards against contention with npu_server.py — see PROJECT_STATE.txt)")
        return False
    return True


def _get_caption_pipeline():
    """
    Loads BLIP's processor and model directly on first use only, rather
    than going through transformers' generic pipeline("image-to-text", ...)
    task-registry lookup. That task name was silently failing on the
    installed transformers version ("Unknown task image-to-text" — the
    registry has renamed/restructured its task names), which meant every
    captioning attempt was failing and quietly falling through to the
    "no clear description" message with no crash and no visible error.
    Loading BlipProcessor/BlipForConditionalGeneration directly sidesteps
    that task-name lookup entirely — this is also what BLIP's own model
    card recommends over the generic pipeline() wrapper.

    Same lazy-load-once, non-fatal-on-failure shape as before: avoids a
    slow multi-second model load on every server startup when most
    messages won't include a photo at all, and any load failure degrades
    to "captioning unavailable" rather than crashing the caller.
    """
    global _caption_processor, _caption_model
    if _caption_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        _caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return _caption_processor, _caption_model


# --- OCR quality gating (tuned against real samples, see PROJECT_STATE) ---
# Mean word confidence (Tesseract's 0-100 scale) at which extracted text is
# presented to the model as trustworthy. Below this but above the floor,
# text is still shown but explicitly labeled low-confidence; below the
# floor, it's treated as no text at all. Tuned on real screenshots: good
# text measured 95, marginal-but-real sign text measured ~36, noise
# measured 0.
OCR_CONFIDENCE_THRESHOLD = 60.0
OCR_LOW_CONFIDENCE_FLOOR = 30.0
# If the image's smaller dimension is below this many pixels, also try an
# upscaled+grayscale variant and keep whichever has higher confidence.
# Real-sample evidence: a 567x892 image yielded ZERO words as-is but real
# (if imperfect) text when 2x-upscaled; conversely a 412x137 logo read
# perfectly as-is while upscaling DESTROYED it — hence confidence-driven
# selection rather than always-upscale.
OCR_UPSCALE_TRIGGER_PX = 800


def _ocr_with_confidence(image: Image.Image) -> tuple[str, float]:
    """Runs Tesseract once, returning (text, mean_word_confidence).
    Text preserves line breaks; empty string + 0.0 when nothing is found
    or OCR fails (failure logged via _log, not silent)."""
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception as e:
        _log(f"[media] OCR failed: {e}")
        return "", 0.0

    lines: dict = {}
    confs = []
    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue
        conf = float(data["conf"][i])
        if conf < 0:  # -1 marks non-word placeholder rows
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        confs.append(conf)

    if not confs:
        return "", 0.0
    text = "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))
    return text.strip(), sum(confs) / len(confs)


def _best_ocr(image: Image.Image) -> tuple[str, float]:
    """OCRs the image as-is, plus an upscaled+grayscale variant for small
    images, keeping whichever variant has higher mean word confidence."""
    best_text, best_conf = _ocr_with_confidence(image)

    min_dim = min(image.size)
    if min_dim < OCR_UPSCALE_TRIGGER_PX:
        factor = min(4, max(2, round(1600 / max(min_dim, 1))))
        try:
            upscaled = image.resize(
                (image.width * factor, image.height * factor), Image.LANCZOS
            ).convert("L")
            up_text, up_conf = _ocr_with_confidence(upscaled)
            if up_text and up_conf > best_conf:
                _log(f"[media] upscale x{factor} won: conf {best_conf:.0f} -> {up_conf:.0f}")
                best_text, best_conf = up_text, up_conf
        except Exception as e:
            _log(f"[media] upscaled OCR variant failed (using as-is result): {e}")

    return best_text, best_conf


def _caption_image(image: Image.Image) -> str:
    """Generates a natural-language description of what the image shows."""
    try:
        processor, model = _get_caption_pipeline()
        inputs = processor(image, return_tensors="pt")
        output_ids = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        _log(f"[media] Captioning failed: {e}")
        return ""


def process_image(file_bytes: bytes, user_message: str = "", session_id: str = "default") -> str:
    """
    Processes an uploaded image via Cactus Mirage (22MB), face tagging/recognition,
    and lazy-loaded BLIP fallback via vision_memory.
    """
    from vision_memory import process_uploaded_image
    return process_uploaded_image(file_bytes, user_message=user_message, session_id=session_id)


def process_pdf(file_bytes: bytes) -> str:
    """Extracts text from a PDF file, page by page."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return "[PDF support requires the 'pypdf' package — run: pip install pypdf --break-system-packages]"

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(text.strip())
        if not pages_text:
            return "[PDF contained no extractable text — it may be a scanned/image-only PDF.]"
        combined = _truncate_doc_text("\n\n".join(pages_text))
        return f"[Document text extracted from PDF:]\n{combined}"
    except Exception as e:
        return f"[Could not read PDF: {e}]"


def process_docx(file_bytes: bytes) -> str:
    """Extracts text from a Word (.docx) file."""
    try:
        import docx
    except ImportError:
        return "[DOCX support requires the 'python-docx' package — run: pip install python-docx --break-system-packages]"

    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        if not paragraphs:
            return "[Word document contained no extractable text.]"
        combined = _truncate_doc_text("\n".join(paragraphs))
        return f"[Document text extracted from Word file:]\n{combined}"
    except Exception as e:
        return f"[Could not read Word document: {e}]"


def process_text(file_bytes: bytes) -> str:
    """Reads a plain .txt file directly."""
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin1")
    return f"[Text file contents:]\n{_truncate_doc_text(text.strip())}"


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def process_uploaded_file(filename: str, file_bytes: bytes, user_message: str = "", session_id: str = "default") -> str:
    """
    Dispatches an uploaded file to the right processor based on its
    extension, and returns plain text describing/containing its
    content — ready to be injected into the chat message just like
    normal typed text. This is the only function main.py needs to call.
    """
    ext = Path(filename).suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        return process_image(file_bytes, user_message=user_message, session_id=session_id)
    elif ext == ".pdf":
        return process_pdf(file_bytes)
    elif ext == ".docx":
        return process_docx(file_bytes)
    elif ext == ".txt":
        return process_text(file_bytes)
    else:
        return f"[Unsupported file type: {ext or 'unknown'}. Supported: images (jpg/png/webp/etc), PDF, DOCX, TXT.]"


def decode_app_attachment(attachment: dict, user_message: str = "", session_id: str = "default") -> str:
    """
    Converts one attachment dict sent by the Android app's /chat JSON
    payload into extracted text. Handles data URLs (base64) for all
    file types (images, PDFs, DOCX, and text), with a fallback for
    raw string payloads.
    """
    name = attachment.get("name") or "attachment"
    att_type = attachment.get("type")
    raw = attachment.get("base64") or ""

    # If raw is a data URL (e.g. "data:text/plain;base64,..." or "data:image/png;base64,...")
    if isinstance(raw, str) and (raw.startswith("data:") or "," in raw):
        try:
            b64_part = raw.split(",", 1)[1] if "," in raw else raw
            file_bytes = base64.b64decode(b64_part)
            return process_uploaded_file(name, file_bytes, user_message=user_message, session_id=session_id)
        except Exception as e:
            return f"[Could not decode attachment '{name}': {e}]"

    # Fallback if raw text was sent directly in "base64" field
    if att_type == "text" or (isinstance(raw, str) and not raw.startswith("data:")):
        if raw and raw.strip():
            return f"[Text file '{name}' contents:]\n{_truncate_doc_text(raw.strip())}"
        return f"[Text file '{name}' was sent but appears empty.]"

    try:
        file_bytes = base64.b64decode(raw)
        return process_uploaded_file(name, file_bytes, user_message=user_message, session_id=session_id)
    except Exception as e:
        return f"[Could not decode attachment '{name}': {e}]"
