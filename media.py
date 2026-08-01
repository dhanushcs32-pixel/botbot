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

import io
import base64
from pathlib import Path

from PIL import Image
import pytesseract

# Windows doesn't add Tesseract to PATH by default — point pytesseract
# directly at the installed binary so it can find it.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

_caption_pipeline = None  # lazy-loaded — only load the model if an image actually needs captioning


def _get_caption_pipeline():
    """
    Loads the BLIP image-captioning model on first use only. This
    avoids a slow, multi-second model load on every server startup
    when most messages won't include a photo at all.
    """
    global _caption_pipeline
    if _caption_pipeline is None:
        from transformers import pipeline
        _caption_pipeline = pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base"
        )
    return _caption_pipeline


def _ocr_image(image: Image.Image) -> str:
    """Extracts any printed/handwritten text found in the image."""
    try:
        return pytesseract.image_to_string(image).strip()
    except Exception as e:
        print(f"OCR failed: {e}")
        return ""


def _caption_image(image: Image.Image) -> str:
    """Generates a natural-language description of what the image shows."""
    try:
        pipe = _get_caption_pipeline()
        result = pipe(image)
        if result and isinstance(result, list):
            return result[0].get("generated_text", "").strip()
        return ""
    except Exception as e:
        print(f"Captioning failed: {e}")
        return ""


def process_image(file_bytes: bytes) -> str:
    """
    Processes an uploaded image and returns a text description ready
    to inject into the chat message. OCR runs first since it's fast;
    if it finds a meaningful amount of text, this is probably a
    screenshot or photographed document, so captioning is skipped
    entirely. If OCR finds little/no text, this is probably a real
    photo, so a caption is generated instead to describe what's shown.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        return f"[Could not open image: {e}]"

    ocr_text = _ocr_image(image)

    if len(ocr_text) >= 15:
        return f"[Image contains the following text, extracted via OCR:]\n{ocr_text}"

    caption = _caption_image(image)
    if caption:
        if ocr_text:
            return f"[Image shows: {caption}. Some additional text detected: {ocr_text}]"
        return f"[Image shows: {caption}]"

    if ocr_text:
        return f"[Image contains the following text, extracted via OCR:]\n{ocr_text}"

    return "[An image was sent, but no readable text or clear description could be extracted from it.]"


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
        combined = "\n\n".join(pages_text)
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
        combined = "\n".join(paragraphs)
        return f"[Document text extracted from Word file:]\n{combined}"
    except Exception as e:
        return f"[Could not read Word document: {e}]"


def process_text(file_bytes: bytes) -> str:
    """Reads a plain .txt file directly."""
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin1")
    return f"[Text file contents:]\n{text.strip()}"


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def process_uploaded_file(filename: str, file_bytes: bytes) -> str:
    """
    Dispatches an uploaded file to the right processor based on its
    extension, and returns plain text describing/containing its
    content — ready to be injected into the chat message just like
    normal typed text. This is the only function main.py needs to call.
    """
    ext = Path(filename).suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        return process_image(file_bytes)
    elif ext == ".pdf":
        return process_pdf(file_bytes)
    elif ext == ".docx":
        return process_docx(file_bytes)
    elif ext == ".txt":
        return process_text(file_bytes)
    else:
        return f"[Unsupported file type: {ext or 'unknown'}. Supported: images (jpg/png/webp/etc), PDF, DOCX, TXT.]"


def decode_app_attachment(attachment: dict) -> str:
    """
    Converts one attachment dict sent by the Android app's /chat JSON
    payload into extracted text. The app packages attachments
    differently depending on type:
      - images/PDFs: "base64" field is a full data URL, e.g.
        "data:image/png;base64,AAAA..." (from FileReader.readAsDataURL)
      - text files: "base64" field is actually already raw decoded
        text, not base64 (from FileReader.readAsText) — the field
        name is just a holdover from how the app's state is shaped

    This mirrors that distinction so both cases are handled correctly,
    then reuses process_uploaded_file() for the actual extraction.
    """
    name = attachment.get("name") or "attachment"
    att_type = attachment.get("type")
    raw = attachment.get("base64") or ""

    if att_type == "text":
        return f"[Text file '{name}' contents:]\n{raw.strip()}" if raw.strip() else \
            f"[Text file '{name}' was sent but appears empty.]"

    # image/pdf case: raw is a data URL like "data:image/png;base64,...."
    if "," in raw:
        raw = raw.split(",", 1)[1]

    try:
        file_bytes = base64.b64decode(raw)
    except Exception as e:
        return f"[Could not decode attachment '{name}': {e}]"

    return process_uploaded_file(name, file_bytes)
