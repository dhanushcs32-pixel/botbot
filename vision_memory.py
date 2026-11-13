"""
Bot Bot — Vision & Face Memory Manager
Integrates Cactus Mirage (lightweight vision), InsightFace buffalo_sc (face tagging/recognition),
and a strict lazy-loaded BLIP fallback.
"""

import os
import io
import re
import json
import time
import sqlite3
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# Flash 3.8 Rule: Disable Cactus telemetry before importing cactus
os.environ["CACTUS_TELEMETRY"] = "0"

import numpy as np
import cv2
from PIL import Image, ImageOps

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "memory.db"
SAVED_IMAGES_DIR = BASE_DIR / "personal_data" / "saved_images"
FACES_DIR = BASE_DIR / "personal_data" / "faces"

# Ensure directories exist
SAVED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
FACES_DIR.mkdir(parents=True, exist_ok=True)

# Common non-name words to exclude from natural-language tagging
EXCLUDED_WORDS = {
    "a", "an", "the", "my", "our", "your", "his", "her", "their", "its",
    "so", "very", "fine", "crazy", "good", "bad", "weird", "cool", "funny",
    "here", "what", "how", "why", "who", "where", "when", "there",
    "something", "someone", "somebody", "nothing", "nobody", "everything", "everyone",
    "us", "them", "him", "me", "you", "it", "this", "that", "pic", "photo", "image", "picture"
}

TAG_PATTERNS = [
    # "remember this is Rahul", "this is Rahul", "that's Rahul", "thats Rahul", "it's Rahul", "its Rahul"
    r"(?i)\b(?:remember\s+)?(?:this|that|it)(?:'s|s|\s+is)\s+(?:as\s+)?([A-Za-z]+(?:\s+[A-Za-z]+)?)",
    # "tag this as Rahul", "tag Rahul"
    r"(?i)\btag(?:\s+this\s+as|\s+as|\s+this)?\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
    # "here is Rahul", "pic of Rahul", "photo of Rahul", "me and Rahul"
    r"(?i)\b(?:here\s+is|pic\s+of|photo\s+of|picture\s+of|me\s+and)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
]

# Strict Lazy-loaded Singletons (Zero load at import/startup)
_mirage = None
_face_app = None
_blip_processor = None
_blip_model = None


def _init_db(db_path: Path = DB_PATH) -> None:
    """Initializes the known_faces and saved_images tables in SQLite."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS known_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_name TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            crop_path TEXT,
            source_image_path TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS saved_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            image_path TEXT NOT NULL,
            tagged_people TEXT,
            scene_summary TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()


# Run DB initialization
_init_db(DB_PATH)


def get_mirage():
    """Lazily loads Cactus Mirage singleton on first image use."""
    global _mirage
    if _mirage is None:
        import cactus
        _mirage = cactus.Mirage()
    return _mirage


def get_face_app():
    """Lazily loads InsightFace buffalo_sc singleton on first face check."""
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        _face_app.prepare(ctx_id=-1, det_size=(640, 640))
    return _face_app


def _run_blip_fallback(pil_image: Image.Image) -> str:
    """
    CRITICAL BLIP RULE: Loads BLIP into memory ONLY IF Mirage raises an exception
    or returns completely empty data on an unsupported or corrupted image.
    """
    global _blip_processor, _blip_model
    print("[vision] Mirage unsupported image — falling back to lazy-loaded BLIP")
    if _blip_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    inputs = _blip_processor(pil_image, return_tensors="pt")
    output_ids = _blip_model.generate(**inputs, max_new_tokens=50)
    caption = _blip_processor.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption


def extract_tag_intent(user_message: str) -> Optional[str]:
    """
    Detects natural language person tagging intent from user messages.
    Returns cleaned 1-2 word name or None.
    """
    if not user_message or not user_message.strip():
        return None

    text = user_message.strip()

    for pattern in TAG_PATTERNS:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1).strip()
            candidate = re.sub(r"[^\w\s]", "", candidate).strip()
            words = candidate.split()
            if not words or len(words) > 2:
                continue

            lower_words = [w.lower() for w in words]
            if any(w in EXCLUDED_WORDS for w in lower_words):
                continue

            if any(len(w) < 2 for w in words):
                continue

            return " ".join(w.capitalize() for w in words)

    return None


class VisionMemoryManager:
    """Manages visual scene extraction, face tagging, and recognition."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        _init_db(self.db_path)

    def extract_tag_intent(self, user_message: str) -> Optional[str]:
        return extract_tag_intent(user_message)

    def recognize_faces(self, img_bgr: np.ndarray, threshold: float = 0.45) -> List[str]:
        """
        Extracts 512-d embeddings for all faces and matches via cosine similarity
        against known_faces in SQLite.
        """
        app = get_face_app()
        faces = app.get(img_bgr)
        if not faces:
            return []

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT person_name, embedding_json FROM known_faces").fetchall()

        if not rows:
            return ["unrecognized person" for _ in faces]

        known: List[Tuple[str, np.ndarray]] = []
        for name, emb_json in rows:
            try:
                emb = np.array(json.loads(emb_json), dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                known.append((name, emb))
            except Exception:
                continue

        if not known:
            return ["unrecognized person" for _ in faces]

        results = []
        for face in faces:
            q_emb = np.array(face.embedding, dtype=np.float32)
            q_norm = np.linalg.norm(q_emb)
            if q_norm > 0:
                q_emb = q_emb / q_norm

            best_name = None
            best_score = -1.0

            for name, ref_emb in known:
                sim = float(np.dot(q_emb, ref_emb))
                if sim > best_score:
                    best_score = sim
                    best_name = name

            if best_score >= threshold and best_name:
                results.append(f"{best_name} (recognized, confidence={best_score:.2f})")
            else:
                results.append("unrecognized person")

        return results

    def tag_face(self, person_name: str, img_bgr: np.ndarray, source_image_path: str = "") -> bool:
        """
        Extracts the primary face embedding and saves crop + embedding to known_faces.
        """
        app = get_face_app()
        faces = app.get(img_bgr)
        if not faces:
            return False

        # Pick primary face by bounding box area
        def face_area(f):
            box = f.bbox
            return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

        primary_face = max(faces, key=face_area)
        q_emb = np.array(primary_face.embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        # Save face crop
        name_lower = re.sub(r"[^\w]", "_", person_name.lower())
        person_dir = FACES_DIR / name_lower
        person_dir.mkdir(parents=True, exist_ok=True)

        existing = list(person_dir.glob("face_*.jpg"))
        crop_filename = f"face_{len(existing) + 1}.jpg"
        crop_path = person_dir / crop_filename

        h, w = img_bgr.shape[:2]
        box = [int(v) for v in primary_face.bbox]
        x1, y1 = max(0, box[0]), max(0, box[1])
        x2, y2 = min(w, box[2]), min(h, box[3])
        crop = img_bgr[y1:y2, x1:x2]

        if crop.size > 0:
            cv2.imwrite(str(crop_path), crop)
        else:
            crop_path = None

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO known_faces (person_name, embedding_json, crop_path, source_image_path)
                VALUES (?, ?, ?, ?)
                """,
                (
                    person_name,
                    json.dumps(q_emb.tolist()),
                    str(crop_path) if crop_path else None,
                    str(source_image_path)
                )
            )
            conn.commit()
        return True

    def process_uploaded_image(
        self,
        image_bytes: bytes,
        user_message: str = "",
        session_id: str = "default"
    ) -> str:
        """
        Processes an uploaded image with face recognition, tagging, and Mirage scene description.
        Returns clean [IMAGE CONTENT — ...] string.
        """
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        except Exception as e:
            return f"[Could not open image: {e}]"

        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Check natural language tagging intent
        tag_name = self.extract_tag_intent(user_message)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r"[^\w]", "_", tag_name.lower()) if tag_name else "image"
        img_filename = f"{timestamp}_{safe_name}.jpg"
        save_img_path = SAVED_IMAGES_DIR / img_filename

        # Save uploaded image to disk
        try:
            pil_img.save(str(save_img_path), "JPEG", quality=90)
        except Exception:
            cv2.imwrite(str(save_img_path), img_bgr)

        # Handle face tagging if intent detected
        tagged_note = ""
        if tag_name:
            tagged = self.tag_face(tag_name, img_bgr, source_image_path=str(save_img_path))
            if tagged:
                tagged_note = f"Tagged & saved face for: {tag_name}."

        # Recognize all faces in the photo
        recognized_people = self.recognize_faces(img_bgr)
        if recognized_people:
            people_str = ", ".join(recognized_people)
        else:
            people_str = "none"

        # Run Cactus Mirage scene/object/text description
        mirage_failed = False
        scene_str = ""
        objects_str = ""
        text_str = ""

        try:
            mirage = get_mirage()
            res = mirage.describe(str(save_img_path), sections=["scene", "objects", "text"])
            if not isinstance(res, dict):
                raise ValueError(f"Mirage returned non-dict: {type(res)}")

            scene_data = res.get("scene")
            objects_data = res.get("objects")
            text_data = res.get("text")

            # Extract scene
            if isinstance(scene_data, dict):
                scene_str = ", ".join(f"{k}: {v}" for k, v in scene_data.items() if v)
            elif scene_data:
                scene_str = str(scene_data)

            # Extract objects
            if isinstance(objects_data, list):
                obj_names = []
                for obj in objects_data:
                    if isinstance(obj, dict) and obj.get("name"):
                        obj_names.append(obj["name"])
                if obj_names:
                    objects_str = ", ".join(obj_names)

            # Extract text
            if isinstance(text_data, list):
                text_items = []
                for t in text_data:
                    if isinstance(t, dict) and t.get("string"):
                        text_items.append(t["string"])
                if text_items:
                    text_str = " ".join(text_items)

            if not scene_str and not objects_str and not text_str:
                raise ValueError("Mirage returned empty scene, objects, and text")

        except Exception as e:
            mirage_failed = True
            caption = _run_blip_fallback(pil_img)
            scene_str = caption if caption else "unspecified scene"
            objects_str = "none"
            text_str = "none"

        # Normalize empty fields
        if not scene_str:
            scene_str = "none"
        if not objects_str:
            objects_str = "none"
        if not text_str:
            text_str = "none"

        # Format unified context string
        prefix_tag = f"{tagged_note} " if tagged_note else ""
        analysis = (
            f"[IMAGE CONTENT — {prefix_tag}People: {people_str} | "
            f"Setting: {scene_str} | Objects: {objects_str} | Visible Text: {text_str}]"
        )

        # Log to saved_images table in SQLite
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO saved_images (session_id, image_path, tagged_people, scene_summary)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, str(save_img_path), people_str, analysis)
                )
                conn.commit()
        except Exception as e:
            print(f"[vision] Failed to log to saved_images: {e}")

        return analysis


# Global module singleton & helper exports
_vision_manager = VisionMemoryManager()


def process_uploaded_image(image_bytes: bytes, user_message: str = "", session_id: str = "default") -> str:
    return _vision_manager.process_uploaded_image(image_bytes, user_message=user_message, session_id=session_id)


def recognize_faces(img_bgr: np.ndarray, threshold: float = 0.45) -> List[str]:
    return _vision_manager.recognize_faces(img_bgr, threshold=threshold)


def tag_face(person_name: str, img_bgr: np.ndarray, source_image_path: str = "") -> bool:
    return _vision_manager.tag_face(person_name, img_bgr, source_image_path=source_image_path)

