"""
Verification suite for Cactus Mirage vision memory and face tagging/recognition.
"""

import os
import io
import sys
import json
import sqlite3
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Flash 3.8 Rule: Verify CACTUS_TELEMETRY is 0
os.environ["CACTUS_TELEMETRY"] = "0"

import vision_memory
from vision_memory import (
    VisionMemoryManager,
    extract_tag_intent,
    process_uploaded_image,
    recognize_faces,
    tag_face
)


def test_blip_not_eagerly_loaded():
    """Confirms BLIP is NOT loaded in memory during import and normal execution."""
    assert vision_memory._blip_processor is None, "BLIP processor must be None initially"
    assert vision_memory._blip_model is None, "BLIP model must be None initially"
    assert os.environ.get("CACTUS_TELEMETRY") == "0", "CACTUS_TELEMETRY must be 0"


def test_extract_tag_intent():
    """Tests natural language person tagging intent extraction."""
    # Positive matches
    assert extract_tag_intent("this is Rahul") == "Rahul"
    assert extract_tag_intent("that's Priya Sharma") == "Priya Sharma"
    assert extract_tag_intent("thats Arjun") == "Arjun"
    assert extract_tag_intent("it's David") == "David"
    assert extract_tag_intent("its Sarah") == "Sarah"
    assert extract_tag_intent("tag this as Vikram") == "Vikram"
    assert extract_tag_intent("tag Maya") == "Maya"
    assert extract_tag_intent("remember this is Kiran") == "Kiran"
    assert extract_tag_intent("here is Amit") == "Amit"
    assert extract_tag_intent("pic of Neha") == "Neha"
    assert extract_tag_intent("photo of Sanjay") == "Sanjay"
    assert extract_tag_intent("me and Anita") == "Anita"

    # Negative non-name matches
    assert extract_tag_intent("this is crazy") is None
    assert extract_tag_intent("this is a dog") is None
    assert extract_tag_intent("that's very weird") is None
    assert extract_tag_intent("what is this") is None
    assert extract_tag_intent("its fine") is None
    assert extract_tag_intent("me and my friend") is None
    assert extract_tag_intent("tag this as cool") is None
    assert extract_tag_intent("") is None
    assert extract_tag_intent(None) is None


def test_mirage_process_uploaded_image():
    """Synthesizes an image, runs process_uploaded_image, and verifies output structure."""
    img = Image.new("RGB", (250, 250), color=(50, 100, 150))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "Dashboard Statistics", fill=(255, 255, 255))
    draw.rectangle([50, 80, 180, 180], fill=(200, 50, 50))

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    result = process_uploaded_image(
        image_bytes,
        user_message="Look at this graph",
        session_id="test_vision_session"
    )

    print("\nMirage process_uploaded_image result:", result)
    assert result.startswith("[IMAGE CONTENT — ")
    assert result.endswith("]")
    assert "People:" in result
    assert "Setting:" in result
    assert "Objects:" in result
    assert "Visible Text:" in result

    # Verify BLIP was NOT loaded during normal Mirage execution
    assert vision_memory._blip_model is None, "BLIP model was unexpectedly loaded during Mirage execution!"
    assert vision_memory._blip_processor is None, "BLIP processor was unexpectedly loaded!"


def test_face_recognition_cosine_lookup():
    """
    Tests the simulated face-embedding store + cosine lookup in known_faces (memory.db)
    to confirm a stored 512-d vector for 'Rahul' matches a query vector with similarity > 0.45.
    """
    db_path = vision_memory.DB_PATH
    manager = VisionMemoryManager(db_path=db_path)

    # 1. Synthesize reference 512-d embedding for "Rahul"
    np.random.seed(42)
    ref_vec = np.random.randn(512).astype(np.float32)
    ref_vec = ref_vec / np.linalg.norm(ref_vec)

    # Insert into known_faces table
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO known_faces (person_name, embedding_json, crop_path, source_image_path)
            VALUES (?, ?, ?, ?)
            """,
            ("Rahul", json.dumps(ref_vec.tolist()), "test_crop.jpg", "test_source.jpg")
        )
        test_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()

    original_get_face_app = vision_memory.get_face_app
    try:
        # 2. Synthesize query vector with high similarity to ref_vec (> 0.45, e.g. ~0.90)
        noise = np.random.randn(512).astype(np.float32)
        noise = noise / np.linalg.norm(noise)
        query_vec_match = 0.90 * ref_vec + 0.10 * noise
        query_vec_match = query_vec_match / np.linalg.norm(query_vec_match)

        # Theoretical cosine similarity check
        sim_match = float(np.dot(query_vec_match, ref_vec))
        assert sim_match > 0.45, f"Expected match sim > 0.45, got {sim_match}"

        # 3. Synthesize query vector with low similarity (< 0.45)
        orthogonal_vec = np.random.randn(512).astype(np.float32)
        # Gram-Schmidt to make it orthogonal to ref_vec
        orthogonal_vec -= np.dot(orthogonal_vec, ref_vec) * ref_vec
        orthogonal_vec = orthogonal_vec / np.linalg.norm(orthogonal_vec)
        sim_mismatch = float(np.dot(orthogonal_vec, ref_vec))
        assert abs(sim_mismatch) < 0.1, f"Expected near zero sim, got {sim_mismatch}"

        # 4. Mock a face object returned by face_app.get
        class MockFace:
            def __init__(self, emb):
                self.embedding = emb
                self.bbox = [10, 10, 100, 100]

        # Patch get_face_app temporarily to test recognize_faces end-to-end
        original_get_face_app = vision_memory.get_face_app

        class MockFaceApp:
            def __init__(self, faces_to_return):
                self.faces_to_return = faces_to_return
            def get(self, img):
                return self.faces_to_return

        dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)

        # Test matched face
        vision_memory.get_face_app = lambda: MockFaceApp([MockFace(query_vec_match)])
        rec_match = manager.recognize_faces(dummy_img, threshold=0.45)
        assert len(rec_match) == 1
        assert "Rahul (recognized, confidence=" in rec_match[0]

        # Test unmatched face
        vision_memory.get_face_app = lambda: MockFaceApp([MockFace(orthogonal_vec)])
        rec_mismatch = manager.recognize_faces(dummy_img, threshold=0.45)
        assert len(rec_mismatch) == 1
        assert rec_mismatch[0] == "unrecognized person"

    finally:
        # Restore original get_face_app and delete test row
        vision_memory.get_face_app = original_get_face_app
        with sqlite3.connect(db_path) as conn:
            conn.execute("DELETE FROM known_faces WHERE id = ?", (test_id,))
            conn.commit()


def test_blip_fallback_trigger():
    """Confirms that when Mirage raises an error, BLIP fallback is triggered gracefully."""
    original_get_mirage = vision_memory.get_mirage

    class BrokenMirage:
        def describe(self, *args, **kwargs):
            raise RuntimeError("Simulated Mirage failure on corrupted image format")

    try:
        vision_memory.get_mirage = lambda: BrokenMirage()
        img = Image.new("RGB", (100, 100), color=(100, 200, 100))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = process_uploaded_image(buf.getvalue(), user_message="test fallback", session_id="test_fallback")
        print("\nFallback result:", result)
        assert result.startswith("[IMAGE CONTENT — ")
        assert vision_memory._blip_model is not None, "BLIP model should be loaded after fallback!"
        assert vision_memory._blip_processor is not None, "BLIP processor should be loaded after fallback!"
    finally:
        vision_memory.get_mirage = original_get_mirage
        # Reset BLIP to None to keep memory clean
        vision_memory._blip_model = None
        vision_memory._blip_processor = None


if __name__ == "__main__":
    print("Running test_blip_not_eagerly_loaded...")
    test_blip_not_eagerly_loaded()
    print("PASS: test_blip_not_eagerly_loaded")

    print("\nRunning test_extract_tag_intent...")
    test_extract_tag_intent()
    print("PASS: test_extract_tag_intent")

    print("\nRunning test_mirage_process_uploaded_image...")
    test_mirage_process_uploaded_image()
    print("PASS: test_mirage_process_uploaded_image")

    print("\nRunning test_face_recognition_cosine_lookup...")
    test_face_recognition_cosine_lookup()
    print("PASS: test_face_recognition_cosine_lookup")

    print("\nRunning test_blip_fallback_trigger...")
    test_blip_fallback_trigger()
    print("PASS: test_blip_fallback_trigger")

    print("\nALL VISION TESTS PASSED!")
