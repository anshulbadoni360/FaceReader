"""
Face Attribute Prediction — Gender, Age, Glasses, Beard, Moustache.

Gender + Age  → DeepFace (VGG-Face, auto-downloads on first run)
Glasses       → OpenAI CLIP Zero-Shot (auto-downloads on first run)
Beard         → OpenAI CLIP Zero-Shot (auto-downloads on first run)
Moustache     → OpenAI CLIP Zero-Shot (auto-downloads on first run)
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MIN_FACE_SIZE       = 48
MAX_FACE_SIZE       = 640
MAX_YAW_ACCESSORIES = 40.0   # degrees — skip accessories beyond this yaw

SAFE_DEFAULTS: Dict[str, str] = {
    "Gender":    "Unknown",
    "Age":       "Unknown",
    "Glasses":   "Unknown",
    "Moustache": "None",
    "Beard":     "None",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_crop(crop: np.ndarray) -> Tuple[bool, str]:
    if crop is None or not isinstance(crop, np.ndarray):
        return False, "None"
    if crop.ndim != 3 or crop.shape[2] != 3:
        return False, f"bad shape {crop.shape}"
    h, w = crop.shape[:2]
    if h < MIN_FACE_SIZE or w < MIN_FACE_SIZE:
        return False, f"too small ({w}x{h})"
    if cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).std() < 5.0:
        return False, "blank image"
    return True, ""


def _safe_crop(image: np.ndarray, bbox, padding: float = 0.25) -> Optional[np.ndarray]:
    try:
        if image is None or image.size == 0:
            return None
        h_img, w_img = image.shape[:2]
        if hasattr(bbox, "x1"):
            x1, y1, w, h = int(bbox.x1), int(bbox.y1), int(bbox.width), int(bbox.height)
        else:
            x1, y1, w, h = [int(v) for v in bbox]
        if w <= 0 or h <= 0:
            return None
        px, py = int(w * padding), int(h * padding)
        x1 = max(0, x1 - px)
        y1 = max(0, y1 - py)
        x2 = min(w_img, x1 + w + 2 * px)
        y2 = min(h_img, y1 + h + 2 * py)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image[y1:y2, x1:x2].copy()
        ch, cw = crop.shape[:2]
        if max(ch, cw) > MAX_FACE_SIZE:
            s    = MAX_FACE_SIZE / max(ch, cw)
            crop = cv2.resize(crop,
                              (max(1, int(cw * s)), max(1, int(ch * s))),
                              interpolation=cv2.INTER_AREA)
        return crop
    except Exception as e:
        logger.warning(f"_safe_crop: {e}")
        return None


def _get_region(face: np.ndarray, y_range: Tuple[float, float]) -> Optional[np.ndarray]:
    h  = face.shape[0]
    y1 = int(h * y_range[0])
    y2 = int(h * y_range[1])
    r  = face[y1:y2, :]
    return r if r.size > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# DeepFace — Gender + Age (UNTOUCHED)
# ─────────────────────────────────────────────────────────────────────────────

class _DeepFaceBackend:
    def __init__(self):
        from deepface import DeepFace
        self._df = DeepFace
        try:
            dummy = np.ones((100, 100, 3), dtype=np.uint8) * 128
            self._df.analyze(dummy, actions=["gender", "age"],
                             enforce_detection=False, silent=True,
                             detector_backend="skip")
        except Exception:
            pass
        logger.info("DeepFace backend ready")

    def predict(self, face_crop: np.ndarray) -> Dict[str, str]:
        try:
            result = self._df.analyze(
                face_crop,
                actions=["gender", "age"],
                enforce_detection=False,
                silent=True,
                detector_backend="skip",
            )
            if isinstance(result, list):
                result = result[0]

            gender_dict = result.get("gender", {})
            gender      = "Male" if gender_dict.get("Man", 0) >= gender_dict.get("Woman", 0) else "Female"

            age_raw = result.get("age", None)
            age_str = str(int(age_raw)) if age_raw is not None else "Unknown"

            return {"Gender": gender, "Age": age_str}
        except Exception as e:
            logger.warning(f"DeepFace error: {e}")
            return {}


# ─────────────────────────────────────────────────────────────────────────────
# Deep Learning Accessory Detector (OpenAI CLIP)
# ─────────────────────────────────────────────────────────────────────────────

class _DeepLearningAccessoryDetector:
    """
    Uses OpenAI's CLIP Zero-Shot image classification to accurately detect 
    Glasses, Beard, and Moustache using Deep Neural Networks.
    Auto-downloads weights to your PC on the first run.
    """
    def __init__(self):
        self.classifier = None
        try:
            import torch
            from transformers import pipeline
            
            # Check if GPU is available, otherwise use CPU
            device = 0 if torch.cuda.is_available() else -1
            
            # This will auto-download the model (~350MB) on the very first run
            self.classifier = pipeline(
                "zero-shot-image-classification",
                model="openai/clip-vit-base-patch32",
                device=device
            )
            logger.info("CLIP backend ready for accessories")
        except ImportError:
            logger.error("transformers or torch not installed. Run: pip install transformers torch")
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")

    def predict(self, face_crop: np.ndarray, head_yaw: float = 0.0) -> Dict[str, str]:
        # Default fallback
        result = {"Glasses": "Unknown", "Moustache": "Unknown", "Beard": "Unknown"}
        
        if self.classifier is None:
            return result
            
        # Skip accessory detection if head is turned too far sideways
        if abs(head_yaw) > MAX_YAW_ACCESSORIES:
            return result
            
        try:
            from PIL import Image
            
            # CLIP requires a PIL Image (RGB), not an OpenCV BGR array
            rgb_image = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # 1. Predict Glasses
            glass_preds = self.classifier(
                pil_image, 
                candidate_labels=["wearing eyeglasses or sunglasses", "no glasses"]
            )
            if glass_preds[0]['label'] == "wearing eyeglasses or sunglasses":
                result["Glasses"] = "Yes"
            else:
                result["Glasses"] = "No"

            # 2. Predict Beard
            beard_preds = self.classifier(
                pil_image, 
                candidate_labels=["man with a beard", "clean shaven face"]
            )
            if beard_preds[0]['label'] == "man with a beard":
                result["Beard"] = "Yes"
            else:
                result["Beard"] = "None"

            # 3. Predict Moustache
            stache_preds = self.classifier(
                pil_image, 
                candidate_labels=["man with a moustache", "no moustache"]
            )
            if stache_preds[0]['label'] == "man with a moustache":
                result["Moustache"] = "Yes"
            else:
                result["Moustache"] = "None"

        except Exception as e:
            logger.debug(f"Deep Learning Accessories Error: {e}")
            
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class FaceAttributePredictor:
    """
    Face attribute predictor — AI tuned.

    Gender + Age  → DeepFace (~97% gender, exact age number)
    Glasses       → OpenAI CLIP
    Beard         → OpenAI CLIP
    Moustache     → OpenAI CLIP

    Never raises. Always returns all 5 keys.
    """

    _instance: Optional["FaceAttributePredictor"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Gender + Age
        self._ga: Optional[_DeepFaceBackend] = None
        try:
            self._ga = _DeepFaceBackend()
            logger.info("✓ Gender/Age: DeepFace")
        except Exception as e:
            logger.warning(f"⚠ DeepFace failed: {e}")

        # Accessories
        self._accessories = _DeepLearningAccessoryDetector()
        logger.info("✓ Accessories: OpenAI CLIP (Deep Learning)")

        self._initialized = True
        logger.info("FaceAttributePredictor ready")

    def predict(
        self,
        image: np.ndarray,
        face_bbox,
        head_yaw: float = 0.0,
    ) -> Dict[str, str]:
        """
        Never raises. Always returns all 5 keys.

        Args:
            image:     Full BGR image
            face_bbox: .x1/.y1/.width/.height object OR (x1, y1, w, h)
            head_yaw:  Head yaw in degrees from geometry dict
        """
        result    = dict(SAFE_DEFAULTS)
        face_crop = _safe_crop(image, face_bbox, padding=0.25)

        if face_crop is None:
            return result

        ok, reason = _validate_crop(face_crop)
        if not ok:
            logger.warning(f"Invalid crop: {reason}")
            return result

        # Gender + Age
        if self._ga is not None:
            try:
                t0 = time.perf_counter()
                ga = self._ga.predict(face_crop)
                ms = (time.perf_counter() - t0) * 1000
                if "Gender" in ga: result["Gender"] = ga["Gender"]
                if "Age"    in ga: result["Age"]    = ga["Age"]
                logger.debug(f"GA: {ga} ({ms:.0f}ms)")
            except Exception as e:
                logger.error(f"GA error: {e}", exc_info=True)

        # Accessories
        try:
            acc = self._accessories.predict(face_crop, head_yaw)
            result.update(acc)
        except Exception as e:
            logger.error(f"Accessory error: {e}", exc_info=True)

        return result


def get_attribute_predictor() -> FaceAttributePredictor:
    return FaceAttributePredictor()