"""Face emotion analysis service."""

import json
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image, ExifTags
import io

from app.config import settings
from app.models.nn_model import AUMultiTaskModel
from app.utils.constants import AU_LIST, FEATURE_NAMES, EMOTION_CONFIG

# Import face pipeline
from face.pipeline import FacePipeline

# Import attribute predictor
from app.services.face_attributes import get_attribute_predictor


def load_image_with_exif_fix(image_data) -> np.ndarray:
    """
    Load image and correctly handle EXIF orientation.
    Works with both file path and bytes.
    """
    if isinstance(image_data, (str, Path)):
        img = Image.open(image_data)
    elif isinstance(image_data, bytes):
        img = Image.open(io.BytesIO(image_data))
    else:
        raise ValueError("image_data must be path or bytes")
    
    try:
        exif = img.getexif()
        orientation = None
        if exif:
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag == 'Orientation':
                    orientation = value
                    break
        
        if orientation:
            if orientation == 2:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 4:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            elif orientation == 5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 7:
                img = img.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
                
    except (AttributeError, KeyError, IndexError):
        pass
    
    # Convert to BGR for OpenCV
    img_array = np.array(img)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    else:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    return img_array


class FaceEmotionAnalyzer:
    """Singleton analyzer for face emotion detection."""
    
    _instance: Optional['FaceEmotionAnalyzer'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.scaler_mean, self.scaler_scale = self._load_model()
        self.pipeline = FacePipeline()

        # ── Face attribute predictor (gender, age, glasses, beard, moustache)
        self.attribute_predictor = get_attribute_predictor()

        self._initialized = True
        print(f"✓ Analyzer initialized on {self.device}")
    
    def _load_model(self) -> Tuple[AUMultiTaskModel, np.ndarray, np.ndarray]:
        """Load trained model and scaler."""
        with open(settings.SCALER_PATH, "r") as f:
            sp = json.load(f)
        mean = np.array(sp["mean"], dtype=np.float32)
        scale = np.array(sp["scale"], dtype=np.float32)

        ckpt = torch.load(settings.MODEL_PATH, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})
        
        model = AUMultiTaskModel(
            n_features=40, 
            n_aus=len(AU_LIST),
            hidden_dims=cfg.get("hidden_dims", [256, 256, 128]), 
            dropout=0.0
        )
        model.to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        return model, mean, scale

    def _try_rotations(self, image: np.ndarray) -> Tuple[List, np.ndarray, int]:
        """
        Try multiple rotations to find a face.
        
        Returns:
            (results, rotated_image, rotation_applied)
        """
        rotations = [
            (0,   None),
            (180, cv2.ROTATE_180),
            (90,  cv2.ROTATE_90_CLOCKWISE),
            (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]
        
        for angle, rotate_code in rotations:
            rotated = cv2.rotate(image, rotate_code) if rotate_code is not None else image
            results = self.pipeline.process(rotated, max_faces=10)
            if results and results[0].geometry is not None:
                return results, rotated, angle
        
        return [], image, 0

    def _predict_aus(self, geometry: dict) -> Dict[str, float]:
        """Predict Action Unit probabilities."""
        x = np.array([geometry[k] for k in FEATURE_NAMES], dtype=np.float32)
        x = (x - self.scaler_mean) / self.scaler_scale
        x_t = torch.tensor(x).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = self.model(x_t)

        probs = torch.sigmoid(logits).cpu().numpy()[0]
        return {AU_LIST[i]: float(probs[i]) for i in range(len(AU_LIST))}

    def _compute_emotions(self, au_probs: Dict[str, float]) -> Dict[str, float]:
        """Compute emotion scores with improved conflict resolution."""
        scores = {}
        
        au12 = au_probs.get("AU12", 0)
        au15 = au_probs.get("AU15", 0)
        au01 = au_probs.get("AU01", 0)
        au04 = au_probs.get("AU04", 0)
        au06 = au_probs.get("AU06", 0)
        au20 = au_probs.get("AU20", 0)

        for emotion, config in EMOTION_CONFIG.items():
            if emotion == "Neutral":
                total_activity = sum(au_probs.values())
                scores[emotion] = max(0.05, 1.0 - total_activity / len(au_probs) * 1.5)
                continue

            required  = config["required"]
            enhancers = config["enhancers"]
            inhibitors = config["inhibitors"]

            base = np.mean([au_probs.get(au, 0) for au in required]) if required else 0.2
            if enhancers:
                base += np.mean([au_probs.get(au, 0) for au in enhancers]) * 0.15
            if inhibitors:
                base -= np.mean([au_probs.get(au, 0) for au in inhibitors]) * 0.5
            scores[emotion] = max(0, min(1, base))

        # Conflict resolution
        if au12 > 0.6 and au12 > au15 + 0.2:
            scores["Happy"] = min(1.0, scores["Happy"] * 1.3)
            scores["Sad"]   = scores["Sad"] * 0.3
        elif au15 > 0.4 and au01 > 0.3 and au12 < 0.4:
            scores["Sad"]   = min(1.0, scores["Sad"] * 1.3)
            scores["Happy"] = scores["Happy"] * 0.3

        if au06 > 0.5 and au12 > 0.5:
            scores["Happy"] = min(1.0, scores["Happy"] * 1.2)

        if au12 > 0.5:
            scores["Angry"] = scores["Angry"] * 0.4

        surprise = scores.get("Surprised", 0.0)
        scared   = scores.get("Scared",    0.0)
        if surprise > 0.25 and au20 > 0.5 and au04 > 0.25:
            scores["Scared"]    = max(scared, surprise * 1.1)
            scores["Surprised"] = surprise * 0.6

        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return scores

    def _compute_valence_arousal(
        self, emotions: Dict[str, float], au_probs: Dict[str, float]
    ) -> Tuple[float, float]:
        """Compute valence and arousal."""
        positive = emotions.get("Happy", 0) + emotions.get("Surprised", 0) * 0.3
        negative = sum(emotions.get(e, 0) for e in ["Sad", "Angry", "Scared", "Disgusted", "Confusion"])
        valence  = (positive - negative + 1) / 2
        arousal  = np.mean([au_probs.get(au, 0) for au in ["AU05", "AU20", "AU25", "AU26", "AU27"]])
        return float(np.clip(valence, 0, 1)), float(np.clip(arousal, 0, 1))

    def _empty_response(self, num_faces: int = 0, error: str = None) -> Dict[str, Any]:
        """Return empty response."""
        response = {
            "FaceAnalyzed": False,
            "FacialExpressions": None,
            "ActionUnits": [],
            "Confidence": 0.0,
            "HeadOrientation": [0.0, 0.0, 0.0],
            "BoundingBox": [0, 0, 0, 0],
            "NumberOfFaces": num_faces,
            "Characteristics": {
                "Gender": "Unknown", "Age": "Unknown",
                "Glasses": "Unknown", "Moustache": "None", "Beard": "None"
            },
            "Focus": 0.0,
            "ActionableEmotion": None,
            "RotationApplied": 0
        }
        if error:
            response["Error"] = error
        return response

    def _build_response(
        self,
        image: np.ndarray,          # ← full (possibly rotated) BGR image
        face,
        results: List,
        geometry: dict,
        au_probs: Dict[str, float],
        emotions: Dict[str, float],
        rotation: int,
    ) -> Dict[str, Any]:
        """Build success response, including face attribute predictions."""
        dominant = max(emotions, key=emotions.get)
        valence, arousal = self._compute_valence_arousal(emotions, au_probs)

        # ── Action Units ─────────────────────────────────────────────────────
        action_units = [
            {"Name": au.replace("AU0", "AU"), "Value": au_probs[au]}
            for au in AU_LIST
        ]

        # ── Characteristics (gender, age, glasses, beard, moustache) ─────────
        head_yaw = float(geometry.get("head_yaw", 0.0))
        characteristics = self.attribute_predictor.predict(
            image=image,
            face_bbox=face.face_box,
            head_yaw=head_yaw,
        )

        return {
            "FaceAnalyzed": True,
            "FacialExpressions": {
                "DominantBasicEmotion": dominant,
                "BasicEmotions": emotions,
                "Valence": valence,
                "Arousal": arousal,
            },
            "Characteristics": characteristics,
            "ActionUnits": action_units,
            "Confidence": float(face.detection_confidence),
            "HeadOrientation": [
                float(geometry.get("head_pitch", 0)),
                float(geometry.get("head_yaw",   0)),
                float(geometry.get("head_roll",  0)),
            ],
            "BoundingBox": [
                int(face.face_box.x1),
                int(face.face_box.y1),
                int(face.face_box.width),
                int(face.face_box.height),
            ],
            "NumberOfFaces": len(results),
            "Focus": float(face.alignment_confidence * 100),
            "ActionableEmotion": dominant if emotions[dominant] > 0.3 else None,
            "RotationApplied": rotation,
        }

    def _analyze_image(self, image: np.ndarray, try_rotations: bool = True) -> Dict[str, Any]:
        """Core analysis logic."""
        if try_rotations:
            results, image, rotation = self._try_rotations(image)
        else:
            results  = self.pipeline.process(image, max_faces=10)
            rotation = 0

        if not results:
            return self._empty_response()

        face = results[0]

        if face.geometry is None:
            return self._empty_response(num_faces=len(results))

        geometry = face.geometry.to_dict()
        au_probs  = self._predict_aus(geometry)
        emotions  = self._compute_emotions(au_probs)

        # Pass the (possibly rotated) image so the attribute predictor
        # receives the same orientation as the detected face.
        return self._build_response(image, face, results, geometry, au_probs, emotions, rotation)

    def analyze(self, image_path: str, try_rotations: bool = True) -> Dict[str, Any]:
        """Analyze image from file path."""
        try:
            image = load_image_with_exif_fix(image_path)
        except Exception as e:
            return self._empty_response(error=str(e))
        return self._analyze_image(image, try_rotations)

    def analyze_bytes(self, image_bytes: bytes, try_rotations: bool = True) -> Dict[str, Any]:
        """Analyze image from bytes (handles EXIF rotation)."""
        try:
            image = load_image_with_exif_fix(image_bytes)
        except Exception as e:
            return self._empty_response(error=str(e))
        return self._analyze_image(image, try_rotations)


def get_analyzer() -> FaceEmotionAnalyzer:
    """Get singleton analyzer instance."""
    return FaceEmotionAnalyzer()