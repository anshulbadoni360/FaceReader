# -*- coding: utf-8 -*-
"""Face detection using YuNet."""

import cv2
import numpy as np
import time
import urllib.request
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path


# =============================================================================
# MODEL DOWNLOAD
# =============================================================================

def download_model(url: str, save_path: str, verbose: bool = False) -> str:
    """Download model file if not exists."""
    path = Path(save_path)
    if path.exists():
        return save_path

    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, save_path)
    return save_path


# Model paths
Path("models").mkdir(exist_ok=True)
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_PATH = download_model(YUNET_URL, "models/face_detection_yunet_2023mar.onnx")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FaceBox:
    """Represents a detected face bounding box."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    landmarks_5: Optional[np.ndarray] = None

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def expand(self, factor: float = 1.2, image_shape: Tuple[int, int] = None) -> 'FaceBox':
        cx, cy = self.center
        new_w = int(self.width * factor)
        new_h = int(self.height * factor)
        x1 = cx - new_w // 2
        y1 = cy - new_h // 2
        x2 = cx + new_w // 2
        y2 = cy + new_h // 2

        if image_shape:
            h, w = image_shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

        return FaceBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=self.confidence, landmarks_5=self.landmarks_5)

    def to_square(self, image_shape: Tuple[int, int] = None) -> 'FaceBox':
        cx, cy = self.center
        size = max(self.width, self.height)
        x1, y1 = cx - size // 2, cy - size // 2
        x2, y2 = cx + size // 2, cy + size // 2

        if image_shape:
            h, w = image_shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

        return FaceBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=self.confidence, landmarks_5=self.landmarks_5)


# =============================================================================
# FACE DETECTOR
# =============================================================================

class FaceDetector:
    """Face detector using YuNet."""

    def __init__(self, model_path: str = None,
                 input_size: Tuple[int, int] = (320, 320), 
                 confidence_threshold: float = 0.7,
                 nms_threshold: float = 0.3, 
                 top_k: int = 5):
        if model_path is None:
            model_path = YUNET_PATH
        self.detector = cv2.FaceDetectorYN.create(
            model_path, "", input_size, confidence_threshold, nms_threshold, top_k
        )
        self.confidence_threshold = confidence_threshold
        self.last_inference_time_ms = 0

    def detect(self, image: np.ndarray) -> List[FaceBox]:
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        
        start = time.perf_counter()
        _, faces = self.detector.detect(image)
        self.last_inference_time_ms = (time.perf_counter() - start) * 1000

        if faces is None:
            return []

        face_boxes = []
        for face in faces:
            x, y, width, height = face[:4].astype(int)
            box = FaceBox(
                x1=max(0, x), 
                y1=max(0, y), 
                x2=min(w, x + width), 
                y2=min(h, y + height),
                confidence=float(face[14]), 
                landmarks_5=face[4:14].reshape(5, 2)
            )
            face_boxes.append(box)

        return sorted(face_boxes, key=lambda x: x.confidence, reverse=True)