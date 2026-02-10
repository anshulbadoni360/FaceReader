# -*- coding: utf-8 -*-
"""Face alignment using similarity transformation."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

from .detector import FaceBox


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AlignedFace:
    """Represents an aligned and cropped face."""
    image: np.ndarray
    original_box: FaceBox
    transform_matrix: np.ndarray
    landmarks_5: Optional[np.ndarray] = None
    alignment_confidence: float = 1.0

    @property
    def size(self) -> Tuple[int, int]:
        return self.image.shape[:2]


# =============================================================================
# FACE ALIGNER
# =============================================================================

class FaceAligner:
    """Align faces using similarity transformation."""

    TEMPLATE_112 = np.array([
        [38.2946, 51.6963], 
        [73.5318, 51.5014], 
        [56.0252, 71.7366],
        [41.5493, 92.3655], 
        [70.7299, 92.2041]
    ], dtype=np.float32)

    def __init__(self, output_size: int = 224):
        self.output_size = output_size
        scale = output_size / 112.0
        self.template = self.TEMPLATE_112 * scale * 0.8 + output_size * 0.1

    def align(self, image: np.ndarray, face_box: FaceBox, 
              skip_alignment: bool = False) -> Optional[AlignedFace]:
        """
        Align face using landmarks.
        
        Args:
            image: Input image
            face_box: Detected face with landmarks
            skip_alignment: If True, just crop without rotation (for debugging)
        """
        if face_box.landmarks_5 is None or skip_alignment:
            return self._simple_crop(image, face_box)

        src_pts = face_box.landmarks_5.copy()

        try:
            M, _ = cv2.estimateAffinePartial2D(
                src_pts, self.template, method=cv2.LMEDS, confidence=0.99
            )
            if M is None:
                M = cv2.getAffineTransform(src_pts[:3], self.template[:3])
        except:
            return self._simple_crop(image, face_box)

        aligned = cv2.warpAffine(
            image, M, (self.output_size, self.output_size), 
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(0, 0, 0)
        )
        
        aligned_lm = (np.hstack([src_pts, np.ones((5, 1))]) @ M.T)
        conf = np.exp(-5 * np.mean(
            np.linalg.norm(aligned_lm - self.template, axis=1) / self.output_size
        ))

        return AlignedFace(aligned, face_box, M, aligned_lm, float(np.clip(conf, 0, 1)))

    def _simple_crop(self, image: np.ndarray, face_box: FaceBox) -> AlignedFace:
        h, w = image.shape[:2]
        box = face_box.to_square((h, w)).expand(1.3, (h, w))
        crop = image[box.y1:box.y2, box.x1:box.x2]
        
        if crop.size > 0:
            aligned = cv2.resize(crop, (self.output_size, self.output_size))
        else:
            aligned = np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)
        
        return AlignedFace(aligned, face_box, np.eye(2, 3, dtype=np.float32), None, 0.5)