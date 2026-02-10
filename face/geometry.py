# -*- coding: utf-8 -*-
"""Geometry feature extraction from facial landmarks."""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

# Check MediaPipe availability
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


# =============================================================================
# GEOMETRY FEATURES DATACLASS
# =============================================================================

@dataclass
class GeometryFeatures:
    """Container for geometric features extracted from facial landmarks."""
    
    # Reference measurement
    inter_ocular_distance: float = 0.0
    
    # === EYE FEATURES ===
    left_eye_aspect_ratio: float = 0.0
    right_eye_aspect_ratio: float = 0.0
    eye_aspect_ratio: float = 0.0
    left_eye_height: float = 0.0
    right_eye_height: float = 0.0
    
    # === EYEBROW FEATURES ===
    left_eyebrow_height: float = 0.0
    right_eyebrow_height: float = 0.0
    left_inner_brow_height: float = 0.0
    right_inner_brow_height: float = 0.0
    left_outer_brow_height: float = 0.0
    right_outer_brow_height: float = 0.0
    eyebrow_distance: float = 0.0
    left_eyebrow_angle: float = 0.0
    right_eyebrow_angle: float = 0.0
    
    # === MOUTH FEATURES ===
    mouth_aspect_ratio: float = 0.0
    mouth_width: float = 0.0
    mouth_height: float = 0.0
    upper_lip_height: float = 0.0
    lower_lip_height: float = 0.0
    lip_corner_left_angle: float = 0.0
    lip_corner_right_angle: float = 0.0
    mouth_corner_distance: float = 0.0
    upper_lip_raise: float = 0.0
    lip_corner_pull: float = 0.0
    lip_stretch: float = 0.0
    jaw_drop: float = 0.0
    
    # === NOSE FEATURES ===
    nose_tip_height: float = 0.0
    nose_width: float = 0.0
    nose_wrinkle_indicator: float = 0.0
    
    # === FACE SHAPE FEATURES ===
    jaw_width: float = 0.0
    face_height: float = 0.0
    face_aspect_ratio: float = 0.0
    chin_height: float = 0.0
    
    # === SYMMETRY FEATURES ===
    horizontal_asymmetry: float = 0.0
    eye_asymmetry: float = 0.0
    eyebrow_asymmetry: float = 0.0
    mouth_asymmetry: float = 0.0
    
    # === HEAD POSE INDICATORS ===
    head_roll: float = 0.0
    head_pitch_indicator: float = 0.0
    head_yaw_indicator: float = 0.0
    
    # === RAW DATA ===
    landmarks_468: Optional[np.ndarray] = None
    landmarks_68: Optional[np.ndarray] = None
    
    def to_vector(self) -> np.ndarray:
        """Convert features to a flat vector for ML models."""
        return np.array([
            # Eye features (5)
            self.left_eye_aspect_ratio, self.right_eye_aspect_ratio, self.eye_aspect_ratio,
            self.left_eye_height, self.right_eye_height,
            # Eyebrow features (9)
            self.left_eyebrow_height, self.right_eyebrow_height,
            self.left_inner_brow_height, self.right_inner_brow_height,
            self.left_outer_brow_height, self.right_outer_brow_height,
            self.eyebrow_distance, self.left_eyebrow_angle, self.right_eyebrow_angle,
            # Mouth features (12)
            self.mouth_aspect_ratio, self.mouth_width, self.mouth_height,
            self.upper_lip_height, self.lower_lip_height,
            self.lip_corner_left_angle, self.lip_corner_right_angle,
            self.mouth_corner_distance, self.upper_lip_raise,
            self.lip_corner_pull, self.lip_stretch, self.jaw_drop,
            # Nose features (3)
            self.nose_tip_height, self.nose_width, self.nose_wrinkle_indicator,
            # Face shape (4)
            self.jaw_width, self.face_height, self.face_aspect_ratio, self.chin_height,
            # Symmetry (4)
            self.horizontal_asymmetry, self.eye_asymmetry, 
            self.eyebrow_asymmetry, self.mouth_asymmetry,
            # Pose (3)
            self.head_roll, self.head_pitch_indicator, self.head_yaw_indicator,
        ], dtype=np.float32)
    
    @staticmethod
    def feature_names() -> List[str]:
        """Return names of all features in vector order."""
        return [
            'left_ear', 'right_ear', 'avg_ear', 'left_eye_h', 'right_eye_h',
            'left_brow_h', 'right_brow_h', 'left_inner_brow_h', 'right_inner_brow_h',
            'left_outer_brow_h', 'right_outer_brow_h', 'brow_dist', 
            'left_brow_angle', 'right_brow_angle',
            'mar', 'mouth_w', 'mouth_h', 'upper_lip_h', 'lower_lip_h',
            'lip_angle_l', 'lip_angle_r', 'mouth_corner_dist',
            'upper_lip_raise', 'lip_corner_pull', 'lip_stretch', 'jaw_drop',
            'nose_tip_h', 'nose_w', 'nose_wrinkle',
            'jaw_w', 'face_h', 'face_ar', 'chin_h',
            'h_asymmetry', 'eye_asymmetry', 'brow_asymmetry', 'mouth_asymmetry',
            'head_roll', 'head_pitch', 'head_yaw',
        ]
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        names = self.feature_names()
        values = self.to_vector()
        return dict(zip(names, values))
    
    @property 
    def num_features(self) -> int:
        return len(self.to_vector())


# =============================================================================
# LANDMARK DETECTOR
# =============================================================================

class LandmarkDetector:
    """Detect 468 facial landmarks using MediaPipe Face Mesh."""
    
    MEDIAPIPE_TO_68 = {
        0: 234, 1: 93, 2: 132, 3: 58, 4: 172, 5: 136, 6: 150, 7: 176, 8: 152,
        9: 400, 10: 379, 11: 365, 12: 397, 13: 288, 14: 361, 15: 323, 16: 454,
        17: 70, 18: 63, 19: 105, 20: 66, 21: 107,
        22: 336, 23: 296, 24: 334, 25: 293, 26: 300,
        27: 168, 28: 6, 29: 197, 30: 195,
        31: 98, 32: 97, 33: 2, 34: 326, 35: 327,
        36: 33, 37: 160, 38: 158, 39: 133, 40: 153, 41: 144,
        42: 362, 43: 385, 44: 387, 45: 263, 46: 373, 47: 380,
        48: 61, 49: 40, 50: 37, 51: 0, 52: 267, 53: 270, 54: 291,
        55: 321, 56: 314, 57: 17, 58: 84, 59: 91,
        60: 78, 61: 81, 62: 13, 63: 311, 64: 308, 65: 402, 66: 14, 67: 178,
    }
    
    KEY_POINTS = {
        'left_eye_outer': 263, 'left_eye_inner': 362,
        'left_eye_top': 386, 'left_eye_bottom': 374,
        'right_eye_outer': 33, 'right_eye_inner': 133,
        'right_eye_top': 159, 'right_eye_bottom': 145,
        'left_eyebrow_inner': 336, 'left_eyebrow_outer': 300, 'left_eyebrow_top': 293,
        'right_eyebrow_inner': 70, 'right_eyebrow_outer': 107, 'right_eyebrow_top': 66,
        'nose_tip': 1, 'nose_bridge': 6, 'left_nostril': 129, 'right_nostril': 358,
        'mouth_left': 61, 'mouth_right': 291, 'mouth_top': 13, 'mouth_bottom': 14,
        'upper_lip_top': 0, 'lower_lip_bottom': 17,
        'chin': 152, 'left_cheek': 234, 'right_cheek': 454, 'forehead': 10,
    }
    
    def __init__(self, static_mode: bool = True, max_faces: int = 1, 
                 refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not available. Install with: pip install mediapipe")
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.refine_landmarks = refine_landmarks
    
    def detect(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Detect facial landmarks in image."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return None, 0.0
        
        face_landmarks = results.multi_face_landmarks[0]
        
        landmarks = np.array([
            [lm.x * w, lm.y * h, lm.z * w]
            for lm in face_landmarks.landmark
        ], dtype=np.float32)
        
        confidence = self._estimate_confidence(landmarks, w, h)
        
        return landmarks, confidence
    
    def detect_68(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Detect and return 68 landmarks (dlib-compatible format)."""
        landmarks_full, confidence = self.detect(image)
        
        if landmarks_full is None:
            return None, 0.0
        
        landmarks_68 = np.array([
            landmarks_full[self.MEDIAPIPE_TO_68[i], :2] 
            for i in range(68)
        ], dtype=np.float32)
        
        return landmarks_68, confidence
    
    def _estimate_confidence(self, landmarks: np.ndarray, w: int, h: int) -> float:
        """Estimate detection confidence based on landmark positions."""
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        in_bounds = np.mean(
            (x_coords >= 0) & (x_coords < w) & 
            (y_coords >= 0) & (y_coords < h)
        )
        
        face_width = x_coords.max() - x_coords.min()
        face_height = y_coords.max() - y_coords.min()
        aspect_ratio = face_height / (face_width + 1e-6)
        
        aspect_score = 1.0 - min(abs(aspect_ratio - 1.25) / 0.5, 1.0)
        confidence = (in_bounds * 0.6 + aspect_score * 0.4)
        
        return float(np.clip(confidence, 0, 1))
    
    def get_key_points(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract key landmark points by name."""
        return {
            name: landmarks[idx, :2] 
            for name, idx in self.KEY_POINTS.items()
        }


# =============================================================================
# GEOMETRY EXTRACTOR
# =============================================================================

class GeometryExtractor:
    """Extract geometric features from facial landmarks."""

    REGIONS_68 = {
        'jaw': list(range(0, 17)),
        'right_eyebrow': list(range(17, 22)),
        'left_eyebrow': list(range(22, 27)),
        'nose_bridge': list(range(27, 31)),
        'nose_bottom': list(range(31, 36)),
        'right_eye': list(range(36, 42)),
        'left_eye': list(range(42, 48)),
        'outer_lip': list(range(48, 60)),
        'inner_lip': list(range(60, 68)),
    }

    POINTS_68 = {
        'right_eye_outer': 36, 'right_eye_inner': 39,
        'left_eye_outer': 45, 'left_eye_inner': 42,
        'right_brow_inner': 21, 'right_brow_outer': 17,
        'left_brow_inner': 22, 'left_brow_outer': 26,
        'nose_tip': 30, 'nose_bridge_top': 27,
        'left_nostril': 35, 'right_nostril': 31,
        'mouth_left': 48, 'mouth_right': 54,
        'mouth_top_outer': 51, 'mouth_bottom_outer': 57,
        'mouth_top_inner': 62, 'mouth_bottom_inner': 66,
        'chin': 8, 'left_jaw': 0, 'right_jaw': 16,
    }

    def __init__(self, baseline_features: Optional[GeometryFeatures] = None):
        self.baseline = baseline_features

    def set_baseline(self, features: GeometryFeatures):
        self.baseline = features

    def extract(self, landmarks: np.ndarray) -> GeometryFeatures:
        """Extract geometric features from landmarks."""
        n_points = landmarks.shape[0]

        if n_points in (468, 478):
            landmarks_68 = self._convert_468_to_68(landmarks)
            landmarks_468 = landmarks[:, :2] if landmarks.shape[1] > 2 else landmarks
        elif n_points == 68:
            landmarks_68 = landmarks[:, :2] if landmarks.shape[1] > 2 else landmarks
            landmarks_468 = None
        else:
            raise ValueError(f"Expected 68 or 468/478 landmarks, got {n_points}")

        features = self._extract_from_68(landmarks_68)
        features.landmarks_68 = landmarks_68
        features.landmarks_468 = landmarks_468
        return features

    def _convert_468_to_68(self, landmarks: np.ndarray) -> np.ndarray:
        return np.array(
            [landmarks[LandmarkDetector.MEDIAPIPE_TO_68[i], :2] for i in range(68)],
            dtype=np.float32
        )

    def _extract_from_68(self, lm: np.ndarray) -> GeometryFeatures:
        # === REFERENCE ===
        left_eye_center = lm[self.REGIONS_68['left_eye']].mean(axis=0)
        right_eye_center = lm[self.REGIONS_68['right_eye']].mean(axis=0)
        inter_ocular = float(np.linalg.norm(left_eye_center - right_eye_center))
        if inter_ocular < 1.0:
            inter_ocular = 1.0

        def norm_dist(p1, p2) -> float:
            return float(np.linalg.norm(np.array(p1) - np.array(p2)) / inter_ocular)

        # === EYE FEATURES ===
        left_ear = self._compute_ear(lm, self.REGIONS_68['left_eye'])
        right_ear = self._compute_ear(lm, self.REGIONS_68['right_eye'])

        left_eye = lm[self.REGIONS_68['left_eye']]
        right_eye = lm[self.REGIONS_68['right_eye']]
        left_eye_height = float((left_eye[1:3, 1].mean() - left_eye[4:6, 1].mean()) / inter_ocular)
        right_eye_height = float((right_eye[1:3, 1].mean() - right_eye[4:6, 1].mean()) / inter_ocular)

        # === EYEBROW FEATURES ===
        left_brow = lm[self.REGIONS_68['left_eyebrow']]
        right_brow = lm[self.REGIONS_68['right_eyebrow']]

        left_brow_height = float((left_eye_center[1] - left_brow.mean(axis=0)[1]) / inter_ocular)
        right_brow_height = float((right_eye_center[1] - right_brow.mean(axis=0)[1]) / inter_ocular)

        left_inner_brow_h = float((left_eye_center[1] - lm[22][1]) / inter_ocular)
        right_inner_brow_h = float((right_eye_center[1] - lm[21][1]) / inter_ocular)

        left_outer_brow_h = float((left_eye_center[1] - lm[26][1]) / inter_ocular)
        right_outer_brow_h = float((right_eye_center[1] - lm[17][1]) / inter_ocular)

        eyebrow_distance = norm_dist(lm[21], lm[22])

        left_brow_angle = float(np.degrees(np.arctan2(lm[26][1] - lm[22][1], lm[26][0] - lm[22][0])))
        right_brow_angle = float(np.degrees(np.arctan2(lm[17][1] - lm[21][1], lm[17][0] - lm[21][0])))

        # === MOUTH FEATURES ===
        p = self.POINTS_68
        mouth_width = norm_dist(lm[p['mouth_left']], lm[p['mouth_right']])
        mouth_height = norm_dist(lm[p['mouth_top_inner']], lm[p['mouth_bottom_inner']])
        mar = float(mouth_height / (mouth_width + 1e-6))

        upper_lip_h = norm_dist(lm[p['mouth_top_outer']], lm[p['mouth_top_inner']])
        lower_lip_h = norm_dist(lm[p['mouth_bottom_outer']], lm[p['mouth_bottom_inner']])

        mouth_center_y = float((lm[p['mouth_top_outer']][1] + lm[p['mouth_bottom_outer']][1]) / 2)

        lip_left_angle = float(np.degrees(np.arctan2(
            lm[p['mouth_left']][1] - mouth_center_y,
            lm[p['mouth_left']][0] - lm[p['mouth_top_outer']][0]
        )))
        lip_right_angle = float(np.degrees(np.arctan2(
            lm[p['mouth_right']][1] - mouth_center_y,
            lm[p['mouth_right']][0] - lm[p['mouth_top_outer']][0]
        )))

        nose_tip = lm[p['nose_tip']]
        left_corner_dist = norm_dist(lm[p['mouth_left']], nose_tip)
        right_corner_dist = norm_dist(lm[p['mouth_right']], nose_tip)
        mouth_corner_distance = float((left_corner_dist + right_corner_dist) / 2)

        upper_lip_raise = float((lm[51][1] - lm[62][1]) / inter_ocular)
        lip_corner_pull = float((mouth_center_y - (lm[48][1] + lm[54][1]) / 2) / inter_ocular)
        lip_stretch = float(mouth_width)
        jaw_drop = float(mouth_height)

        # === NOSE FEATURES ===
        nose_tip_height = float((nose_tip[1] - left_eye_center[1]) / inter_ocular)
        nose_width = norm_dist(lm[p['left_nostril']], lm[p['right_nostril']])

        nose_bridge = lm[p['nose_bridge_top']]
        between_eyes = (left_eye_center + right_eye_center) / 2
        nose_wrinkle = norm_dist(nose_bridge, between_eyes)

        # === FACE SHAPE ===
        jaw_width = norm_dist(lm[p['left_jaw']], lm[p['right_jaw']])
        chin = lm[p['chin']]
        face_height = norm_dist(chin, nose_bridge)
        face_ar = float(face_height / (jaw_width + 1e-6))
        chin_height = float((chin[1] - lm[p['mouth_bottom_outer']][1]) / inter_ocular)

        # === SYMMETRY ===
        center_x = float((lm[p['left_jaw']][0] + lm[p['right_jaw']][0]) / 2)

        left_indices = [0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 19, 20, 21, 36, 37, 38, 39, 40, 41]
        right_indices = [16, 15, 14, 13, 12, 11, 10, 9, 26, 25, 24, 23, 22, 45, 44, 43, 42, 47, 46]

        left_pts = lm[left_indices]
        right_pts = lm[right_indices].copy()
        right_pts[:, 0] = 2 * center_x - right_pts[:, 0]

        h_asymmetry = float(np.mean(np.linalg.norm(left_pts - right_pts, axis=1)) / inter_ocular)

        eye_asymmetry = float(abs(left_ear - right_ear))
        eyebrow_asymmetry = float(abs(left_brow_height - right_brow_height))
        mouth_asymmetry = float(abs(lm[p['mouth_left']][1] - lm[p['mouth_right']][1]) / inter_ocular)

        # === POSE ===
        head_roll = float(np.degrees(np.arctan2(
            right_eye_center[1] - left_eye_center[1],
            right_eye_center[0] - left_eye_center[0]
        )))
        face_center_y = float((left_eye_center[1] + chin[1]) / 2)
        head_pitch = float((nose_tip[1] - face_center_y) / inter_ocular)

        left_face_width = abs(float(nose_tip[0] - lm[p['left_jaw']][0]))
        right_face_width = abs(float(lm[p['right_jaw']][0] - nose_tip[0]))
        head_yaw = float((left_face_width - right_face_width) / inter_ocular)

        return GeometryFeatures(
            inter_ocular_distance=inter_ocular,
            left_eye_aspect_ratio=left_ear,
            right_eye_aspect_ratio=right_ear,
            eye_aspect_ratio=(left_ear + right_ear) / 2,
            left_eye_height=left_eye_height,
            right_eye_height=right_eye_height,
            left_eyebrow_height=left_brow_height,
            right_eyebrow_height=right_brow_height,
            left_inner_brow_height=left_inner_brow_h,
            right_inner_brow_height=right_inner_brow_h,
            left_outer_brow_height=left_outer_brow_h,
            right_outer_brow_height=right_outer_brow_h,
            eyebrow_distance=eyebrow_distance,
            left_eyebrow_angle=left_brow_angle,
            right_eyebrow_angle=right_brow_angle,
            mouth_aspect_ratio=mar,
            mouth_width=mouth_width,
            mouth_height=mouth_height,
            upper_lip_height=upper_lip_h,
            lower_lip_height=lower_lip_h,
            lip_corner_left_angle=lip_left_angle,
            lip_corner_right_angle=lip_right_angle,
            mouth_corner_distance=mouth_corner_distance,
            upper_lip_raise=upper_lip_raise,
            lip_corner_pull=lip_corner_pull,
            lip_stretch=lip_stretch,
            jaw_drop=jaw_drop,
            nose_tip_height=nose_tip_height,
            nose_width=nose_width,
            nose_wrinkle_indicator=nose_wrinkle,
            jaw_width=jaw_width,
            face_height=face_height,
            face_aspect_ratio=face_ar,
            chin_height=chin_height,
            horizontal_asymmetry=h_asymmetry,
            eye_asymmetry=eye_asymmetry,
            eyebrow_asymmetry=eyebrow_asymmetry,
            mouth_asymmetry=mouth_asymmetry,
            head_roll=head_roll,
            head_pitch_indicator=head_pitch,
            head_yaw_indicator=head_yaw,
        )

    def _compute_ear(self, landmarks: np.ndarray, eye_indices: List[int]) -> float:
        """Compute Eye Aspect Ratio (EAR)."""
        eye = landmarks[eye_indices]
        v1 = np.linalg.norm(eye[1] - eye[5])
        v2 = np.linalg.norm(eye[2] - eye[4])
        h = np.linalg.norm(eye[0] - eye[3])
        if h < 1e-6:
            return 0.0
        return float((v1 + v2) / (2.0 * h))

    def compute_relative_features(self, current: GeometryFeatures) -> Optional[np.ndarray]:
        if self.baseline is None:
            return None
        return current.to_vector() - self.baseline.to_vector()