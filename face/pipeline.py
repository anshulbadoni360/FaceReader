# -*- coding: utf-8 -*-
"""Face processing pipeline - NO print statements."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from PIL import Image, ExifTags

from .detector import FaceDetector, FaceBox, YUNET_PATH
from .aligner import FaceAligner
from .geometry import (
    GeometryFeatures, 
    LandmarkDetector, 
    GeometryExtractor, 
    MEDIAPIPE_AVAILABLE
)


# =============================================================================
# IMAGE LOADING WITH EXIF HANDLING
# =============================================================================

def load_image_with_exif_fix(path: str) -> np.ndarray:
    """Load image and correctly handle EXIF orientation."""
    img = Image.open(path)
    
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
    
    return np.array(img)


def read_image(path: str) -> np.ndarray:
    """Read image with EXIF fix, return BGR for OpenCV."""
    img = load_image_with_exif_fix(path)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ProcessedFace:
    """Complete processed face with all extracted information."""
    face_box: FaceBox
    detection_confidence: float
    aligned_image: np.ndarray
    alignment_confidence: float
    transform_matrix: np.ndarray
    landmarks_68: Optional[np.ndarray] = None
    landmarks_468: Optional[np.ndarray] = None
    landmark_confidence: float = 0.0
    geometry: Optional[GeometryFeatures] = None
    
    @property
    def is_valid(self) -> bool:
        """Check if all processing stages succeeded."""
        return (self.detection_confidence > 0.5 and 
                self.alignment_confidence > 0.5 and
                self.landmark_confidence > 0.5 and
                self.geometry is not None)


# =============================================================================
# FACE PIPELINE
# =============================================================================

class FacePipeline:
    """
    Complete face processing pipeline.
    
    Stages:
    1. Face Detection (YuNet)
    2. Face Alignment & Crop
    3. Facial Landmarks (MediaPipe)
    4. Geometry Feature Extraction
    """
    
    def __init__(self, 
                 detection_threshold: float = 0.7,
                 alignment_size: int = 224,
                 static_mode: bool = True):
        self.detector = FaceDetector(YUNET_PATH, confidence_threshold=detection_threshold)
        self.aligner = FaceAligner(output_size=alignment_size)
        
        if MEDIAPIPE_AVAILABLE:
            self.landmark_detector = LandmarkDetector(static_mode=static_mode)
        else:
            self.landmark_detector = None
        
        self.geometry_extractor = GeometryExtractor()
    
    def process(self, image: np.ndarray, max_faces: int = 5) -> List[ProcessedFace]:
        """
        Process image through complete pipeline.
        
        Args:
            image: BGR image
            max_faces: Maximum number of faces to process
            
        Returns:
            List of ProcessedFace objects
        """
        results = []
        
        faces = self.detector.detect(image)
        
        for face_box in faces[:max_faces]:
            aligned = self.aligner.align(image, face_box)
            if aligned is None:
                continue
            
            landmarks_68 = None
            landmarks_468 = None
            landmark_conf = 0.0
            geometry = None
            
            if self.landmark_detector is not None:
                landmarks_468, landmark_conf = self.landmark_detector.detect(aligned.image)
                
                if landmarks_468 is not None:
                    landmarks_68, _ = self.landmark_detector.detect_68(aligned.image)
                    
                    if landmarks_68 is not None:
                        geometry = self.geometry_extractor.extract(landmarks_68)
            
            processed = ProcessedFace(
                face_box=face_box,
                detection_confidence=face_box.confidence,
                aligned_image=aligned.image,
                alignment_confidence=aligned.alignment_confidence,
                transform_matrix=aligned.transform_matrix,
                landmarks_68=landmarks_68,
                landmarks_468=landmarks_468,
                landmark_confidence=landmark_conf,
                geometry=geometry
            )
            
            results.append(processed)
        
        return results


# =============================================================================
# QUICK ANALYSIS FUNCTION
# =============================================================================

# Global pipeline instance (lazy initialization)
_pipeline: Optional[FacePipeline] = None


def _get_pipeline() -> FacePipeline:
    """Get or create global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = FacePipeline()
    return _pipeline


def analyze_face(image_path: str, visualize: bool = False) -> Optional[ProcessedFace]:
    """
    Quick function to analyze a single face.
    
    Args:
        image_path: Path to image file
        visualize: If True, display results (requires debug_demo module)
        
    Returns:
        ProcessedFace object with all extracted features, or None if no face
    """
    if not MEDIAPIPE_AVAILABLE:
        return None
    
    pipeline = _get_pipeline()
    image = read_image(image_path)
    results = pipeline.process(image, max_faces=1)
    
    if not results:
        return None
    
    result = results[0]
    
    if visualize:
        try:
            from .debug_demo import visualize_pipeline_result, plot_geometry_features
            visualize_pipeline_result(image, result)
            if result.geometry:
                plot_geometry_features(result.geometry, f"Geometry: {Path(image_path).name}")
        except ImportError:
            pass
    
    return result