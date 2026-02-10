"""Visualization, debugging, and demo code for face processing."""

import sys
from pathlib import Path

# Fix imports when running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
import urllib.request
from typing import List, Optional

from face.detector import FaceDetector, FaceBox, YUNET_PATH
from face.aligner import FaceAligner, AlignedFace
from face.geometry import GeometryFeatures, GeometryExtractor, MEDIAPIPE_AVAILABLE
from face.pipeline import FacePipeline, ProcessedFace, read_image



# =============================================================================
# PRINT SYSTEM INFO
# =============================================================================

def print_system_info():
    """Print Python and library versions."""
    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"MediaPipe available: {MEDIAPIPE_AVAILABLE}")


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def visualize_detection(image: np.ndarray, faces: List[FaceBox]) -> np.ndarray:
    """Draw detection boxes and landmarks on image."""
    vis = image.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for i, face in enumerate(faces):
        color = colors[i % len(colors)]
        cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), color, 2)
        cv2.putText(vis, f"{face.confidence:.2f}", (face.x1, face.y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if face.landmarks_5 is not None:
            for x, y in face.landmarks_5:
                cv2.circle(vis, (int(x), int(y)), 3, color, -1)
    
    return vis


def visualize_alignment(original: np.ndarray, face_box: FaceBox, aligned_face: AlignedFace):
    """Visualize original detection and aligned face."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not found. Skipping visualization.")
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    vis_orig = visualize_detection(original, [face_box])
    axes[0].imshow(cv2.cvtColor(vis_orig, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Detection (conf: {face_box.confidence:.2f})")
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(aligned_face.image, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Aligned (conf: {aligned_face.alignment_confidence:.2f})")
    axes[1].axis('off')
    
    vis_aligned = aligned_face.image.copy()
    if aligned_face.landmarks_5 is not None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        labels = ["R_Eye", "L_Eye", "Nose", "R_Mouth", "L_Mouth"]
        for (x, y), color, label in zip(aligned_face.landmarks_5, colors, labels):
            cv2.circle(vis_aligned, (int(x), int(y)), 4, color, -1)
            cv2.putText(vis_aligned, label, (int(x)+5, int(y)-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    axes[2].imshow(cv2.cvtColor(vis_aligned, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Aligned with Landmarks")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def display_faces_grid(aligned_faces: List[AlignedFace], cols: int = 5):
    """Display aligned faces in a grid."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not found. Skipping visualization.")
        return
    n = len(aligned_faces)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = np.array(axes).flatten()
    
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(cv2.cvtColor(aligned_faces[i].image, cv2.COLOR_BGR2RGB))
            ax.set_title(f"Conf: {aligned_faces[i].alignment_confidence:.2f}", fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_landmarks_68(image: np.ndarray, landmarks: np.ndarray, 
                           show_indices: bool = False) -> np.ndarray:
    """Draw 68 landmarks on image with region colors."""
    vis = image.copy()
    
    region_colors = {
        'jaw': (255, 200, 0),
        'right_eyebrow': (0, 255, 0),
        'left_eyebrow': (0, 255, 0),
        'nose_bridge': (255, 0, 255),
        'nose_bottom': (255, 0, 255),
        'right_eye': (0, 0, 255),
        'left_eye': (0, 0, 255),
        'outer_lip': (0, 165, 255),
        'inner_lip': (0, 100, 255),
    }
    
    for region, indices in GeometryExtractor.REGIONS_68.items():
        color = region_colors.get(region, (255, 255, 255))
        pts = landmarks[indices].astype(int)
        
        for i, (x, y) in enumerate(pts):
            cv2.circle(vis, (x, y), 2, color, -1)
            if show_indices:
                cv2.putText(vis, str(indices[i]), (x+3, y-3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
        
        if region != 'jaw':
            for i in range(len(pts) - 1):
                cv2.line(vis, tuple(pts[i]), tuple(pts[i+1]), color, 1)
            if region in ['right_eye', 'left_eye', 'outer_lip', 'inner_lip']:
                cv2.line(vis, tuple(pts[-1]), tuple(pts[0]), color, 1)
        else:
            for i in range(len(pts) - 1):
                cv2.line(vis, tuple(pts[i]), tuple(pts[i+1]), color, 1)
    
    return vis


def visualize_geometry_features(image: np.ndarray, landmarks: np.ndarray, 
                                features: GeometryFeatures) -> np.ndarray:
    """Visualize key geometric measurements on face."""
    vis = visualize_landmarks_68(image, landmarks)
    
    p = GeometryExtractor.POINTS_68
    
    left_eye = landmarks[GeometryExtractor.REGIONS_68['left_eye']].mean(axis=0).astype(int)
    right_eye = landmarks[GeometryExtractor.REGIONS_68['right_eye']].mean(axis=0).astype(int)
    
    cv2.line(vis, tuple(left_eye), tuple(right_eye), (255, 255, 0), 2)
    mid = ((left_eye[0] + right_eye[0])//2, (left_eye[1] + right_eye[1])//2 - 10)
    cv2.putText(vis, f"IOD: {features.inter_ocular_distance:.0f}px", mid,
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    mouth_l = landmarks[p['mouth_left']].astype(int)
    mouth_r = landmarks[p['mouth_right']].astype(int)
    mouth_t = landmarks[p['mouth_top_inner']].astype(int)
    mouth_b = landmarks[p['mouth_bottom_inner']].astype(int)
    
    cv2.line(vis, tuple(mouth_l), tuple(mouth_r), (0, 255, 255), 2)
    cv2.line(vis, tuple(mouth_t), tuple(mouth_b), (0, 255, 255), 2)
    
    return vis


def plot_geometry_features(features: GeometryFeatures, title: str = "Geometry Features"):
    """Create bar plot of geometry features."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not found. Skipping visualization.")
        return
    names = features.feature_names()
    values = features.to_vector()
    
    groups = {
        'Eye': slice(0, 5),
        'Eyebrow': slice(5, 14),
        'Mouth': slice(14, 26),
        'Nose': slice(26, 29),
        'Face': slice(29, 33),
        'Symmetry': slice(33, 37),
        'Pose': slice(37, 40),
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    
    for idx, (group_name, sl) in enumerate(groups.items()):
        ax = axes[idx]
        group_names = names[sl]
        group_values = values[sl]
        
        ax.barh(range(len(group_values)), group_values, color=colors[idx])
        ax.set_yticks(range(len(group_values)))
        ax.set_yticklabels(group_names, fontsize=8)
        ax.set_title(group_name)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    axes[-1].axis('off')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def compare_geometry(features_list: List[GeometryFeatures], labels: List[str]):
    """Compare geometry features across multiple faces."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not found. Skipping visualization.")
        return
    n_features = features_list[0].num_features
    names = features_list[0].feature_names()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(n_features)
    width = 0.8 / len(features_list)
    
    for i, (features, label) in enumerate(zip(features_list, labels)):
        offset = (i - len(features_list)/2 + 0.5) * width
        ax.bar(x + offset, features.to_vector(), width, label=label, alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.legend()
    ax.set_title("Geometry Feature Comparison")
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()


def visualize_pipeline_result(image: np.ndarray, result: ProcessedFace):
    """Visualize complete pipeline result for one face."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not found. Skipping visualization.")
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    vis_orig = visualize_detection(image, [result.face_box])
    axes[0, 0].imshow(cv2.cvtColor(vis_orig, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"Detection (conf: {result.detection_confidence:.2f})")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(result.aligned_image, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"Aligned (conf: {result.alignment_confidence:.2f})")
    axes[0, 1].axis('off')
    
    if result.landmarks_68 is not None:
        vis_lm = visualize_landmarks_68(result.aligned_image, result.landmarks_68)
        axes[0, 2].imshow(cv2.cvtColor(vis_lm, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title(f"68 Landmarks (conf: {result.landmark_confidence:.2f})")
    else:
        axes[0, 2].imshow(cv2.cvtColor(result.aligned_image, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title("Landmarks: N/A")
    axes[0, 2].axis('off')
    
    if result.geometry is not None and result.landmarks_68 is not None:
        vis_geom = visualize_geometry_features(result.aligned_image, 
                                               result.landmarks_68, 
                                               result.geometry)
        axes[1, 0].imshow(cv2.cvtColor(vis_geom, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("Geometry Measurements")
    else:
        axes[1, 0].axis('off')
    
    if result.geometry is not None:
        g = result.geometry
        text = f"""Key Geometry Features:
            
Eye Aspect Ratio (EAR): {g.eye_aspect_ratio:.3f}
  Left: {g.left_eye_aspect_ratio:.3f}  Right: {g.right_eye_aspect_ratio:.3f}

Mouth Aspect Ratio (MAR): {g.mouth_aspect_ratio:.3f}
  Width: {g.mouth_width:.3f}  Height: {g.mouth_height:.3f}

Eyebrow Height: L={g.left_eyebrow_height:.3f} R={g.right_eyebrow_height:.3f}
Eyebrow Distance: {g.eyebrow_distance:.3f}

Lip Corner Pull (smile): {g.lip_corner_pull:.3f}
Jaw Drop: {g.jaw_drop:.3f}

Head Roll: {g.head_roll:.1f}°
Asymmetry: {g.horizontal_asymmetry:.3f}
"""
        axes[1, 1].text(0.1, 0.9, text, transform=axes[1, 1].transAxes,
                      fontsize=9, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].set_title("Feature Values")
    axes[1, 1].axis('off')
    
    if result.geometry is not None:
        key_features = ['avg_ear', 'mar', 'left_brow_h', 'right_brow_h',
                       'lip_corner_pull', 'jaw_drop', 'eye_asymmetry', 'mouth_asymmetry']
        feat_dict = result.geometry.to_dict()
        values = [feat_dict[k] for k in key_features]
        
        colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in values]
        axes[1, 2].barh(key_features, values, color=colors)
        axes[1, 2].axvline(x=0, color='gray', linestyle='--')
        axes[1, 2].set_title("Key Features")
    else:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# DEBUG FUNCTIONS
# =============================================================================

def debug_image_orientation(image_path: str, detector: FaceDetector = None):
    """Debug tool to check image orientation at each step."""
    print(f"\n{'='*60}")
    print(f"DEBUG: {image_path}")
    print(f"{'='*60}")
    
    if detector is None:
        detector = FaceDetector(YUNET_PATH, confidence_threshold=0.7)
    
    img_raw = cv2.imread(image_path)
    print(f"Raw cv2.imread shape: {img_raw.shape}")
    
    img_fixed = read_image(image_path)
    print(f"After EXIF fix shape: {img_fixed.shape}")
    
    faces = detector.detect(img_fixed)
    print(f"Detected {len(faces)} faces")
    
    if faces:
        face = faces[0]
        print(f"Face landmarks:\n{face.landmarks_5}")
        
        if face.landmarks_5 is not None:
            left_eye = face.landmarks_5[1]
            right_eye = face.landmarks_5[0]
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            angle = np.degrees(np.arctan2(dy, dx))
            print(f"Eye angle: {angle:.1f}° (0° = horizontal eyes)")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Raw Load\n{img_raw.shape[1]}x{img_raw.shape[0]}")
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(img_fixed, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"EXIF Fixed\n{img_fixed.shape[1]}x{img_fixed.shape[0]}")
    axes[1].axis('off')
    
    if faces:
        vis = visualize_detection(img_fixed, faces)
        axes[2].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[2].set_title("With Detection")
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def compare_alignment_modes(image_path: str, detector: FaceDetector = None, 
                           aligner: FaceAligner = None):
    """Compare aligned vs simple crop."""
    if detector is None:
        detector = FaceDetector(YUNET_PATH, confidence_threshold=0.7)
    if aligner is None:
        aligner = FaceAligner(output_size=224)
    
    image = read_image(image_path)
    faces = detector.detect(image)
    
    if not faces:
        print("No faces detected")
        return
    
    face = faces[0]
    
    aligned_with_rotation = aligner.align(image, face, skip_alignment=False)
    aligned_no_rotation = aligner.align(image, face, skip_alignment=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    vis = visualize_detection(image, [face])
    axes[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Detection")
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(aligned_no_rotation.image, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Simple Crop (No Rotation)")
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(aligned_with_rotation.image, cv2.COLOR_BGR2RGB))
    axes[2].set_title("With Alignment (Rotated Upright)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# SAMPLE IMAGE DOWNLOAD
# =============================================================================

def download_sample_images() -> List[Path]:
    """Download sample images for testing."""
    sample_dir = Path("samples")
    sample_dir.mkdir(exist_ok=True)
    
    samples = [
        ("https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=600", "woman_smile.jpg"),
        ("https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=600", "man_neutral.jpg"),
        ("https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=600", "woman_serious.jpg"),
        ("https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?w=600", "group.jpg"),
    ]
    
    downloaded = []
    for url, filename in samples:
        filepath = sample_dir / filename
        if not filepath.exists():
            try:
                print(f"⬇ Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)
                print(f"  ✓ Saved")
                downloaded.append(filepath)
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        else:
            print(f"✓ {filename} already exists")
            downloaded.append(filepath)
    
    return downloaded


# =============================================================================
# MAIN DEMO
# =============================================================================

def run_full_demo():
    """Run complete demonstration of the face pipeline."""
    print_system_info()
    print("\n" + "="*60)
    print("FACE PROCESSING PIPELINE DEMO")
    print("="*60)
    
    # Download samples
    sample_images = download_sample_images()
    
    # Initialize components
    detector = FaceDetector(YUNET_PATH, confidence_threshold=0.7)
    aligner = FaceAligner(output_size=224)
    
    print("\n✓ Components initialized!")
    
    # Debug first image
    if sample_images:
        print("\n" + "="*60)
        print("DEBUGGING FIRST IMAGE")
        print("="*60)
        debug_image_orientation(str(sample_images[0]), detector)
        compare_alignment_modes(str(sample_images[0]), detector, aligner)
    
    # Main detection & alignment test
    print("\n" + "="*60)
    print("FACE DETECTION & ALIGNMENT TEST")
    print("="*60)
    
    aligned_faces = []
    
    for img_path in sample_images:
        print(f"\n📷 {img_path.name}")
        
        image = read_image(str(img_path))
        print(f"  Size: {image.shape[1]}x{image.shape[0]}")
        
        faces = detector.detect(image)
        print(f"  Detected {len(faces)} face(s) in {detector.last_inference_time_ms:.2f}ms")
        
        if not faces:
            continue
        
        vis = visualize_detection(image, faces)
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title(f"{img_path.name}: {len(faces)} faces")
        plt.axis('off')
        plt.show()
        
        for i, face in enumerate(faces[:3]):
            aligned = aligner.align(image, face, skip_alignment=False)
            
            if aligned:
                print(f"  Face {i+1}: conf={aligned.alignment_confidence:.3f}")
                aligned_faces.append(aligned)
                visualize_alignment(image, face, aligned)
    
    print(f"\n✓ Total aligned: {len(aligned_faces)}")
    
    if aligned_faces:
        display_faces_grid(aligned_faces, cols=4)
        
        output_dir = Path("aligned_faces")
        output_dir.mkdir(exist_ok=True)
        for i, face in enumerate(aligned_faces):
            cv2.imwrite(str(output_dir / f"aligned_{i:03d}.jpg"), face.image)
        print(f"✓ Saved {len(aligned_faces)} faces to aligned_faces/")
    
    # Test complete pipeline with geometry
    if MEDIAPIPE_AVAILABLE:
        print("\n" + "="*60)
        print("TESTING COMPLETE PIPELINE WITH GEOMETRY EXTRACTION")
        print("="*60)
        
        pipeline = FacePipeline(
            detection_threshold=0.7,
            alignment_size=224,
            static_mode=True
        )
        
        all_results = []
        
        for img_path in sample_images[:3]:
            print(f"\n📷 Processing: {img_path.name}")
            
            image = read_image(str(img_path))
            results = pipeline.process(image, max_faces=3)
            
            print(f"  Found {len(results)} face(s)")
            
            for i, result in enumerate(results):
                print(f"  Face {i+1}:")
                print(f"    Detection conf: {result.detection_confidence:.3f}")
                print(f"    Alignment conf: {result.alignment_confidence:.3f}")
                print(f"    Landmark conf:  {result.landmark_confidence:.3f}")
                
                if result.geometry:
                    g = result.geometry
                    print(f"    Geometry extracted: {g.num_features} features")
                    print(f"      EAR: {g.eye_aspect_ratio:.3f}")
                    print(f"      MAR: {g.mouth_aspect_ratio:.3f}")
                    print(f"      Smile indicator: {g.lip_corner_pull:.3f}")
                    print(f"      Head roll: {g.head_roll:.1f}°")
                
                visualize_pipeline_result(image, result)
                all_results.append(result)
        
        # Compare geometry
        if len(all_results) >= 2:
            print("\n" + "="*60)
            print("COMPARING GEOMETRY ACROSS FACES")
            print("="*60)
            
            geometries = [r.geometry for r in all_results if r.geometry is not None]
            labels = [f"Face {i+1}" for i in range(len(geometries))]
            
            if len(geometries) >= 2:
                compare_geometry(geometries[:4], labels[:4])
    
    print("\n✓ Demo complete!")
    
    print("""
Usage:
  from face import analyze_face, FacePipeline
  
  # Process single image
  result = analyze_face("path/to/image.jpg", visualize=True)
  
  # Access geometry features
  if result and result.geometry:
      g = result.geometry
      print(f"Eye Aspect Ratio: {g.eye_aspect_ratio}")   
      print(f"Mouth Aspect Ratio: {g.mouth_aspect_ratio}")
      print(f"Smile indicator: {g.lip_corner_pull}")
      
      # Get feature vector for ML
      features = g.to_vector()  # Shape: (40,)
      feature_names = g.feature_names()
      
  # Process multiple faces
  pipeline = FacePipeline()
  results = pipeline.process(image)
  for r in results:
      print(r.geometry.to_dict())
""")


if __name__ == "__main__":
    run_full_demo()