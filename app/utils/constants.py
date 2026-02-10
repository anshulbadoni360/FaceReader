"""Constants for AU detection and emotion mapping."""

AU_LIST = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU12",
    "AU14", "AU15", "AU17", "AU18", "AU20", "AU23", "AU24", "AU25", "AU26", 
    "AU27", "AU43"
]

FEATURE_NAMES = [
    'left_ear', 'right_ear', 'avg_ear', 'left_eye_h', 'right_eye_h',
    'left_brow_h', 'right_brow_h', 'left_inner_brow_h', 'right_inner_brow_h',
    'left_outer_brow_h', 'right_outer_brow_h', 'brow_dist', 'left_brow_angle', 
    'right_brow_angle', 'mar', 'mouth_w', 'mouth_h', 'upper_lip_h', 'lower_lip_h',
    'lip_angle_l', 'lip_angle_r', 'mouth_corner_dist', 'upper_lip_raise', 
    'lip_corner_pull', 'lip_stretch', 'jaw_drop', 'nose_tip_h', 'nose_w', 
    'nose_wrinkle', 'jaw_w', 'face_h', 'face_ar', 'chin_h', 'h_asymmetry', 
    'eye_asymmetry', 'brow_asymmetry', 'mouth_asymmetry', 'head_roll', 
    'head_pitch', 'head_yaw',
]

EMOTION_CONFIG = {
    "Happy": {
        "required": ["AU12"], 
        "enhancers": ["AU06", "AU25"], 
        "inhibitors": ["AU04", "AU15", "AU01"]
    },
    "Sad": {
        "required": ["AU01", "AU15"], 
        "enhancers": ["AU04", "AU17"], 
        "inhibitors": ["AU12", "AU06"]
    },
    "Angry": {
        "required": ["AU04", "AU07"], 
        "enhancers": ["AU05", "AU23", "AU24"], 
        "inhibitors": ["AU12", "AU01"]
    },
    "Surprised": {
        "required": ["AU01", "AU02", "AU05"], 
        "enhancers": ["AU25", "AU26", "AU27"], 
        "inhibitors": ["AU04"]
    },
    "Scared": {
        "required": ["AU01", "AU02", "AU04", "AU20"], 
        "enhancers": ["AU05", "AU25"], 
        "inhibitors": ["AU12"]
    },
    "Disgusted": {
        "required": ["AU09"], 
        "enhancers": ["AU10", "AU04", "AU17"], 
        "inhibitors": ["AU12"]
    },
    "Neutral": {
        "required": [], 
        "enhancers": [], 
        "inhibitors": []
    },
}

AU_DESCRIPTIONS = {
    "AU01": "Inner Brow Raiser",
    "AU02": "Outer Brow Raiser",
    "AU04": "Brow Lowerer",
    "AU05": "Upper Lid Raiser",
    "AU06": "Cheek Raiser",
    "AU07": "Lid Tightener",
    "AU09": "Nose Wrinkler",
    "AU10": "Upper Lip Raiser",
    "AU12": "Lip Corner Puller (Smile)",
    "AU14": "Dimpler",
    "AU15": "Lip Corner Depressor",
    "AU17": "Chin Raiser",
    "AU18": "Lip Pucker",
    "AU20": "Lip Stretcher",
    "AU23": "Lip Tightener",
    "AU24": "Lip Pressor",
    "AU25": "Lips Part",
    "AU26": "Jaw Drop",
    "AU27": "Mouth Stretch",
    "AU43": "Eyes Closed",
}
