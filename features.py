"""
features.py
Extracts a numeric feature vector from a hand contour. This is the
ML layer's "eyes" -- it turns a raw contour shape into numbers a
classifier can learn from.
"""

import cv2
import numpy as np

FEATURE_NAMES = [
    "solidity", "extent", "aspect_ratio", "num_defects",
    "avg_defect_depth", "contour_area_norm", "perimeter_norm",
    "hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7"
]


def extract_features(contour, roi_shape):
    """
    Build a feature vector describing the shape of a hand contour.

    contour   : the largest hand contour (from cv2.findContours)
    roi_shape : (height, width, ...) of the ROI the contour came from,
                used to normalize size so the model isn't sensitive
                to how close the hand is to the camera
    """
    roi_h, roi_w = roi_shape[:2]
    roi_diag = np.sqrt(roi_h ** 2 + roi_w ** 2)

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0
    aspect_ratio = w / h if h > 0 else 0

    # Convexity defects (finger valleys): count + average depth
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    num_defects = 0
    avg_depth = 0.0
    if hull_idx is not None and len(hull_idx) > 3:
        defects = cv2.convexityDefects(contour, hull_idx)
        if defects is not None:
            depths = defects[:, 0, 3]
            num_defects = len(depths)
            avg_depth = float(np.mean(depths)) / 256.0  # OpenCV stores depth*256

    # Hu moments: 7 rotation/scale-invariant shape descriptors
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)  # log-scale, raw values are tiny

    features = [
        solidity,
        extent,
        aspect_ratio,
        num_defects,
        avg_depth / roi_diag,
        area / (roi_diag ** 2),
        perimeter / roi_diag,
        *hu_log
    ]
    return np.array(features, dtype=np.float32)
