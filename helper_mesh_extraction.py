import numpy as np
import cv2
from scipy import ndimage
from skimage import morphology, measure
from skimage.measure import find_contours
from matplotlib import pyplot as plt
from collections import defaultdict
def fit_quad_or_tri(mask):
    """
    Use OpenCV's minAreaRect or convex hull + polygon fit
    directly on the pixel region.
    """
    pts = np.argwhere(mask)
    pts_cv = pts[:, ::-1].astype(np.float32)  # cv2 wants (x, y)

    hull = cv2.convexHull(pts_cv)

    # Try fitting with 4 vertices first, then 3
    for target_n in (4, 3):
        epsilon = 0.01 * cv2.arcLength(hull, True)
        for _ in range(50):
            approx = cv2.approxPolyDP(hull, epsilon, True)
            n = len(approx)
            if n == target_n:
                return approx.reshape(-1, 2)[:, ::-1]  # back to (row, col)
            elif n > target_n:
                epsilon *= 1.1
            else:
                epsilon *= 0.9
        
    # Fallback: minimum area rectangle (always gives 4 points)
    rect = cv2.minAreaRect(pts_cv)
    box = cv2.boxPoints(rect).astype(np.float32)
    return box[:, ::-1]


def extract_corners_adaptive(mask, pixels, area_threshold=200):
    area = pixels.shape[0]
    
    if area < area_threshold:
        # Too small for contour methods — use minimum area rect
        pts_cv = pixels[:, ::-1].astype(np.float32)
        rect = cv2.minAreaRect(pts_cv)
        box = cv2.boxPoints(rect)
        return box[:, ::-1]  # (row, col)
    else:
        # Large enough for contour-based approach
        return fit_quad_or_tri(mask)
    
def detect_corners_harris(mask, pixels, max_corners=4):
    """Use Harris/Shi-Tomasi corner detection on the face mask."""
    mask_uint8 = (mask.astype(np.uint8)) * 255

    corners = cv2.goodFeaturesToTrack(
        mask_uint8,
        maxCorners=max_corners,
        qualityLevel=0.1,
        minDistance=max(5, int(np.sqrt(pixels.shape[0]) * 0.2)),
    )

    if corners is None or len(corners) < 3:
        # Fallback to minAreaRect
        pts_cv = pixels[:, ::-1].astype(np.float32)
        rect = cv2.minAreaRect(pts_cv)
        box = cv2.boxPoints(rect)
        return box[:, ::-1]

    corners = corners.reshape(-1, 2)[:, ::-1]  # (row, col)
    
    # Order corners consistently (convex hull order)
    pts_cv = corners[:, ::-1].astype(np.float32)
    hull = cv2.convexHull(pts_cv)
    return hull.reshape(-1, 2)[:, ::-1]


def extract_corners(mask, pixels):
    area = pixels.shape[0]
    if area < 200:
        return extract_corners_adaptive(mask, pixels)  # minAreaRect
    elif area < 500:
        return detect_corners_harris(mask, pixels)      # feature-based
    else:
        return fit_quad_or_tri(mask)                     # contour-based
    


def expand_faces_fair(labeled_clean, valid_labels, background_mask):
    """
    Expand faces into unclaimed non-background, non-edge pixels.
    
    Args:
        labeled_clean:  labeled face array
        valid_labels:   list of face IDs
        edge_mask:      binary mask of detected edges (borders between faces)
        background_mask: binary mask of image background (outside the mesh)
    """
    expanded = labeled_clean.copy()
    
    # Pixels eligible for expansion: not a face, not background, not an edge
    unclaimed = (labeled_clean == 0) & ~background_mask 
    
    if not unclaimed.any():
        return expanded
    
    min_dist = np.full(labeled_clean.shape, np.inf)
    nearest_label = np.zeros(labeled_clean.shape, dtype=int)
    
    for i in valid_labels:
        if (labeled_clean == i).sum() > 200:
            continue  # skip large faces to avoid over-expansion
        mask = (labeled_clean == i)
        dist = ndimage.distance_transform_edt(~mask)
        closer = dist < min_dist
        min_dist[closer] = dist[closer]
        nearest_label[closer] = i
    
    expanded[unclaimed] = nearest_label[unclaimed]
    
    return expanded