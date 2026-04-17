import cv2
from matplotlib import contour
from matplotlib.patches import Polygon
from networkx import radius
from simplification.cutil import simplify_coords_vw
from sklearn.cluster import DBSCAN
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
from skimage import img_as_ubyte, measure, morphology
from skimage.feature import corner_harris, corner_peaks
from skimage.measure import approximate_polygon, find_contours, regionprops
from skimage import color
from shapely.geometry import Polygon
from skimage.segmentation import find_boundaries
from skimage.draw import disk

def crop_image(img, margin=10):
    """
    Hardcoded crop: 500px left and right, 200px top and bottom.
    Returns the cropped image and a dict with crop offsets so
    coordinates can be mapped back to the original frame later.
    """
    h, w = img.shape[:2]

    top = 150
    bottom = 150
    left = 500
    right = 400

    cropped = img[top:h - bottom, left:w - right]

    crop_info = {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "original_shape": (h, w),
    }
    return cropped, crop_info
def _extract_grayscale(img, min_face_size):
#     """Original grayscale path: CLAHE + threshold at 150."""
#     # In _extract_grayscale
#     gray = (color.rgb2gray(img) * 255).astype(np.uint8)

# # Bypassing CLAHE for clean, synthetic images:
# # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# # img_eq = clahe.apply(gray)
#     img_eq = gray.copy() 

#     edge_mask = cv2.Canny(img_eq, threshold1=75, threshold2=120)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     closed = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
#     skeleton = morphology.skeletonize(closed > 0)

#     bright_mask = (img_eq > 220).astype(np.uint8)
#     bright_mask = cv2.dilate(bright_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
#     img_eq = cv2.inpaint(img_eq, bright_mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
#     img_eq[skeleton] = 0
#     img_eq[img_eq < 150] = 0

#     not_black = img_eq > 0
#     not_black = morphology.remove_small_objects(not_black, min_size=200)
#     labeled_clean = measure.label(not_black, connectivity=1)

#     valid_labels = [
#         i for i in range(1, labeled_clean.max() + 1)
#         if (labeled_clean == i).sum() > min_face_size
#     ]
#     face_pixels = {i: np.argwhere(labeled_clean == i) for i in valid_labels}

#     # Drop faces whose pixels touch the left or bottom image margin —
#     # these are axis/border artifacts that are never adjacent to real faces.
#     H_img, W_img = labeled_clean.shape
#     EDGE_MARGIN = 200
#     valid_labels = [fid for fid in valid_labels
#                     if face_pixels[fid][:, 1].min() >= EDGE_MARGIN
#                     and face_pixels[fid][:, 0].max() <= H_img - EDGE_MARGIN /2 ]
#     face_pixels = {fid: face_pixels[fid] for fid in valid_labels}

#     return labeled_clean, valid_labels, face_pixels

# def _extract_grayscale(img, min_face_size):
#     # 1. Convert to grayscale
#     gray = (color.rgb2gray(img) * 255).astype(np.uint8)
    
#     # 2. Thresholding: Red lines (Y~76) and background (Y=0) become black.
#     # The green faces (Y~210) easily survive the 150 threshold.
#     # This automatically separates the faces!
#     gray[gray < 150] = 0 
    
#     # 3. Quick Canny pass just to ensure the red line gaps are fully respected
#     edge_mask = cv2.Canny(gray, threshold1=50, threshold2=100)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     closed = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
#     skeleton = morphology.skeletonize(closed > 0)
    
#     # Cut along the skeleton
#     gray[skeleton] = 0
    
#     # 4. Label the surviving bright green islands
#     not_black = gray > 0
#     not_black = morphology.remove_small_objects(not_black, min_size=200)
#     labeled_clean = measure.label(not_black, connectivity=1)

#     valid_labels = [
#         i for i in range(1, labeled_clean.max() + 1)
#         if (labeled_clean == i).sum() > min_face_size
#     ]
#     face_pixels = {i: np.argwhere(labeled_clean == i) for i in valid_labels}

#     # 5. Relaxed margin (10 instead of 200) so outer faces aren't deleted
#     H_img, W_img = labeled_clean.shape
#     EDGE_MARGIN = 10
    
#     valid_labels = [fid for fid in valid_labels
#                     if face_pixels[fid][:, 1].min() >= EDGE_MARGIN
#                     and face_pixels[fid][:, 0].max() <= H_img - EDGE_MARGIN / 2 ]
    
#     face_pixels = {fid: face_pixels[fid] for fid in valid_labels}

#     return labeled_clean, valid_labels, face_pixels


def _extract_hsv(img, min_face_size):
    """HSV Value-channel path for dark-colored meshes (purple, dark orange).

    Value-based foreground:
      - excludes black background (V≈0)
      - includes any non-black mesh face regardless of hue
    """
    img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    foreground_mask = hsv[:, :, 2] > 30

    gray = (color.rgb2gray(img) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(gray)

    white_dots = (img_eq > 230).astype(np.uint8)
    white_dots = cv2.dilate(white_dots, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    foreground_mask[white_dots > 0] = False
    img_eq = cv2.inpaint(img_eq, white_dots, inpaintRadius=12, flags=cv2.INPAINT_TELEA)

    # Edge pass: union CLAHE-gray Canny with Canny on raw V. The V channel
    # preserves subtle internal face-boundary lines that CLAHE flattens,
    # which on dark meshes (088) is the difference between detecting and
    # missing a whole face split.
    edge_gray = cv2.Canny(img_eq, threshold1=75, threshold2=120)
    edge_v = cv2.Canny(hsv[:, :, 2], threshold1=80, threshold2=180)
    edge_mask = cv2.bitwise_or(edge_gray, edge_v)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
    skeleton = morphology.skeletonize(closed > 0)

    face_region = foreground_mask.astype(np.uint8)
    face_region[skeleton] = 0
    face_region_bool = morphology.remove_small_objects(face_region.astype(bool), min_size=200)

    padded = np.pad(face_region_bool, 1, mode='constant', constant_values=0)
    labeled_padded = measure.label(padded, connectivity=1)
    labeled_clean = labeled_padded[1:-1, 1:-1]

    valid_labels = []
    for i in range(1, labeled_clean.max() + 1):
        pix = np.argwhere(labeled_clean == i)
        if len(pix) <= min_face_size:
            continue
        # sliver reject: thin high-aspect strips with low bbox fill are
        # edge-detection artifacts from the V-Canny union, not real faces.
        r0, c0 = pix.min(0)
        r1, c1 = pix.max(0)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        ar = max(h, w) / max(1, min(h, w))
        fill = len(pix) / (h * w)
        if ar > 2 and fill < 0.3:
            labeled_clean[labeled_clean == i] = 0
            continue
        valid_labels.append(i)
    face_pixels = {i: np.argwhere(labeled_clean == i) for i in valid_labels}

    # Drop faces whose pixels touch the left or bottom image margin —
    # these are axis/border artifacts that are never adjacent to real faces.
    H_img, W_img = labeled_clean.shape
    EDGE_MARGIN = 200
    valid_labels = [
            fid for fid in valid_labels
            if face_pixels[fid][:, 1].min() >= EDGE_MARGIN          # Left margin
            and face_pixels[fid][:, 1].max() <= W_img - EDGE_MARGIN # Right margin
            and face_pixels[fid][:, 0].min() >= EDGE_MARGIN          # Top margin
            and face_pixels[fid][:, 0].max() <= H_img - EDGE_MARGIN  # Bottom margin
        ]
    face_pixels = {fid: face_pixels[fid] for fid in valid_labels}

    return labeled_clean, valid_labels, face_pixels


_used_hsv = False


def _mean_nonblack_luminosity(img):
    """Return the mean grayscale luminosity of non-black pixels.

    Used to decide whether the grayscale (rgb2gray + threshold 150)
    path will work: dark-colored meshes (purple, dark-orange) have
    low luminance and fall below the threshold, so we need the HSV
    fallback.
    """
    gray = (color.rgb2gray(img) * 255).astype(np.uint8)
    nonblack = gray > 20
    if not nonblack.any():
        return 0
    return int(gray[nonblack].mean())


def extract_face_masks(img, min_face_size=500, dilation_radius=1,
                       lum_fallback_thresh=148):
    """
    Extraction with luminosity-based dispatch.

    Mean luminosity of non-black pixels determines which path runs:
      * L >= lum_fallback_thresh → grayscale CLAHE+threshold path
      * L <  lum_fallback_thresh → HSV/saturation fallback (for dark
        colored meshes that the grayscale threshold would kill).
    """
    global _used_hsv

    lum = _mean_nonblack_luminosity(img)
    if lum < lum_fallback_thresh:
        print(f"[extract_face_masks] mean luminosity {lum} < {lum_fallback_thresh}, using HSV")
        _used_hsv = True
        return _extract_hsv(img, min_face_size)

    labeled, valid, fpix = _extract_grayscale(img, min_face_size)
    if len(valid) >= 3:
        _used_hsv = False
        return labeled, valid, fpix
    print(f"[extract_face_masks] grayscale found only {len(valid)} faces (lum={lum}), falling back to HSV")
    _used_hsv = True
    return _extract_hsv(img, min_face_size)


def _refine_grayscale(img, labeled_low, valid_low, min_face_size=500,
                      cannylow=300, cannyhigh=400):
    """Original grayscale refine path."""
    gray = (color.rgb2gray(img) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(gray)

    edge_mask = cv2.Canny(img_eq, cannylow, cannyhigh, apertureSize=3, L2gradient=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
    skeleton = morphology.skeletonize(closed > 0)

    bright_mask = (img_eq > 220).astype(np.uint8)
    bright_mask = cv2.dilate(bright_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    img_eq = cv2.inpaint(img_eq, bright_mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
    img_eq[skeleton] = 0
    img_eq[img_eq < 150] = 0

    not_black = img_eq > 0
    not_black = morphology.remove_small_objects(not_black, min_size=200)
    labeled_high = measure.label(not_black, connectivity=1)

    refined = np.zeros_like(labeled_low)
    used_high = set()

    for fid in valid_low:
        mask_low = labeled_low == fid
        overlap = labeled_high[mask_low]
        overlap = overlap[overlap > 0]
        if len(overlap) == 0:
            refined[mask_low] = fid
            continue

        candidates, counts = np.unique(overlap, return_counts=True)
        best = candidates[np.argmax(counts)]
        mask_high = labeled_high == best

        if mask_high.sum() > 1.6 * mask_low.sum():
            refined[mask_low] = fid
            continue

        low_in_high = labeled_low[mask_high]
        low_in_high = low_in_high[low_in_high > 0]
        foreign_px = int((low_in_high != fid).sum())
        if foreign_px > 0.2 * mask_high.sum():
            refined[mask_low] = fid
            continue

        if best in used_high:
            refined[mask_low] = fid
            continue

        refined[mask_high] = fid
        used_high.add(best)

    face_pixels = {}
    valid_labels = []
    for i in valid_low:
        pix = np.argwhere(refined == i)
        if len(pix) == 0:
            continue
        face_pixels[i] = pix
        valid_labels.append(i)
    return refined, valid_labels, face_pixels


def _refine_hsv(img, labeled_low, valid_low, min_face_size=500,
                cannylow=300, cannyhigh=400):
    """Value-based refine path (matches _extract_hsv)."""
    img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    foreground_mask = hsv[:, :, 2] > 30

    gray = (color.rgb2gray(img) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(gray)

    white_dots = (img_eq > 230).astype(np.uint8)
    white_dots = cv2.dilate(white_dots, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    foreground_mask[white_dots > 0] = False
    img_eq = cv2.inpaint(img_eq, white_dots, inpaintRadius=12, flags=cv2.INPAINT_TELEA)

    # Edge pass: union CLAHE-gray Canny with Canny on raw V (see _extract_hsv).
    edge_gray = cv2.Canny(img_eq, cannylow, cannyhigh, apertureSize=3, L2gradient=True)
    edge_v = cv2.Canny(hsv[:, :, 2], 80, 180)
    edge_mask = cv2.bitwise_or(edge_gray, edge_v)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
    skeleton = morphology.skeletonize(closed > 0)

    # Use foreground mask as base; cut along skeleton barriers
    face_region = foreground_mask.astype(np.uint8)
    face_region[skeleton] = 0

    face_region_bool = morphology.remove_small_objects(face_region.astype(bool), min_size=200)

    padded = np.pad(face_region_bool, 1, mode='constant', constant_values=0)
    labeled_padded = measure.label(padded, connectivity=1)
    labeled_high = labeled_padded[1:-1, 1:-1]

    refined = np.zeros_like(labeled_low)
    used_high = set()

    for fid in valid_low:
        mask_low = labeled_low == fid
        overlap = labeled_high[mask_low]
        overlap = overlap[overlap > 0]
        if len(overlap) == 0:
            refined[mask_low] = fid
            continue

        candidates, counts = np.unique(overlap, return_counts=True)
        best = candidates[np.argmax(counts)]
        mask_high = labeled_high == best

        if mask_high.sum() > 1.6 * mask_low.sum():
            refined[mask_low] = fid
            continue

        low_in_high = labeled_low[mask_high]
        low_in_high = low_in_high[low_in_high > 0]
        foreign_px = int((low_in_high != fid).sum())
        if foreign_px > 0.2 * mask_high.sum():
            refined[mask_low] = fid
            continue

        if best in used_high:
            refined[mask_low] = fid
            continue

        refined[mask_high] = fid
        used_high.add(best)

    face_pixels = {}
    valid_labels = []
    for i in valid_low:
        pix = np.argwhere(refined == i)
        if len(pix) == 0:
            continue
        face_pixels[i] = pix
        valid_labels.append(i)
    return refined, valid_labels, face_pixels


def refine_faces(img, labeled_low, face_pixels_low, valid_low,
                 min_face_size=500, cannylow=300, cannyhigh=400):
    global _used_hsv
    if _used_hsv:
        return _refine_hsv(img, labeled_low, valid_low,
                           min_face_size, cannylow, cannyhigh)
    return _refine_grayscale(img, labeled_low, valid_low,
                             min_face_size, cannylow, cannyhigh)


def extract_face_corners(labeled_clean, face_pixels, tolerance=4):
    face_corners = {}
    for i, pixels in face_pixels.items():
        mask = (labeled_clean == i)
        contours = find_contours(mask, 0.5)
        if not contours:
            continue
        tol = 4 if len(pixels) < 300 else 10
        contour = max(contours, key=len)
        approx = approximate_polygon(contour.astype(np.float32), tolerance=tolerance)
        face_corners[i] = approx.reshape(-1, 2)
    return face_corners

def filter_corners(face_corners, labeled_clean, valid_labels, radius=15):
    filtered = {}
    for fid in valid_labels:
        mask = (labeled_clean == fid)
        kept = []
        for pt in face_corners[fid]:
            r, c = int(pt[0]), int(pt[1])
            rr, cc = disk((r, c), radius, shape=mask.shape)
            circle_area = len(rr)
            inside = mask[rr, cc].sum()
            if inside / circle_area <= 0.5:
                kept.append(pt)
        filtered[fid] = np.array(kept) if kept else np.empty((0, 2))
    return filtered

def compute_adjacency(labeled_clean, valid_labels, face_pixels,
                      shared_border_threshold=30, bulbs=None, bulb_radius=4):
    n = len(valid_labels)
    label_to_idx = {l: a for a, l in enumerate(valid_labels)}
    adjacency = np.zeros((n, n), dtype=int)
    face_centroids = {i: pixels.mean(axis=0) for i, pixels in face_pixels.items()}

    selem = morphology.disk(6)
    debug_pairs = [(8, 7), (5,4)]

    # ── vectorized border counting ────────────────────────────────────────
    for a, i in enumerate(valid_labels):
        mask_i = (labeled_clean == i)
        dilated_i = morphology.binary_dilation(mask_i, selem)
        border_i = dilated_i & ~mask_i

        border_labels = labeled_clean[border_i]
        labels, counts = np.unique(border_labels, return_counts=True)
        for lbl, cnt in zip(labels, counts):
            if lbl in label_to_idx and lbl != i:
                b = label_to_idx[lbl]
                adjacency[a, b] += cnt
                # debug
                for di, dj in debug_pairs:
                    if i == di and lbl == dj:
                        print(f"  Border {i}→{lbl}: {cnt} pixels")

    adjacency = np.maximum(adjacency, adjacency.T)

    for di, dj in debug_pairs:
        if di in label_to_idx and dj in label_to_idx:
            a, b = label_to_idx[di], label_to_idx[dj]
            print(f"[DEBUG] Pair {di}-{dj}: border count (symmetrized) = {adjacency[a, b]}")

    # ── vectorized bulb boost ─────────────────────────────────────────────
    if bulbs is not None and len(bulbs) > 1:
        bulb_coords = np.round(np.array(bulbs)).astype(int)
        bulb_area = int(np.pi * bulb_radius ** 2)

        bulb_in_face = np.zeros((len(bulb_coords), n), dtype=bool)
        for a, i in enumerate(valid_labels):
            mask_i = (labeled_clean == i)
            dilated_i = morphology.binary_dilation(mask_i, selem)
            bulb_in_face[:, a] = dilated_i[bulb_coords[:, 0], bulb_coords[:, 1]]

        shared_bulbs = bulb_in_face.astype(int).T @ bulb_in_face.astype(int)

        for di, dj in debug_pairs:
            if di in label_to_idx and dj in label_to_idx:
                a, b = label_to_idx[di], label_to_idx[dj]
                shared_count = shared_bulbs[a, b]
                print(f"[DEBUG] Pair {di}-{dj}: shared bulbs = {shared_count}")
                # which bulbs?
                mask = bulb_in_face[:, a] & bulb_in_face[:, b]
                for idx in np.where(mask)[0]:
                    print(f"  bulb {idx} at ({bulb_coords[idx, 0]}, {bulb_coords[idx, 1]})")

        boost_mask = shared_bulbs >= 2
        adjacency += boost_mask * shared_bulbs * bulb_area

        for di, dj in debug_pairs:
            if di in label_to_idx and dj in label_to_idx:
                a, b = label_to_idx[di], label_to_idx[dj]
                print(f"[DEBUG] Pair {di}-{dj}: final count after boost = {adjacency[a, b]}, threshold = {shared_border_threshold}")

    # ── threshold ─────────────────────────────────────────────────────────
    adj_bool = adjacency > shared_border_threshold
    np.fill_diagonal(adj_bool, False)

    adjacent_faces = {i: [] for i in valid_labels}
    for a, i in enumerate(valid_labels):
        for b, j in enumerate(valid_labels):
            if adj_bool[a, b]:
                adjacent_faces[i].append(j)

    return adj_bool, adjacent_faces, face_centroids, adjacency

import numpy as np
import networkx as nx
from collections import defaultdict

def merge_vertices(face_corners, valid_labels, adjacency, face_pixels,
                               bulbs=None, dilation_radius=8):
    """
    Merges corners into vertices by finding the spatial intersection
    of morphologically dilated face masks for every topological cycle.

    Each cycle (size <= 6) with at least one candidate corner produces
    exactly one vertex. Cycles with no candidates are skipped. Corners
    not absorbed into any junction stay as their own vertices.

    If `bulbs` (true-corner coordinates) are provided, any bulb that
    lies inside a cycle's intersection mask wins and becomes the cycle's
    vertex. If multiple bulbs lie inside, the one closest to the
    centroid of the intersection region is chosen.
    """
    # 1. Global corner pool
    all_pts = []
    face_pt_indices = {}
    pt_to_face = {}
    curr_idx = 0
    for fid in valid_labels:
        corners = face_corners.get(fid, np.empty((0, 2)))
        face_pt_indices[fid] = np.arange(curr_idx, curr_idx + len(corners))
        for k in range(len(corners)):
            pt_to_face[curr_idx + k] = fid
        all_pts.extend(corners)
        curr_idx += len(corners)
    all_pts = np.array(all_pts)

    # 2. Build dilated mask per face using morphology.disk
    max_r = max(int(p[:, 0].max()) for p in face_pixels.values() if len(p) > 0)
    max_c = max(int(p[:, 1].max()) for p in face_pixels.values() if len(p) > 0)
    H = max_r + dilation_radius + 2
    W = max_c + dilation_radius + 2
    selem = morphology.disk(dilation_radius)

    dilated_masks = {}
    for fid in valid_labels:
        mask = np.zeros((H, W), dtype=bool)
        pix = face_pixels[fid]
        mask[pix[:, 0], pix[:, 1]] = True
        dilated_masks[fid] = morphology.binary_dilation(mask, selem)

    # 3. Bulb array (in row, col)
    if bulbs is not None and len(bulbs) > 0:
        bulb_arr = np.asarray(bulbs, dtype=float)
    else:
        bulb_arr = np.empty((0, 2))

    # 4. Adjacency graph — add every face as a node first so that
    # isolated faces (0 neighbours) don't cause NetworkXError in step 8b.
    G = nx.Graph()
    G.add_nodes_from(valid_labels)
    for a, i in enumerate(valid_labels):
        for b, j in enumerate(valid_labels):
            if adjacency[a, b] and j > i:
                G.add_edge(i, j)

    # 5. Cycles
    cycles = nx.minimum_cycle_basis(G)
    edge_coverage = defaultdict(int)  # times each edge was covered by a bulb or cycle
    # 6. Per-cycle junction vertex
    vertices = {}
    pt_to_vertex = {}
    face_extra_vids = defaultdict(list)
    face_merged_vids = defaultdict(set)   # track merged vids per face
    used_bulbs = set()  # bulb indices already materialized as a vertex
    next_vid = 1
    MAX_CORNERS = 4
    DEBUG_FACES = [8]  # face to trace

    # 6a. Materialize every bulb as a vertex up-front. Bulbs are
    # ground-truth junction locations; if we wait until after the cycle
    # and edge passes (as before), cycles whose intersection mask misses
    # the bulb will spawn their own junction corners and we end up with
    # both the bulb vertex and the corner-averaged vertex on the same
    # face — which then trips MAX_CORNERS=4 and drops the *bulb*.
    if len(bulb_arr) > 0:
        bulb_rc = np.round(bulb_arr).astype(int)
        for bi in range(len(bulb_arr)):
            br, bc = bulb_rc[bi]
            if not (0 <= br < H and 0 <= bc < W):
                continue
            faces_here = [fid for fid in valid_labels
                          if dilated_masks[fid][br, bc]]
            if not faces_here:
                continue
            bulb_pos = bulb_arr[bi]
            vertices[next_vid] = bulb_pos
            absorbed = []
            for fid in faces_here:
                for idx in face_pt_indices[fid]:
                    idx = int(idx)
                    if idx in pt_to_vertex:
                        continue
                    if np.linalg.norm(all_pts[idx] - bulb_pos) <= dilation_radius:
                        pt_to_vertex[idx] = next_vid
                        absorbed.append(idx)
            for fid in faces_here:
                face_merged_vids[fid].add(next_vid)
                if not any(pt_to_face[idx] == fid for idx in absorbed):
                    face_extra_vids[fid].append(next_vid)
                if fid in DEBUG_FACES:
                    print(f"[BORN] vid={next_vid} @ {bulb_pos} | stage=6a-upfront-bulb bi={bi} | assigned to face {fid} | faces_here={faces_here} | extra={not any(pt_to_face[idx] == fid for idx in absorbed)}")
            used_bulbs.add(bi)
            # Count all adjacent pairs from faces_here as one coverage event.
            for i_f in range(len(faces_here)):
                for j_f in range(i_f + 1, len(faces_here)):
                    u_f, v_f = faces_here[i_f], faces_here[j_f]
                    if G.has_edge(u_f, v_f):
                        edge_coverage[(min(u_f, v_f), max(u_f, v_f))] += 1
            next_vid += 1

    for cycle in cycles:
        if len(cycle) > 6:
            continue

        # Intersection of dilated masks for every face in this cycle.
        # Drop any face whose mask kills the intersection (destroyed/
        # poorly-refined face that doesn't reach the junction).
        surviving = [cycle[0]]
        inter = dilated_masks[cycle[0]].copy()
        for fid in cycle[1:]:
            trial = inter & dilated_masks[fid]
            if trial.any():
                inter = trial
                surviving.append(fid)
        if len(surviving) < 2:
            continue
        dropped = [fid for fid in cycle if fid not in surviving]
        cycle = surviving

        relevant_indices = np.concatenate([face_pt_indices[fid] for fid in cycle])
        if len(relevant_indices) == 0:
            continue

        # ── Try bulb-based vertex first ───────────────────────────────
        # A bulb lies in the cycle's junction zone iff it falls inside
        # the intersection mask. If multiple do, pick the one nearest
        # to the centroid of the intersection region.
        vertex_pos = None
        vertex_from_bulb = False
        if len(bulb_arr) > 0:
            bulb_rc = np.round(bulb_arr).astype(int)
            in_bounds = (
                (bulb_rc[:, 0] >= 0) & (bulb_rc[:, 0] < H) &
                (bulb_rc[:, 1] >= 0) & (bulb_rc[:, 1] < W)
            )
            inside = np.zeros(len(bulb_arr), dtype=bool)
            inside[in_bounds] = inter[bulb_rc[in_bounds, 0],
                                      bulb_rc[in_bounds, 1]]
            # Skip bulbs already materialized in the up-front sweep.
            hits = np.array([h for h in np.where(inside)[0] if int(h) not in used_bulbs])
            if len(hits) == 1:
                vertex_pos = bulb_arr[hits[0]]
                vertex_from_bulb = True
                used_bulbs.add(int(hits[0]))
            elif len(hits) > 1:
                inter_rows, inter_cols = np.where(inter)
                centroid = np.array([inter_rows.mean(), inter_cols.mean()])
                d = np.linalg.norm(bulb_arr[hits] - centroid, axis=1)
                chosen = int(hits[np.argmin(d)])
                vertex_pos = bulb_arr[chosen]
                vertex_from_bulb = True
                used_bulbs.add(chosen)

        # ── Fallback: average of intersection-zone corners ────────────
        if vertex_pos is None:
            corner_in_inter = []
            for idx in relevant_indices:
                idx = int(idx)
                r, c = all_pts[idx]
                ri, ci2 = int(round(r)), int(round(c))
                if 0 <= ri < H and 0 <= ci2 < W and inter[ri, ci2]:
                    corner_in_inter.append(idx)
            if not corner_in_inter:
                continue
            vertex_pos = all_pts[corner_in_inter].mean(axis=0)

        # Claim every unclaimed corner of a cycle face that belongs to
        # this junction. Always claim corners that lie inside the
        # intersection mask. When the vertex was pinned to a bulb (a
        # ground-truth corner location), additionally absorb any
        # straggler corner within `dilation_radius` of the bulb, so that
        # duplicate polygon-approximation corners don't survive as
        # singleton vertices visually near the junction.
        winners = []
        for idx in relevant_indices:
            idx = int(idx)
            if idx in pt_to_vertex:
                continue
            r, c = all_pts[idx]
            ri, ci2 = int(round(r)), int(round(c))
            in_inter = 0 <= ri < H and 0 <= ci2 < W and inter[ri, ci2]
            near_bulb = (
                vertex_from_bulb
                and np.linalg.norm(all_pts[idx] - vertex_pos) <= dilation_radius
            )
            if in_inter or near_bulb:
                winners.append(idx)

        vertices[next_vid] = vertex_pos
        for idx in winners:
            pt_to_vertex[idx] = next_vid

        winner_faces = {pt_to_face[idx] for idx in winners}
        for fid in cycle:
            # Respect MAX_CORNERS: a face already at the cap doesn't take
            # on a 5th vertex — this is where a bulb-anchored cluster
            # earlier in the pipeline outranks a later cycle junction.
            if len(face_merged_vids[fid]) >= MAX_CORNERS:
                if fid in DEBUG_FACES:
                    print(f"[SKIP] vid={next_vid} @ {vertex_pos} | stage=6-cycle | face {fid} already at MAX_CORNERS={MAX_CORNERS} | cycle={cycle}")
                continue
            face_merged_vids[fid].add(next_vid)
            if fid not in winner_faces:
                face_extra_vids[fid].append(next_vid)
            if fid in DEBUG_FACES:
                print(f"[BORN] vid={next_vid} @ {vertex_pos} | stage=6-cycle | cycle={cycle} | from_bulb={vertex_from_bulb} | assigned to face {fid} | extra={fid not in winner_faces}")

        # Dropped faces were originally in this cycle — always assign
        # the junction vertex to them.
        for fid in dropped:
            if len(face_merged_vids[fid]) >= MAX_CORNERS:
                if fid in DEBUG_FACES:
                    print(f"[SKIP] vid={next_vid} @ {vertex_pos} | stage=6-cycle-dropped | face {fid} already at MAX_CORNERS | cycle original included {fid }")
                continue
            face_merged_vids[fid].add(next_vid)
            face_extra_vids[fid].append(next_vid)
            if fid in DEBUG_FACES:
                print(f"[BORN] vid={next_vid} @ {vertex_pos} | stage=6-cycle-dropped | cycle={cycle} | assigned to face {fid} (was dropped from cycle)")
        # Dropped faces were originally in this cycle — always assign
        # the junction vertex to them.
        for fid in dropped:
            if len(face_merged_vids[fid]) >= MAX_CORNERS:
                if fid in DEBUG_FACES:
                    print(f"[SKIP] vid={next_vid} @ {vertex_pos} | stage=6-cycle-dropped | face {fid} already at MAX_CORNERS | cycle original included {fid }")
                continue
            face_merged_vids[fid].add(next_vid)
            face_extra_vids[fid].append(next_vid)
            if fid in DEBUG_FACES:
                print(f"[BORN] vid={next_vid} @ {vertex_pos} | stage=6-cycle-dropped | cycle={cycle} | assigned to face {fid} (was dropped from cycle)")

        # Count all edges covered by this cycle vertex as one coverage event.
        assigned_faces = cycle + dropped
        for i_f in range(len(assigned_faces)):
            for j_f in range(i_f + 1, len(assigned_faces)):
                u_f, v_f = assigned_faces[i_f], assigned_faces[j_f]
                if G.has_edge(u_f, v_f):
                    edge_coverage[(min(u_f, v_f), max(u_f, v_f))] += 1
        next_vid += 1


    # 6b. Boundary-cycle pass. A chain of faces x-y-...-d where x and d
    # are NOT directly adjacent in G, but their dilated masks still
    # intersect, behaves like a cycle whose closing edge runs through the
    # image boundary (typical for inner/outer-circle faces cut in half).
    # We apply the same junction-vertex logic that stage 6 uses for real
    # topological cycles. Purely additive: only spawns new vertices for
    # corners not yet claimed by a real cycle or bulb, and respects
    # MAX_CORNERS exactly like stage 6 does.
    existing_cycle_sets = {frozenset(c) for c in cycles if len(c) <= 6}

    def _is_subset_of_existing_cycle(path_set):
        for cs in existing_cycle_sets:
            if path_set.issubset(cs):
                return True
        return False

    boundary_cycles = []
    seen_boundary = set()
    valid_list = list(valid_labels)
    for a_idx, u_face in enumerate(valid_list):
        for v_face in valid_list[a_idx + 1:]:
            if G.has_edge(u_face, v_face):
                continue
            if u_face == v_face:
                continue
            # Masks of the two endpoints must intersect (they "share" a
            # vertex through the boundary even though the adjacency graph
            # doesn't connect them directly).
            inter_end = dilated_masks[u_face] & dilated_masks[v_face]
            if not inter_end.any():
                continue
            try:
                path = nx.shortest_path(G, u_face, v_face)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            # Need at least one intermediate face; cap at 6 like stage 6.
            if len(path) < 3 or len(path) > 6:
                continue
            path_set = frozenset(path)
            if path_set in seen_boundary:
                continue
            if path_set in existing_cycle_sets:
                continue
            if _is_subset_of_existing_cycle(path_set):
                continue
            seen_boundary.add(path_set)
            boundary_cycles.append(list(path))

    for cycle in boundary_cycles:
        if len(cycle) > 6:
            continue

        # Intersection of dilated masks for every face in this pseudo-cycle.
        surviving = [cycle[0]]
        inter = dilated_masks[cycle[0]].copy()
        for fid in cycle[1:]:
            trial = inter & dilated_masks[fid]
            if trial.any():
                inter = trial
                surviving.append(fid)
        if len(surviving) < 2:
            continue
        dropped = [fid for fid in cycle if fid not in surviving]
        cycle = surviving

        relevant_indices = np.concatenate([face_pt_indices[fid] for fid in cycle])
        if len(relevant_indices) == 0:
            continue

        # ── Try bulb-based vertex first ───────────────────────────────
        vertex_pos = None
        vertex_from_bulb = False
        if len(bulb_arr) > 0:
            bulb_rc = np.round(bulb_arr).astype(int)
            in_bounds = (
                (bulb_rc[:, 0] >= 0) & (bulb_rc[:, 0] < H) &
                (bulb_rc[:, 1] >= 0) & (bulb_rc[:, 1] < W)
            )
            inside = np.zeros(len(bulb_arr), dtype=bool)
            inside[in_bounds] = inter[bulb_rc[in_bounds, 0],
                                      bulb_rc[in_bounds, 1]]
            hits = np.array([h for h in np.where(inside)[0] if int(h) not in used_bulbs])
            if len(hits) == 1:
                vertex_pos = bulb_arr[hits[0]]
                vertex_from_bulb = True
                used_bulbs.add(int(hits[0]))
            elif len(hits) > 1:
                inter_rows, inter_cols = np.where(inter)
                centroid = np.array([inter_rows.mean(), inter_cols.mean()])
                d = np.linalg.norm(bulb_arr[hits] - centroid, axis=1)
                chosen = int(hits[np.argmin(d)])
                vertex_pos = bulb_arr[chosen]
                vertex_from_bulb = True
                used_bulbs.add(chosen)

        # ── Fallback: average of intersection-zone corners ────────────
        if vertex_pos is None:
            corner_in_inter = []
            for idx in relevant_indices:
                idx = int(idx)
                r, c = all_pts[idx]
                ri, ci2 = int(round(r)), int(round(c))
                if 0 <= ri < H and 0 <= ci2 < W and inter[ri, ci2]:
                    corner_in_inter.append(idx)
            if not corner_in_inter:
                continue
            vertex_pos = all_pts[corner_in_inter].mean(axis=0)

        # Claim intersection-zone corners (and, if bulb-anchored, any
        # stragglers within dilation_radius of the bulb).
        winners = []
        for idx in relevant_indices:
            idx = int(idx)
            if idx in pt_to_vertex:
                continue
            r, c = all_pts[idx]
            ri, ci2 = int(round(r)), int(round(c))
            in_inter = 0 <= ri < H and 0 <= ci2 < W and inter[ri, ci2]
            near_bulb = (
                vertex_from_bulb
                and np.linalg.norm(all_pts[idx] - vertex_pos) <= dilation_radius
            )
            if in_inter or near_bulb:
                winners.append(idx)

        vertices[next_vid] = vertex_pos
        for idx in winners:
            pt_to_vertex[idx] = next_vid

        winner_faces = {pt_to_face[idx] for idx in winners}
        for fid in cycle:
            if len(face_merged_vids[fid]) >= MAX_CORNERS:
                if fid in DEBUG_FACES:
                    print(f"[SKIP] vid={next_vid} @ {vertex_pos} | stage=6b-boundary-cycle | face {fid} already at MAX_CORNERS={MAX_CORNERS} | cycle={cycle}")
                continue
            face_merged_vids[fid].add(next_vid)
            if fid not in winner_faces:
                face_extra_vids[fid].append(next_vid)
            if fid in DEBUG_FACES:
                print(f"[BORN] vid={next_vid} @ {vertex_pos} | stage=6b-boundary-cycle | cycle={cycle} | from_bulb={vertex_from_bulb} | assigned to face {fid} | extra={fid not in winner_faces}")

        # Dropped faces were part of the boundary path — always assign.
        for fid in dropped:
            if len(face_merged_vids[fid]) >= MAX_CORNERS:
                if fid in DEBUG_FACES:
                    print(f"[SKIP] vid={next_vid} @ {vertex_pos} | stage=6b-boundary-cycle-dropped | face {fid} already at MAX_CORNERS")
                continue
            face_merged_vids[fid].add(next_vid)
            face_extra_vids[fid].append(next_vid)
            if fid in DEBUG_FACES:
                print(f"[BORN] vid={next_vid} @ {vertex_pos} | stage=6b-boundary-cycle-dropped | cycle={cycle} | assigned to face {fid}")

        # Count edge coverage exactly like stage 6 does.
        assigned_faces = cycle + dropped
        for i_f in range(len(assigned_faces)):
            for j_f in range(i_f + 1, len(assigned_faces)):
                u_f, v_f = assigned_faces[i_f], assigned_faces[j_f]
                if G.has_edge(u_f, v_f):
                    edge_coverage[(min(u_f, v_f), max(u_f, v_f))] += 1
        next_vid += 1



    # 7. Boundary-edge pass: adjacent face pairs (u, v) that weren't
    # fully covered by cycles. The intersection of their dilated masks
    # is a strip along the shared edge; the two endpoints of that edge
    # are the vertices we want to produce here.
    def _corner_in_mask(idx, mask):
        r, c = all_pts[idx]
        ri, ci = int(round(r)), int(round(c))
        return 0 <= ri < H and 0 <= ci < W and mask[ri, ci]

    for u, v in G.edges():
        # Skip if this edge was already covered by 2+ events (bulbs or cycles).
        if edge_coverage[(min(u, v), max(u, v))] >= 2:
            continue
        inter_uv = dilated_masks[u] & dilated_masks[v]
        if not inter_uv.any():
            continue

        cand_u = [int(idx) for idx in face_pt_indices[u]
                  if int(idx) not in pt_to_vertex
                  and _corner_in_mask(int(idx), inter_uv)]
        cand_v = [int(idx) for idx in face_pt_indices[v]
                  if int(idx) not in pt_to_vertex
                  and _corner_in_mask(int(idx), inter_uv)]
        if not cand_u and not cand_v:
            continue

        # ── Bulb-assisted merges first ────────────────────────────────
        if len(bulb_arr) > 0:
            bulb_rc = np.round(bulb_arr).astype(int)
            in_bounds = (
                (bulb_rc[:, 0] >= 0) & (bulb_rc[:, 0] < H) &
                (bulb_rc[:, 1] >= 0) & (bulb_rc[:, 1] < W)
            )
            inside = np.zeros(len(bulb_arr), dtype=bool)
            inside[in_bounds] = inter_uv[bulb_rc[in_bounds, 0],
                                         bulb_rc[in_bounds, 1]]
            for bi in np.where(inside)[0]:
                if int(bi) in used_bulbs:
                    continue
                bulb_pos = bulb_arr[bi]
                near_u = [idx for idx in cand_u
                          if np.linalg.norm(all_pts[idx] - bulb_pos) <= dilation_radius]
                near_v = [idx for idx in cand_v
                          if np.linalg.norm(all_pts[idx] - bulb_pos) <= dilation_radius]
                if not near_u and not near_v:
                    continue
                vertices[next_vid] = bulb_pos
                for idx in near_u + near_v:
                    pt_to_vertex[idx] = next_vid
                if len(face_merged_vids[u]) < MAX_CORNERS:
                    face_merged_vids[u].add(next_vid)
                    if u in DEBUG_FACES:
                        print(f"[BORN] vid={next_vid} @ {bulb_pos} | stage=7-edge-bulb bi={bi} | edge ({u},{v}) | assigned to face {u}")
                elif u in DEBUG_FACES:
                    print(f"[SKIP] vid={next_vid} @ {bulb_pos} | stage=7-edge-bulb | face {u} at MAX_CORNERS")
                if len(face_merged_vids[v]) < MAX_CORNERS:
                    face_merged_vids[v].add(next_vid)
                    if v in DEBUG_FACES:
                        print(f"[BORN] vid={next_vid} @ {bulb_pos} | stage=7-edge-bulb bi={bi} | edge ({u},{v}) | assigned to face {v}")
                elif v in DEBUG_FACES:
                    print(f"[SKIP] vid={next_vid} @ {bulb_pos} | stage=7-edge-bulb | face {v} at MAX_CORNERS")
                used_bulbs.add(int(bi))
                next_vid += 1
                cand_u = [idx for idx in cand_u if idx not in pt_to_vertex]
                cand_v = [idx for idx in cand_v if idx not in pt_to_vertex]

        # ── Greedy pairwise match of leftover u/v corners ────────────
        # After creating a vertex, sweep leftover candidates and absorb
        # any that sit within dilation_radius of the new vertex — handles
        # the common case where polygon approximation emits duplicate
        # corners at the same (r, c) on the same face.
        while cand_u and cand_v:
            d_mat = np.linalg.norm(
                all_pts[cand_u][:, None] - all_pts[cand_v][None, :],
                axis=2,
            )
            flat_idx = int(np.argmin(d_mat))
            i, j = divmod(flat_idx, d_mat.shape[1])
            if d_mat[i, j] > 2 * dilation_radius:
                break
            ui, vi = cand_u[i], cand_v[j]
            midpoint = (all_pts[ui] + all_pts[vi]) / 2.0
            vertices[next_vid] = midpoint
            pt_to_vertex[ui] = next_vid
            pt_to_vertex[vi] = next_vid
            if len(face_merged_vids[u]) < MAX_CORNERS:
                face_merged_vids[u].add(next_vid)
                if u in DEBUG_FACES:
                    print(f"[BORN] vid={next_vid} @ {midpoint} | stage=7-edge-greedy | edge ({u},{v}) | ui={ui} @ {all_pts[ui]} vi={vi} @ {all_pts[vi]} | assigned to face {u}")
            elif u in DEBUG_FACES:
                print(f"[SKIP] vid={next_vid} @ {midpoint} | stage=7-edge-greedy | face {u} at MAX_CORNERS")
            if len(face_merged_vids[v]) < MAX_CORNERS:
                face_merged_vids[v].add(next_vid)
                if v in DEBUG_FACES:
                    print(f"[BORN] vid={next_vid} @ {midpoint} | stage=7-edge-greedy | edge ({u},{v}) | ui={ui} @ {all_pts[ui]} vi={vi} @ {all_pts[vi]} | assigned to face {v}")
            elif v in DEBUG_FACES:
                print(f"[SKIP] vid={next_vid} @ {midpoint} | stage=7-edge-greedy | face {v} at MAX_CORNERS")

            # Absorb leftover candidates (from either side) within
            # dilation_radius of the new vertex position.
            new_cand_u = []
            for idx in cand_u:
                if idx == ui:
                    continue
                if np.linalg.norm(all_pts[idx] - midpoint) <= dilation_radius:
                    pt_to_vertex[idx] = next_vid
                else:
                    new_cand_u.append(idx)
            new_cand_v = []
            for idx in cand_v:
                if idx == vi:
                    continue
                if np.linalg.norm(all_pts[idx] - midpoint) <= dilation_radius:
                    pt_to_vertex[idx] = next_vid
                else:
                    new_cand_v.append(idx)
            cand_u, cand_v = new_cand_u, new_cand_v
            next_vid += 1

    # 7b. Isolated-bulb sweep. Bulbs are hard truth — any bulb not yet
    # materialized always becomes a vertex at its exact centre. Absorb
    # nearby unclaimed corners; also re-home already-claimed corners
    # within dilation_radius so they point to the bulb vertex instead.
    if len(bulb_arr) > 0:
        bulb_rc = np.round(bulb_arr).astype(int)
        for bi in range(len(bulb_arr)):
            if bi in used_bulbs:
                continue
            br, bc = bulb_rc[bi]
            if not (0 <= br < H and 0 <= bc < W):
                continue
            faces_here = [fid for fid in valid_labels
                          if dilated_masks[fid][br, bc]]
            if not faces_here:
                continue
            bulb_pos = bulb_arr[bi]
            # Always create the vertex — bulbs are ground truth
            vertices[next_vid] = bulb_pos
            absorbed = []
            for fid in faces_here:
                for idx in face_pt_indices[fid]:
                    idx = int(idx)
                    if np.linalg.norm(all_pts[idx] - bulb_pos) <= dilation_radius:
                        pt_to_vertex[idx] = next_vid
                        absorbed.append(idx)
            for fid in faces_here:
                face_merged_vids[fid].add(next_vid)
                if not any(pt_to_face[idx] == fid for idx in absorbed):
                    face_extra_vids[fid].append(next_vid)
                if fid in DEBUG_FACES:
                    print(f"[BORN] vid={next_vid} @ {bulb_pos} | stage=7b-isolated-bulb bi={bi} | assigned to face {fid} | faces_here={faces_here} | extra={not any(pt_to_face[idx] == fid for idx in absorbed)}")
            used_bulbs.add(bi)
            next_vid += 1

    # 7c. Merge nearby vertices along face chains. After cycles and
    # edge passes, an N-face chain meeting at one point may have produced
    # separate vertices for each edge pair. Merge any two vertices within
    # dilation_radius whose owning faces are connected through adjacency.
    if len(vertices) > 1:
        merge_map = {}  # old_vid -> canonical_vid
        vert_ids = sorted(vertices.keys())
        vert_arr = np.array([vertices[vid] for vid in vert_ids])
        id_to_idx = {vid: i for i, vid in enumerate(vert_ids)}

        for i, vi in enumerate(vert_ids):
            if vi in merge_map:
                continue
            # find all vids within dilation_radius of vi
            dists = np.linalg.norm(vert_arr - vert_arr[i], axis=1)
            nearby = [vert_ids[j] for j in range(len(vert_ids))
                      if j != i and dists[j] <= dilation_radius
                      and vert_ids[j] not in merge_map]
            for vj in nearby:
                # Check: do vi and vj share an adjacent face path?
                # (faces owning vi and vj should overlap or be neighbors)
                faces_i = {fid for fid in valid_labels if vi in face_merged_vids[fid]}
                faces_j = {fid for fid in valid_labels if vj in face_merged_vids[fid]}
                connected = bool(faces_i & faces_j)
                if not connected:
                    for fi in faces_i:
                        for fj in faces_j:
                            if G.has_edge(fi, fj):
                                connected = True
                                break
                        if connected:
                            break
                if connected:
                    merge_map[vj] = vi

        # Apply merges
        if merge_map:
            for old_vid, new_vid in merge_map.items():
                # Transfer face ownership
                for fid in valid_labels:
                    if old_vid in face_merged_vids[fid]:
                        face_merged_vids[fid].discard(old_vid)
                        face_merged_vids[fid].add(new_vid)
                        if fid in DEBUG_FACES:
                            print(f"[MERGE] stage=7c-nearby-merge | face {fid}: vid {old_vid} @ {vertices.get(old_vid)} -> {new_vid} @ {vertices.get(new_vid)}")
                    if old_vid in face_extra_vids[fid]:
                        face_extra_vids[fid] = [
                            new_vid if v == old_vid else v
                            for v in face_extra_vids[fid]
                        ]
                # Remap corner assignments
                for idx in range(len(all_pts)):
                    if idx in pt_to_vertex and pt_to_vertex[idx] == old_vid:
                        pt_to_vertex[idx] = new_vid
                del vertices[old_vid]

    # 8. Snap remaining unclaimed corners to existing vertices when
    # close enough. This handles path-junction cases where 3 faces meet
    # at a corner but the face adjacency graph is missing one of the
    # three edges (e.g. 4-7-5, 8-14-30), so no 3-cycle claims the
    # junction. The first edge pass creates a vertex from 2 of the 3
    # faces; stray corners from the third face would otherwise become
    # singleton duplicates at the same spot.
    if vertices:
        vert_ids = list(vertices.keys())
        vert_arr = np.array([vertices[vid] for vid in vert_ids])
        for idx in range(len(all_pts)):
            if idx in pt_to_vertex:
                continue
            d = np.linalg.norm(vert_arr - all_pts[idx], axis=1)
            j = int(np.argmin(d))
            if d[j] <= dilation_radius:
                pt_to_vertex[idx] = vert_ids[j]
                face_merged_vids[pt_to_face[idx]].add(vert_ids[j])
                if pt_to_face[idx] in DEBUG_FACES:
                    print(f"[SNAP] stage=8-snap | face {pt_to_face[idx]}: corner idx={idx} @ {all_pts[idx]} snapped to vid={vert_ids[j]} @ {vertices[vert_ids[j]]} dist={d[j]:.1f}")

    # 8b. Topology-aware snap: for each unclaimed corner, check all
    # vertices of adjacent faces (not just shared ones). If a neighbor's
    # vertex is close, snap to it. This catches chain junctions where
    # face A's corner should merge with a vertex owned by neighbor B.
    for idx in range(len(all_pts)):
        if idx in pt_to_vertex:
            continue
        fid = pt_to_face[idx]
        pt = all_pts[idx]
        best_vid, best_dist = None, float('inf')
        for neighbor_fid in G.neighbors(fid):
            for vid in face_merged_vids[neighbor_fid]:
                if vid not in vertices:
                    continue
                d = np.linalg.norm(vertices[vid] - pt)
                if d < best_dist:
                    best_dist, best_vid = d, vid
        if best_vid is not None and best_dist <= 2 * dilation_radius:
            pt_to_vertex[idx] = best_vid
            face_merged_vids[fid].add(best_vid)
            if fid in DEBUG_FACES:
                print(f"[SNAP] stage=8b-topo-snap | face {fid}: corner idx={idx} @ {all_pts[idx]} snapped to vid={best_vid} @ {vertices[best_vid]} dist={best_dist:.1f}")

    # 8c. Pair unclaimed corners across adjacent faces. Catches shared-edge
    # junctions that the edge pass missed because the two corners are
    # slightly outside the tight mutual dilation intersection (e.g. a
    # corner that drifts perpendicular to the shared edge by more than
    # dilation_radius from the other face's mask).
    unclaimed_by_face = defaultdict(list)
    for idx in range(len(all_pts)):
        if idx not in pt_to_vertex:
            unclaimed_by_face[pt_to_face[idx]].append(idx)

    for u, v in G.edges():
        cu = unclaimed_by_face.get(u, [])
        cv = unclaimed_by_face.get(v, [])
        while cu and cv:
            d_mat = np.linalg.norm(
                all_pts[cu][:, None] - all_pts[cv][None, :], axis=2)
            flat = int(np.argmin(d_mat))
            i, j = divmod(flat, d_mat.shape[1])
            if d_mat[i, j] > 2 * dilation_radius:
                break
            ui, vi = cu[i], cv[j]
            mid = (all_pts[ui] + all_pts[vi]) / 2.0
            vertices[next_vid] = mid
            pt_to_vertex[ui] = next_vid
            pt_to_vertex[vi] = next_vid
            face_merged_vids[u].add(next_vid)
            face_merged_vids[v].add(next_vid)
            if u in DEBUG_FACES or v in DEBUG_FACES:
                print(f"[BORN] vid={next_vid} @ {mid} | stage=8c-unclaimed-pair | edge ({u},{v}) | ui={ui} @ {all_pts[ui]} vi={vi} @ {all_pts[vi]} | assigned to face {u if u in DEBUG_FACES else v}")
            cu = [x for x in cu if x not in pt_to_vertex]
            cv = [x for x in cv if x not in pt_to_vertex]
            unclaimed_by_face[u] = cu
            unclaimed_by_face[v] = cv
            next_vid += 1

    # 9. Singleton vertices for unclaimed corners. Skip if:
    #    - face already has MAX_CORNERS merged vertices, OR
    #    - the point sits on the segment between two already-assigned
    #      vertices (polygon-approximation noise along an edge).
    def _point_on_segment(pt, a, b, tol):
        """True if pt is within tol of segment a-b."""
        ab = b - a
        seg_len = np.linalg.norm(ab)
        if seg_len < 1e-6:
            return np.linalg.norm(pt - a) <= tol
        t = np.dot(pt - a, ab) / (seg_len * seg_len)
        t = np.clip(t, 0, 1)
        proj = a + t * ab
        return np.linalg.norm(pt - proj) <= tol

    for idx in range(len(all_pts)):
        if idx not in pt_to_vertex:
            fid = pt_to_face[idx]
            if len(face_merged_vids[fid]) >= MAX_CORNERS:
                continue
            # Check if this point lies between two already-assigned vids
            existing = [vid for vid in face_merged_vids[fid] if vid in vertices]
            pt = all_pts[idx]
            between = False
            for i_v in range(len(existing)):
                for j_v in range(i_v + 1, len(existing)):
                    if _point_on_segment(pt, vertices[existing[i_v]],
                                         vertices[existing[j_v]],
                                         dilation_radius):
                        between = True
                        break
                if between:
                    break
            if between:
                continue
            vertices[next_vid] = all_pts[idx]
            pt_to_vertex[idx] = next_vid
            face_merged_vids[fid].add(next_vid)
            if fid in DEBUG_FACES:
                print(f"[BORN] vid={next_vid} @ {all_pts[idx]} | stage=9-singleton | face {fid} | idx={idx}")
            next_vid += 1

    # Any still-unclaimed corners need a mapping so face_vertices
    # build doesn't KeyError — snap them to the nearest vertex that
    # already belongs to the same face (not any global vertex).
    for idx in range(len(all_pts)):
        if idx not in pt_to_vertex:
            fid = pt_to_face[idx]
            face_vids = list(face_merged_vids.get(fid, set()))
            if face_vids:
                face_pos = np.array([vertices[vid] for vid in face_vids])
                d = np.linalg.norm(face_pos - all_pts[idx], axis=1)
                pt_to_vertex[idx] = face_vids[int(np.argmin(d))]
            elif vertices:
                vert_ids_final = list(vertices.keys())
                vert_arr_final = np.array([vertices[vid] for vid in vert_ids_final])
                d = np.linalg.norm(vert_arr_final - all_pts[idx], axis=1)
                pt_to_vertex[idx] = vert_ids_final[int(np.argmin(d))]



    # 10. Build face_vertices, preserving original corner order, dedup'd.
    # Enforce MAX_CORNERS at the final step: bulb-anchored / cycle-core
    # vertices (earliest vids) always win; any extras past the cap are
    # simply dropped — this is the condition the bulb-first pass relies
    # on to discard redundant cluster duplicates.
    face_vertices = {}
    for fid in valid_labels:
        seen = []
        for idx in face_pt_indices[fid]:
            vid = pt_to_vertex[int(idx)]
            if vid in seen:
                continue
            seen.append(vid)
        for vid in face_extra_vids[fid]:
            if vid in seen:
                continue
            seen.append(vid)
        if len(seen) > MAX_CORNERS:
            keep = set(sorted(seen)[:MAX_CORNERS])
            seen = [v for v in seen if v in keep]
        face_vertices[fid] = seen
    for face, vert in face_vertices.items():
        print(f'{face} -> {vert} \n')
    return vertices, face_vertices, all_pts, face_pt_indices



def filter_edge_faces(vertices, face_vertices, valid_labels, face_pixels,
                      labeled_clean, img_shape, margin=200):
    """
    Drop any merged vertex that sits within `margin` px of the bottom or
    left edge of the image (col < margin OR row > H - margin), and drop
    any face that references one of those vertices.

    Returns (vertices, face_vertices, valid_labels, face_pixels,
             labeled_clean) with edge-touching entries removed.
    `labeled_clean` is returned with dropped face labels zeroed out.
    """
    H, W = img_shape[:2]

    bad_vids = {vid for vid, pos in vertices.items()
                if pos[1] < margin or pos[0] > H - margin}

    bad_faces = {fid for fid, vids in face_vertices.items()
                 if any(v in bad_vids for v in vids)}

    new_face_vertices = {fid: vids for fid, vids in face_vertices.items()
                         if fid not in bad_faces}
    new_valid_labels = [fid for fid in valid_labels if fid not in bad_faces]
    new_face_pixels = {fid: px for fid, px in face_pixels.items()
                       if fid not in bad_faces}

    # Keep only vertices still referenced by a surviving face.
    used_vids = {v for vids in new_face_vertices.values() for v in vids}
    new_vertices = {vid: pos for vid, pos in vertices.items()
                    if vid in used_vids}

    new_labeled = labeled_clean.copy()
    for fid in bad_faces:
        new_labeled[new_labeled == fid] = 0

    return (new_vertices, new_face_vertices, new_valid_labels,
            new_face_pixels, new_labeled)


def detect_bulb_corners(img, brightness_thresh=240, min_size=5, max_size=200):
    gray = color.rgb2gray(img)
    bright = gray > (brightness_thresh / 255.0)
    labeled_bulbs = measure.label(bright, connectivity=1)
    corners = []
    for region in regionprops(labeled_bulbs):
        if min_size < region.area < max_size:
            corners.append(region.centroid)  # (row, col)
    return np.array(corners)