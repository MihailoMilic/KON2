import cv2
from matplotlib import contour
from matplotlib.patches import Polygon
from networkx import radius
from numpy.compat import Path
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
def _has_green_border(img, lo=300, hi=50000):
    """Detect a bright green border bleed between neighbouring holes.

    Covers two border tones observed in the dataset:
      * hole_081-style yellow-green lime (H~42, S=255, V~220)
      * hole_012-style pale pure green    (H~60, S~100, V~210)
    Both share: green channel noticeably above both R and B, and
    reasonably bright. Criterion: G > R+25 AND G > B+25 AND V >= 170.

    Band-pass by count: the raw pixel count of green-dominant pixels
    separates three regimes cleanly across the 100-hole dataset:
      - <300  → no bleed, clean image (hole_083's non-green-faced sibs)
      - [300, 50000] → thin border strip bleeding in (hole_081,
        hole_012, hole_091 and ~10 others)
      - >50000 → the mesh FACES themselves are green (hole_083,
        hole_024) — not a bleed, leave it to the skeleton path.

    When True → caller should fall back to the CLAHE grayscale path,
    which was immune to the border artifact.
    """
    img_u8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    V = hsv[:, :, 2]
    R = img_u8[:, :, 0].astype(int)
    G = img_u8[:, :, 1].astype(int)
    B = img_u8[:, :, 2].astype(int)
    n = int(((G > R + 25) & (G > B + 25) & (V >= 170)).sum())
    return lo <= n <= hi


def _extract_grayscale_clahe(img, min_face_size):
    """Original grayscale path: CLAHE-equalised gray + Canny skeleton +
    bright-dot inpainting + threshold at 150. Kept as a fallback for
    images where the lime-green inter-hole border confuses the simpler
    skeleton-cut path (hole_081 family).
    """
    gray = (color.rgb2gray(img) * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(gray)

    edge_mask = cv2.Canny(img_eq, threshold1=75, threshold2=120)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
    skeleton = morphology.skeletonize(closed > 0)

    # Amplify the 1px green inter-hole border directly from RGB — rgb2gray
    # collapses its luminance into the face interior so Canny loses it in
    # patches (hole_030 face 1). Reuse the same predicate as _has_green_border.
    img_u8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    R = img_u8[:, :, 0].astype(int)
    G = img_u8[:, :, 1].astype(int)
    B = img_u8[:, :, 2].astype(int)
    green_border = (G > R + 25) & (G > B + 25) & (hsv[:, :, 2] >= 170)

    bright_mask = (img_eq > 220).astype(np.uint8)
    bright_mask = cv2.dilate(bright_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    img_eq = cv2.inpaint(img_eq, bright_mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
    img_eq[skeleton] = 0
    img_eq[green_border] = 0
    img_eq[img_eq < 150] = 0

    not_black = img_eq > 0
    not_black = morphology.remove_small_objects(not_black, min_size=200)
    labeled_clean = measure.label(not_black, connectivity=1)

    valid_labels = [
        i for i in range(1, labeled_clean.max() + 1)
        if (labeled_clean == i).sum() > min_face_size
    ]
    face_pixels = {i: np.argwhere(labeled_clean == i) for i in valid_labels}

    # Drop faces whose pixels touch the left or bottom image margin —
    # these are axis/border artifacts that are never adjacent to real faces.
    H_img, W_img = labeled_clean.shape
    EDGE_MARGIN = 200
    valid_labels = [fid for fid in valid_labels
                    if face_pixels[fid][:, 1].min() >= EDGE_MARGIN
                    and face_pixels[fid][:, 0].max() <= H_img - EDGE_MARGIN / 2]
    face_pixels = {fid: face_pixels[fid] for fid in valid_labels}

    return labeled_clean, valid_labels, face_pixels


def _extract_grayscale_skeleton(img, min_face_size):
    """CLAHE-free grayscale path: straight rgb2gray threshold + Canny
    skeleton cut. Works best on clean images without the inter-hole
    lime border (hole_083 family). Faster and preserves dim-green
    faces that CLAHE tends to muddle.
    """
    # 1. Convert to grayscale
    gray = (color.rgb2gray(img) * 255).astype(np.uint8)

    # 2. Thresholding: Red lines (Y~76) and background (Y=0) become black.
    # The green faces (Y~210) easily survive the 150 threshold.
    # This automatically separates the faces!
    gray[gray < 150] = 0

    # 3. Quick Canny pass just to ensure the red line gaps are fully respected
    edge_mask = cv2.Canny(gray, threshold1=50, threshold2=100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
    skeleton = morphology.skeletonize(closed > 0)

    # Cut along the skeleton
    gray[skeleton] = 0

    # 4. Label the surviving bright green islands
    not_black = gray > 0
    not_black = morphology.remove_small_objects(not_black, min_size=200)
    labeled_clean = measure.label(not_black, connectivity=1)

    valid_labels = [
        i for i in range(1, labeled_clean.max() + 1)
        if (labeled_clean == i).sum() > min_face_size
    ]
    face_pixels = {i: np.argwhere(labeled_clean == i) for i in valid_labels}

    # 5. Relaxed margin (10 instead of 200) so outer faces aren't deleted
    H_img, W_img = labeled_clean.shape
    EDGE_MARGIN = 200

    valid_labels = [fid for fid in valid_labels
                    if face_pixels[fid][:, 1].min() >= EDGE_MARGIN
                    and face_pixels[fid][:, 0].max() <= H_img - EDGE_MARGIN / 2]

    face_pixels = {fid: face_pixels[fid] for fid in valid_labels}

    return labeled_clean, valid_labels, face_pixels


def _extract_grayscale(img, min_face_size):
    """Grayscale-path dispatcher.

    If the image carries a green inter-hole border bleed
    (hole_081 / hole_012 family) → fall back to the CLAHE-based path,
    which is robust to that artifact. Otherwise use the faster,
    CLAHE-free skeleton-cut path that performs better on clean
    images (hole_083 family).
    """
    if _has_green_border(img):
        print("[extract_face_masks] green border detected → using CLAHE grayscale path")
        return _extract_grayscale_clahe(img, min_face_size)
    return _extract_grayscale_skeleton(img, min_face_size)


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


def _mean_nonblack_luminosity(img, margin=200):
    """Return the mean grayscale luminosity of non-black pixels,
    ignoring a `margin`-pixel frame around the image.

    Used to decide whether the grayscale (rgb2gray + threshold 150)
    path will work: dark-colored meshes (purple, dark-orange) have
    low luminance and fall below the threshold, so we need the HSV
    fallback.

    Why the margin: the outer ~200px of some inputs carry dim
    antialiased pixels / boundary noise that are *just* bright enough
    to count as non-black (>20) but darkish enough to drag the mean
    down. hole_012 in particular measures 139 uncropped (→ HSV) but
    181 once the outer 200px frame is dropped (→ grayscale, matching
    hole_003 which has the same face colors).
    """
    h, w = img.shape[:2]
    if margin > 0 and h > 2 * margin and w > 2 * margin:
        img = img[margin:h - margin, margin:w - margin]
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
    print(f"[STAGE 1a] extract_face_masks: mean non-black luminosity = {lum} "
          f"(threshold = {lum_fallback_thresh}), min_face_size={min_face_size}")
    if lum < lum_fallback_thresh:
        print(f"[STAGE 1a] path = HSV (lum < threshold)")
        _used_hsv = True
        labeled, valid, fpix = _extract_hsv(img, min_face_size)
    else:
        labeled, valid, fpix = _extract_grayscale(img, min_face_size)
        if len(valid) >= 3:
            _used_hsv = False
            print(f"[STAGE 1a] path = grayscale")
        else:
            print(f"[STAGE 1a] path = grayscale → HSV fallback ({len(valid)} faces only)")
            _used_hsv = True
            labeled, valid, fpix = _extract_hsv(img, min_face_size)

    if len(valid) > 0:
        sizes = [len(fpix[f]) for f in valid]
        print(f"[STAGE 1a] extract_face_masks: {len(valid)} faces | "
              f"sizes min/med/max = {min(sizes)}/{int(np.median(sizes))}/{max(sizes)}")
    else:
        print(f"[STAGE 1a] extract_face_masks: 0 faces")
    return labeled, valid, fpix


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
    cv2.imwrite("debug_refine.png", img_as_ubyte(not_black))
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
    n_before = len(valid_low) if hasattr(valid_low, "__len__") else None
    path = "HSV" if _used_hsv else "grayscale"
    print(f"[STAGE 1b] refine_faces: path={path}, faces before={n_before}")
    if _used_hsv:
        out = _refine_hsv(img, labeled_low, valid_low,
                          min_face_size, cannylow, cannyhigh)
    else:
        out = _refine_grayscale(img, labeled_low, valid_low,
                                min_face_size, cannylow, cannyhigh)
    # out is (labeled, valid, face_pixels)
    try:
        _, valid_after, _ = out
        print(f"[STAGE 1b] refine_faces: faces after = {len(valid_after)}")
    except Exception:
        pass
    return out


def extract_face_corners(labeled_clean, face_pixels, tolerance=4):
    face_corners = {}
    skipped_no_contour = 0
    for i, pixels in face_pixels.items():
        mask = (labeled_clean == i)
        contours = find_contours(mask, 0.5)
        if not contours:
            skipped_no_contour += 1
            continue
        tol = 2 if len(pixels) < 150 else 4 if len(pixels) < 300 else 10
        contour = max(contours, key=len)
        approx = approximate_polygon(contour.astype(np.float32), tolerance=tolerance)
        face_corners[i] = approx.reshape(-1, 2)
    counts = [len(v) for v in face_corners.values()]
    if counts:
        print(f"[STAGE 2a] extract_face_corners (tolerance={tolerance}): "
              f"{sum(counts)} raw corners across {len(face_corners)} faces "
              f"| per-face min/med/max = {min(counts)}/{int(np.median(counts))}/{max(counts)}"
              f"{' | skipped (no contour): ' + str(skipped_no_contour) if skipped_no_contour else ''}")
    return face_corners

def filter_corners(face_corners, labeled_clean, valid_labels, radius=15):
    filtered = {}
    dropped_total = 0
    affected = 0
    for fid in valid_labels:
        mask = (labeled_clean == fid)
        kept = []
        pts = face_corners.get(fid, np.empty((0, 2)))
        for pt in pts:
            r, c = int(pt[0]), int(pt[1])
            rr, cc = disk((r, c), radius, shape=mask.shape)
            circle_area = len(rr)
            inside = mask[rr, cc].sum()
            if inside / circle_area <= 0.5:
                kept.append(pt)
        dropped = len(pts) - len(kept)
        if dropped:
            affected += 1
            dropped_total += dropped
        filtered[fid] = np.array(kept) if kept else np.empty((0, 2))
    total_after = sum(len(v) for v in filtered.values())
    print(f"[STAGE 2b] filter_corners (radius={radius}): dropped "
          f"{dropped_total} interior corners from {affected} faces "
          f"→ {total_after} corners remain")
    return filtered


def filter_flat_corners(face_corners, labeled_clean, valid_labels,
                        angle_threshold_deg=160.0,
                        sample_frac=0.33,
                        sample_min=4,
                        sample_max=15,
                        min_keep=3,
                        debug_faces=None):
    """Drop corners where the face boundary is nearly straight.

    For every candidate corner we snap it onto the face's boundary contour,
    then compare the boundary's direction shortly before and shortly after
    that point. A real corner produces a sharp angle between the two
    directions; a spurious corner (e.g. the 4th vertex a max-N detector
    adds to a triangle on a long, fairly straight edge) produces an angle
    close to 180° because the boundary barely turns.

    The check is local, scale-invariant, and uses only the face's own
    pixel mask — so it handles free-floating triangles and small quads
    alike without needing adjacency information.

    Parameters
    ----------
    angle_threshold_deg
        Drop a corner if the turn angle between the back- and fwd-direction
        vectors is >= this. 160° is conservative: a 90° quad corner gives
        ~90°, an equilateral triangle corner gives ~60°, and only corners
        where the boundary is essentially straight will exceed 160°.
    sample_frac, sample_min, sample_max
        How far along the contour to look on each side of the corner when
        estimating local direction. We use a fraction of the arc length to
        the neighbouring corner, clamped into [sample_min, sample_max]
        pixels so very small faces don't use a zero-length baseline and
        very large faces don't average over a curve.
    min_keep
        Never reduce a face below this many corners via this filter.
        If dropping would violate that, the face is left untouched.
    debug_faces
        Optional iterable of face-ids; for each, per-corner angles are
        printed.
    """
    debug_set = set(debug_faces) if debug_faces else set()
    filtered = {}
    total_dropped = 0
    affected_faces = 0
    for fid in valid_labels:
        corners = face_corners.get(fid)
        if corners is None or len(corners) < 3:
            filtered[fid] = corners if corners is not None else np.empty((0, 2))
            continue

        mask = (labeled_clean == fid)
        contours_ = find_contours(mask, 0.5)
        if not contours_:
            filtered[fid] = corners
            continue
        contour_pts = max(contours_, key=len)  # (N, 2) in (row, col)
        N = len(contour_pts)
        if N < 6:
            filtered[fid] = corners
            continue

        # Snap each corner to its nearest point on the contour, then order
        # the corners by their position along the contour so "prev"/"next"
        # are well-defined.
        dists = np.linalg.norm(
            contour_pts[None, :, :] - np.asarray(corners, dtype=float)[:, None, :],
            axis=2,
        )
        idx_on_contour = np.argmin(dists, axis=1)
        order = np.argsort(idx_on_contour)
        ordered_corners = np.asarray(corners, dtype=float)[order]
        ordered_idx = idx_on_contour[order]
        K = len(ordered_corners)

        kept = []
        dropped_log = []
        angles = []
        for k in range(K):
            i_c = int(ordered_idx[k])
            i_prev = int(ordered_idx[(k - 1) % K])
            i_next = int(ordered_idx[(k + 1) % K])

            # Forward arc lengths along the (closed) contour.
            back_gap = (i_c - i_prev) % N
            fwd_gap = (i_next - i_c) % N
            # Never sample up to or past the neighbouring corner.
            back_steps = int(np.clip(round(sample_frac * back_gap), sample_min, sample_max))
            fwd_steps = int(np.clip(round(sample_frac * fwd_gap), sample_min, sample_max))
            back_steps = max(1, min(back_steps, back_gap - 1))
            fwd_steps = max(1, min(fwd_steps, fwd_gap - 1))

            corner_point = contour_pts[i_c]
            back_point = contour_pts[(i_c - back_steps) % N]
            fwd_point = contour_pts[(i_c + fwd_steps) % N]

            v1 = back_point - corner_point
            v2 = fwd_point - corner_point
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < 1e-6 or n2 < 1e-6:
                kept.append(ordered_corners[k])
                angles.append(None)
                continue
            cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(cos_a)))
            angles.append(angle_deg)

            if angle_deg >= angle_threshold_deg:
                dropped_log.append((ordered_corners[k], angle_deg, back_steps, fwd_steps))
            else:
                kept.append(ordered_corners[k])

        # Safety: don't let this filter push a face below min_keep corners.
        if len(kept) < min_keep and K >= min_keep:
            if fid in debug_set:
                print(f"[FLAT-CORNER] face {fid}: would drop to {len(kept)} corners "
                      f"(< min_keep={min_keep}); keeping originals. angles={['%.1f' % a if a is not None else 'n/a' for a in angles]}")
            filtered[fid] = ordered_corners
            continue

        if fid in debug_set:
            for k, pt in enumerate(ordered_corners):
                ang = angles[k]
                tag = "DROP" if (ang is not None and ang >= angle_threshold_deg) else "keep"
                ang_s = f"{ang:.1f}" if ang is not None else "n/a"
                print(f"[FLAT-CORNER] face {fid} corner @ ({pt[0]:.1f},{pt[1]:.1f}) "
                      f"angle={ang_s}° -> {tag}")
        for pt, ang, bs, fs in dropped_log:
            print(f"[FLAT-CORNER] face {fid}: dropping flat corner @ "
                  f"({pt[0]:.1f},{pt[1]:.1f}) angle={ang:.1f}° "
                  f"(back_steps={bs}, fwd_steps={fs})")

        if dropped_log:
            total_dropped += len(dropped_log)
            affected_faces += 1
        filtered[fid] = np.array(kept) if kept else ordered_corners

    total_after = sum(len(v) for v in filtered.values())
    print(f"[STAGE 2c] filter_flat_corners (angle >= {angle_threshold_deg}°): "
          f"dropped {total_dropped} flat corners from {affected_faces} faces "
          f"→ {total_after} corners remain")
    return filtered

def compute_adjacency(labeled_clean, valid_labels, face_pixels,
                      shared_border_threshold=30, bulbs=None, bulb_radius=4):
    n = len(valid_labels)
    label_to_idx = {l: a for a, l in enumerate(valid_labels)}
    adjacency = np.zeros((n, n), dtype=int)
    blocked = np.zeros((n, n), dtype=bool)  # barrier-walk verdict per direction
    face_centroids = {i: pixels.mean(axis=0) for i, pixels in face_pixels.items()}

    selem = morphology.disk(7)
    RADIUS = 7
    selem_step = morphology.square(3)  # 8-connected 1-pixel step
    debug_pairs = [(25, 34), (31, 34), (23, 25)]

    H_img, W_img = labeled_clean.shape
    pad = RADIUS + 2

    # ── vectorized border counting (+ barrier-aware reach) ────────────────
    for a, i in enumerate(valid_labels):
        pix = face_pixels[i]
        if len(pix) == 0:
            continue
        r0 = max(int(pix[:, 0].min()) - pad, 0)
        r1 = min(int(pix[:, 0].max()) + pad + 1, H_img)
        c0 = max(int(pix[:, 1].min()) - pad, 0)
        c1 = min(int(pix[:, 1].max()) + pad + 1, W_img)
        lbl_crop = labeled_clean[r0:r1, c0:c1]
        mask_i = (lbl_crop == i)

        dilated_i = morphology.binary_dilation(mask_i, selem)
        border_i = dilated_i & ~mask_i

        border_labels = lbl_crop[border_i]
        labels, counts = np.unique(border_labels, return_counts=True)
        for lbl, cnt in zip(labels, counts):
            if lbl in label_to_idx and lbl != i:
                b = label_to_idx[lbl]
                adjacency[a, b] += cnt
                for di, dj in debug_pairs:
                    if i == di and lbl == dj:
                        print(f"  Border {i}→{lbl}: {cnt} pixels")

        # ── Barrier-aware reach from face i ────────────────────────────
        # BFS spread through `allowed` = background + own-face. Other
        # labeled faces block propagation. Anything seen in the frontier
        # is directly reachable; anything seen by disk(7) but NOT reached
        # here means another face was in the way.
        other_faces = (lbl_crop != 0) & ~mask_i
        allowed = ~other_faces

        reached = mask_i.copy()
        for _ in range(RADIUS):
            grown = morphology.binary_dilation(reached, selem_step) & allowed
            if grown.sum() == reached.sum():
                break
            reached = grown
        frontier = morphology.binary_dilation(reached, selem_step) & ~reached
        reachable_labels = set(
            int(l) for l in np.unique(lbl_crop[frontier])
            if l != 0 and l != i
        )

        for lbl in valid_labels:
            if lbl == i or lbl not in label_to_idx:
                continue
            b = label_to_idx[lbl]
            if adjacency[a, b] <= 0:
                continue
            if lbl in reachable_labels:
                continue
            blocked[a, b] = True
            for di, dj in debug_pairs:
                if (i == di and lbl == dj) or (i == dj and lbl == di):
                    # Identify the blocking faces: labels that sit in
                    # the disk(7) border but not in the barrier frontier.
                    disk_border_labels = set(
                        int(l) for l in np.unique(lbl_crop[border_i])
                        if l != 0 and l != i
                    )
                    blockers = disk_border_labels & set(
                        int(l) for l in np.unique(lbl_crop[other_faces])
                    ) - reachable_labels
                    # Where in the crop are blocker pixels sitting between
                    # face i and face lbl? Print a few sample coords.
                    mask_lbl = (lbl_crop == lbl)
                    if blockers and mask_lbl.any():
                        sample_pts = []
                        for bl in blockers:
                            bl_mask = (lbl_crop == bl)
                            if not bl_mask.any():
                                continue
                            ys, xs = np.where(bl_mask)
                            # translate back to full-image coords
                            sample_pts.append(
                                (bl, int(ys[0] + r0), int(xs[0] + c0))
                            )
                        print(f"  [BARRIER] {i}→{lbl}: blocked by labels {sorted(blockers)} (sample coords in full-image frame: {sample_pts})")
                    else:
                        print(f"  [BARRIER] {i}→{lbl}: raw border count {adjacency[a, b]}, but BFS frontier did not reach — reachable_labels={sorted(reachable_labels)}, disk_border_labels={sorted(disk_border_labels)}")

    adjacency = np.maximum(adjacency, adjacency.T)

    # Drop adjacency only when BOTH directions agree the path is blocked.
    # A one-sided quirk (e.g. one face's bbox-crop slightly cuts off a
    # thin neighbour) won't kill a real pair this way.
    drop = blocked & blocked.T
    if drop.any():
        for a, b in zip(*np.where(drop)):
            if a < b:
                i_lbl, j_lbl = valid_labels[a], valid_labels[b]
                prev = adjacency[a, b]
                print(f"[BLOCKED] pair {i_lbl}-{j_lbl}: dropping adjacency (raw count was {prev}, both directions blocked by another face)")
    adjacency = np.where(drop, 0, adjacency)

    # Also report one-sided blocks that did NOT cause a drop — these are
    # the cases to inspect when a real neighbour goes missing.
    one_sided = blocked ^ (blocked & blocked.T)
    for di, dj in debug_pairs:
        if di in label_to_idx and dj in label_to_idx:
            a, b = label_to_idx[di], label_to_idx[dj]
            if one_sided[a, b] or one_sided[b, a]:
                print(f"[DEBUG] Pair {di}-{dj}: one-sided barrier block (blocked[{di}→{dj}]={blocked[a,b]}, blocked[{dj}→{di}]={blocked[b,a]}) — kept because not both sides agreed")
            print(f"[DEBUG] Pair {di}-{dj}: border count (symmetrized) = {adjacency[a, b]}")
            print(f"[DEBUG] Pair {di}-{dj}: final count after boost = {adjacency[a, b]}, threshold = {shared_border_threshold}")

    # ── centroid-line cleanup ─────────────────────────────────────────────
    # Drop (i, j) whose straight line from centroid_i to centroid_j
    # passes through the pixel body of a third valid face. Catches the
    # "stacked faces" case where barrier BFS sneaks around a thin pinch
    # (e.g. 31-34 leaking past the tip of 32, 23-25 past 26).
    #
    # Guardrails — a centroid-to-centroid line is a *1-D* probe of 2-D
    # shapes, so it has well-known failure modes and we must protect
    # against them:
    #   * Endpoint trim so the i/j bodies near the centroid don't matter.
    #   * A blocker must appear on >= MIN_BLOCKER_RUN consecutive samples.
    #   * Sandwich rule: the blocker run must sit between an i-run and a
    #     j-run along the line. A run past the first/last occurrence of
    #     the other face is not "between" them.
    #   * Strong-border override: if the raw shared-border count is large
    #     (>= STRONG_BORDER_FACTOR * threshold), the faces clearly do
    #     share a substantial 2-D edge somewhere — the 1-D centroid
    #     probe is unreliable for concave / elongated faces and must not
    #     overrule that evidence (this fixes the 25-34 regression where
    #     a 100-pixel border was being dropped because the centroid line
    #     happened to cross 26 and 32).
    from skimage.draw import line as _draw_line
    H_img_, W_img_ = labeled_clean.shape
    valid_set = set(valid_labels)
    MIN_BLOCKER_RUN = 3
    STRONG_BORDER_FACTOR = 2.0
    strong_border_cutoff = int(round(shared_border_threshold * STRONG_BORDER_FACTOR))
    debug_centroid_pairs = {(25, 34), (31, 34), (23, 25), (25, 23), (34, 31), (34, 25)}

    for a, i in enumerate(valid_labels):
        ci = face_centroids[i]
        for b in range(a + 1, n):
            if adjacency[a, b] <= 0:
                continue
            j = valid_labels[b]
            cj = face_centroids[j]
            raw_count = int(adjacency[a, b])
            is_debug = (i, j) in debug_centroid_pairs or (j, i) in debug_centroid_pairs

            rr, cc = _draw_line(int(round(ci[0])), int(round(ci[1])),
                                int(round(cj[0])), int(round(cj[1])))
            trim = max(1, len(rr) // 20)
            if len(rr) > 2 * trim:
                rr = rr[trim:-trim]
                cc = cc[trim:-trim]
            inb = (rr >= 0) & (rr < H_img_) & (cc >= 0) & (cc < W_img_)
            rr, cc = rr[inb], cc[inb]
            if len(rr) == 0:
                continue
            line_lbls = labeled_clean[rr, cc]

            # Compute ordered runs once: used for both the blocker check
            # and the diagnostic print.
            runs = []  # list of (label, start_idx, end_idx_exclusive, length)
            if len(line_lbls) > 0:
                change = np.concatenate(([True], line_lbls[1:] != line_lbls[:-1]))
                run_starts = np.where(change)[0]
                run_ends = np.concatenate((run_starts[1:], [len(line_lbls)]))
                runs = [(int(line_lbls[s]), int(s), int(e), int(e - s))
                        for s, e in zip(run_starts, run_ends)]

            # Positions where face i and face j themselves appear on the line.
            i_starts = [s for (lbl, s, e, _) in runs if lbl == i]
            j_starts = [s for (lbl, s, e, _) in runs if lbl == j]
            i_ends   = [e for (lbl, s, e, _) in runs if lbl == i]
            j_ends   = [e for (lbl, s, e, _) in runs if lbl == j]

            # Find candidate blockers: a third valid face whose run length
            # is >= MIN_BLOCKER_RUN AND whose run sits strictly between an
            # i-run and a j-run along the line (sandwich rule).
            blockers = set()
            if i_starts and j_starts:
                for (lbl, s, e, length) in runs:
                    if length < MIN_BLOCKER_RUN:
                        continue
                    if lbl == 0 or lbl == i or lbl == j:
                        continue
                    if lbl not in valid_set:
                        continue
                    has_i_before = any(ie <= s for ie in i_ends)
                    has_j_after  = any(js >= e for js in j_starts)
                    has_j_before = any(je <= s for je in j_ends)
                    has_i_after  = any(isx >= e for isx in i_starts)
                    if (has_i_before and has_j_after) or (has_j_before and has_i_after):
                        blockers.add(lbl)
            else:
                # Centroids didn't hit their own face bodies on the line
                # (happens with very concave faces). Fall back to the
                # original plain-run check — no sandwich info available.
                if len(line_lbls) >= MIN_BLOCKER_RUN:
                    for (lbl, s, e, length) in runs:
                        if length < MIN_BLOCKER_RUN:
                            continue
                        if lbl == 0 or lbl == i or lbl == j:
                            continue
                        if lbl in valid_set:
                            blockers.add(lbl)

            if is_debug:
                run_brief = [(lbl, length) for (lbl, s, e, length) in runs]
                print(f"[CENTROID-LINE] {i}-{j}: raw_count={raw_count}, "
                      f"runs={run_brief}, "
                      f"i_runs@{list(zip(i_starts, i_ends))}, "
                      f"j_runs@{list(zip(j_starts, j_ends))}, "
                      f"sandwiched-blockers(run>={MIN_BLOCKER_RUN})={sorted(blockers)}")

            if not blockers:
                continue

            # Strong-border safety net. The centroid line is a 1-D slice
            # and can miss the real shared edge entirely when either face
            # is concave or elongated. A raw border count well above the
            # acceptance threshold is near-certain evidence of a real
            # shared edge — don't let the 1-D probe overrule it.
            if raw_count >= strong_border_cutoff:
                print(f"[KEEP-STRONG] pair {i}-{j}: centroid line crosses "
                      f"{sorted(blockers)} but raw border count {raw_count} "
                      f">= strong cutoff {strong_border_cutoff} "
                      f"({STRONG_BORDER_FACTOR}x threshold) — keeping adjacency")
                continue

            print(f"[BLOCKED-CENTROID] pair {i}-{j}: centroid line crosses face(s) {sorted(blockers)} — dropping adjacency (raw count was {raw_count})")
            adjacency[a, b] = 0
            adjacency[b, a] = 0

    # ── threshold ─────────────────────────────────────────────────────────
    adj_bool = adjacency > shared_border_threshold
    np.fill_diagonal(adj_bool, False)

    adjacent_faces = {i: [] for i in valid_labels}
    for a, i in enumerate(valid_labels):
        for b, j in enumerate(valid_labels):
            if adj_bool[a, b]:
                adjacent_faces[i].append(j)

    # ── summary ───────────────────────────────────────────────────────────
    accepted_pairs = int(adj_bool.sum() // 2)
    degrees = [len(v) for v in adjacent_faces.values()]
    if degrees:
        print(f"[STAGE 3 SUMMARY] compute_adjacency: "
              f"{accepted_pairs} accepted pairs "
              f"(threshold={shared_border_threshold}, strong-keep cutoff={strong_border_cutoff}) | "
              f"per-face degree min/med/max = "
              f"{min(degrees)}/{int(np.median(degrees))}/{max(degrees)}")
    else:
        print(f"[STAGE 3 SUMMARY] compute_adjacency: no faces")

    return adj_bool, adjacent_faces, face_centroids, adjacency



import numpy as np
import networkx as nx
from collections import defaultdict

def merge_vertices(face_corners, valid_labels, adjacency, face_pixels,
                               bulbs=None, dilation_radius=8, img=None):
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

    If `img` is provided, an extra late pass (stage 8d) uses the red
    annotation lines in the source image to saturate any shared edge
    (u, v) that ended up with fewer than 2 shared vertices after all
    other passes — picking the red-line pixel nearest to both faces.
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
    print(f"[STAGE 4] merge_vertices: entering with {len(valid_labels)} faces, "
          f"{len(all_pts)} raw corners, {0 if bulbs is None else len(bulbs)} bulbs, "
          f"{int(adjacency.sum() // 2)} adjacency edges")

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
    DEBUG_FACES = [25]  # face to trace
    # Per-stage vid checkpoint — at the end we diff these to report how
    # many vertices each stage contributed. Keeps the summary accurate
    # without having to thread a counter through every stage's print.
    _stage_checkpoint = {"0": next_vid}
    _stage_checkpoint["6a"] = next_vid

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
    _stage_checkpoint["6-cycle"] = next_vid

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


    _stage_checkpoint["6b"] = next_vid

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



    _stage_checkpoint["7"] = next_vid

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

    _stage_checkpoint["7b"] = next_vid

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

    _stage_checkpoint["7c"] = next_vid

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

    _stage_checkpoint["8"] = next_vid

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

    _stage_checkpoint["8d"] = next_vid

    # 8d. Red-border fallback for unsaturated adjacency edges. Every
    # shared edge (u, v) should have exactly 2 endpoint vertices shared
    # between face_merged_vids[u] and face_merged_vids[v]. When only one
    # is found (e.g. a 4-cycle like 28-19-20-27 contributes one junction
    # but the other end of edge 28-19 never got covered), try to locate
    # the missing endpoint directly from the red annotation lines in the
    # source image.
    if img is not None:
        img_u8 = img if img.dtype == np.uint8 else (img * 255).astype(np.uint8)
        hsv_full = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
        red_mask_full = (
            cv2.inRange(hsv_full, (0, 50, 50), (10, 255, 255)) |
            cv2.inRange(hsv_full, (170, 50, 50), (180, 255, 255))
        ).astype(bool)
        # Align the red mask to the (H, W) frame used by dilated_masks.
        rh, rw = red_mask_full.shape
        red_mask = np.zeros((H, W), dtype=bool)
        red_mask[:min(rh, H), :min(rw, W)] = red_mask_full[:min(rh, H), :min(rw, W)]
    else:
        red_mask = None

    def _subsample(pts, cap=2000):
        if len(pts) <= cap:
            return pts
        sel = np.random.default_rng(0).choice(len(pts), cap, replace=False)
        return pts[sel]

    for u, v in G.edges():
        shared = face_merged_vids[u] & face_merged_vids[v]
        if len(shared) >= 2:
            continue

        # ── Promote-before-spawn ─────────────────────────────────────────
        # Earlier stages sometimes register an edge vertex on only one
        # side (e.g. stage 7-edge-greedy adds a (u,v) vertex to
        # face_merged_vids[u] but not to face_merged_vids[v]). If that
        # vertex sits inside the u∩v dilated strip it is, in fact, a
        # shared-edge vertex — we just forgot to flag it as such. Adding
        # it to the other face's set here avoids the classic bug where
        # stage 8d then mints a near-duplicate red-border vertex on top
        # of it (e.g. hole_081 face 3: vid 85 on face-3 only, then
        # stage-8d adds vid 113 one pixel away, inflating a triangle to
        # 4 vertices).
        inter_uv = dilated_masks[u] & dilated_masks[v]
        one_sided = (face_merged_vids[u] ^ face_merged_vids[v])
        for svid in list(one_sided):
            if svid not in vertices:
                continue
            pos = vertices[svid]
            r, c = int(round(pos[0])), int(round(pos[1]))
            if not (0 <= r < H and 0 <= c < W):
                continue
            if not inter_uv[r, c]:
                continue
            # Inside the shared strip → promote.
            added_to = None
            if svid not in face_merged_vids[u]:
                face_merged_vids[u].add(svid)
                added_to = u
            elif svid not in face_merged_vids[v]:
                face_merged_vids[v].add(svid)
                added_to = v
            shared.add(svid)
            if u in DEBUG_FACES or v in DEBUG_FACES:
                print(f"[PROMOTE] edge ({u},{v}): vid={svid} @ [{pos[0]:.2f} {pos[1]:.2f}] "
                      f"sits in u∩v strip — added to face {added_to} "
                      f"(was one-sided, now shared {len(shared)}/2)")
        if len(shared) >= 2:
            continue

        if (len(face_merged_vids[u]) >= MAX_CORNERS
                or len(face_merged_vids[v]) >= MAX_CORNERS):
            continue
        if red_mask is None:
            print(f"[UNSATURATED] edge ({u},{v}) has {len(shared)} shared vertex — no image for red-border fallback")
            continue
        red_in_strip = inter_uv & red_mask
        if not red_in_strip.any():
            print(f"[UNSATURATED] edge ({u},{v}) has {len(shared)} shared vertex — no red border found in dilated intersection")
            continue
        rr, cc = np.where(red_in_strip)
        cand = np.column_stack([rr, cc]).astype(float)
        # Closest-to-both-faces score: sum of min-distances to face u
        # and face v pixel sets (subsampled for perf).
        u_px = _subsample(face_pixels[u].astype(float))
        v_px = _subsample(face_pixels[v].astype(float))
        d_u = cdist(cand, u_px).min(axis=1)
        d_v = cdist(cand, v_px).min(axis=1)
        cost = d_u + d_v
        # Avoid landing on top of ANY existing vertex of u or v (not just
        # the shared ones). The promote-before-spawn pass above handles
        # the common "vertex in strip" case; this guard catches the
        # remaining case where an existing vertex sits just outside the
        # strip but still within dilation_radius of the best red pixel.
        avoid_set = face_merged_vids[u] | face_merged_vids[v]
        for svid in avoid_set:
            if svid in vertices:
                d_existing = np.linalg.norm(cand - vertices[svid], axis=1)
                cost = np.where(d_existing < dilation_radius, np.inf, cost)
        if not np.isfinite(cost).any():
            print(f"[UNSATURATED] edge ({u},{v}) has {len(shared)} shared vertex — red pixels all near existing endpoint")
            continue
        best = int(np.argmin(cost))
        vertex_pos = cand[best]
        vertices[next_vid] = vertex_pos
        face_merged_vids[u].add(next_vid)
        face_merged_vids[v].add(next_vid)
        face_extra_vids[u].append(next_vid)
        face_extra_vids[v].append(next_vid)
        if u in DEBUG_FACES or v in DEBUG_FACES:
            print(f"[BORN] vid={next_vid} @ {vertex_pos} | stage=8d-red-border | edge ({u},{v}) | saturating (was {len(shared)}/2)")
        next_vid += 1

    _stage_checkpoint["9"] = next_vid

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
    # Enforce MAX_CORNERS at the final step: real-corner-backed vids
    # (those that absorbed at least one polygon corner of this face) win
    # over pure-extra cycle/boundary vertices. We achieve this by keeping
    # insertion order — face_pt_indices-backed vids are appended first,
    # face_extra_vids second — and truncating in that order. Previously
    # this step sorted by vid, which preferred early cycle vertices over
    # later stage-8c/8d/9 vertices that saturate real shared edges (e.g.
    # hole_029 face 6/22 outer boundary: stage-8c vid 60 backed real
    # corners on both sides but was dropped in favour of a spurious
    # cycle-extra vid 20 that neither face owned a corner for).
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
            seen = seen[:MAX_CORNERS]
        face_vertices[fid] = seen
    for face, vert in face_vertices.items():
        print(f'{face} -> {vert} \n')

    # ── per-stage + final summary ─────────────────────────────────────────
    _stage_checkpoint["end"] = next_vid
    stage_order = ["6a", "6-cycle", "6b", "7", "7b", "7c", "8", "8d", "9", "end"]
    born_by_stage = {}
    prev = _stage_checkpoint.get("0", 1)
    for name in stage_order:
        if name not in _stage_checkpoint:
            continue
        cur = _stage_checkpoint[name]
        if cur > prev:
            # The *checkpoint at N* marks the start of stage N, i.e. the
            # value of next_vid before any stage-N BORN runs. So vertices
            # produced by stage N live between checkpoint[N] and the next
            # checkpoint. We fold them into the stage whose checkpoint
            # they sit after.
            label_for_prev = [n for n in stage_order if _stage_checkpoint.get(n) == prev]
            key = label_for_prev[0] if label_for_prev else "?"
            born_by_stage[key] = cur - prev
        prev = cur

    vids_seen = set()
    face_vid_counts = []
    under_three = []
    for fid, vids in face_vertices.items():
        face_vid_counts.append(len(vids))
        for v in vids:
            vids_seen.add(v)
        if len(vids) < 3:
            under_three.append((fid, len(vids)))

    print(f"[STAGE 4 SUMMARY] merge_vertices: {len(vertices)} vertices total, "
          f"{len(vids_seen)} referenced by {len(face_vertices)} faces | "
          f"face vid counts min/med/max = "
          f"{min(face_vid_counts) if face_vid_counts else 0}/"
          f"{int(np.median(face_vid_counts)) if face_vid_counts else 0}/"
          f"{max(face_vid_counts) if face_vid_counts else 0}")
    print(f"[STAGE 4 SUMMARY] born-per-stage: "
          + ", ".join(f"{k}={v}" for k, v in sorted(born_by_stage.items())))
    if under_three:
        print(f"[STAGE 4 WARN] faces with <3 vertices: {under_three}")

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

    print(f"[STAGE 5] filter_edge_faces (margin={margin}): "
          f"dropped {len(bad_vids)} edge vertices and {len(bad_faces)} faces "
          f"→ {len(new_valid_labels)} faces, {len(new_vertices)} vertices remain")
    if bad_faces:
        print(f"[STAGE 5] dropped face ids: {sorted(bad_faces)}")

    return (new_vertices, new_face_vertices, new_valid_labels,
            new_face_pixels, new_labeled)


def write_nas(vertices, face_vertices, output_path, file_id=101):
    """Write a Nastran bulk data (.nas) file.

    vertices      : dict {vid: (row, col)}  — pixel coordinates
    face_vertices : dict {fid: [vid, ...]}  — 3 vids → CTRIA3, 4 vids → CQUAD4
    output_path   : destination file path
    file_id       : prop_id written into every element line (default 101)
    """
    def _fmt8(val):
        # Format a float to fit exactly in an 8-character Nastran field.
        for fmt in ("{:.5g}", "{:.4g}", "{:.3g}", "{:g}"):
            s = fmt.format(val)
            if len(s) <= 8:
                return f"{s:>8}"
        return f"{val:8.2g}"

    # Map internal vertex IDs → sequential 1-based node IDs
    vid_to_nid = {vid: i + 1 for i, vid in enumerate(sorted(vertices))}

    lines = ["BEGIN BULK"]

    # GRID rows — standard 8-char small-field format:
    # GRID  | node_id | CP(blank) | X | Y | Z
    for vid in sorted(vertices):
        nid = vid_to_nid[vid]
        row, col = vertices[vid]
        x, y = float(col), float(row)
        lines.append(
            f"{'GRID':<8}{nid:>8}{'':>8}{_fmt8(x)}{_fmt8(y)}{'0.':<8}"
        )

    # Element rows
    elem_id = 1
    for fid in sorted(face_vertices):
        vids = face_vertices[fid]
        n = len(vids)
        if n not in (3, 4):
            print(f"[NAS] skipping face {fid}: {n} vertices (need 3 or 4)")
            continue
        nids = [vid_to_nid[v] for v in vids]
        nid_fields = "".join(f"{n:>8}" for n in nids)
        tag = "CQUAD4" if n == 4 else "CTRIA3"
        lines.append(f"{tag:<8}{elem_id:>8}{file_id:>8}{nid_fields}")
        elem_id += 1

    lines.append("ENDDATA")

    content = "\n".join(lines) + "\n"
    Path(output_path).write_text(content)
    print(
        f"[NAS] wrote {len(vid_to_nid)} nodes, {elem_id - 1} elements "
        f"→ {output_path}"
    )
    return content


def extract_scale_reference(img, vertical_units=75.0, horizontal_units=100.0):
    """Detect the magenta L-shaped scale bars drawn on the image and
    return the pixel→world calibration.

    The image carries two straight magenta bars: a vertical bar on the
    left (representing `vertical_units` world units) and a horizontal
    bar along the bottom (representing `horizontal_units` world units).
    Their intersection (the corner of the L) is the world origin (0, 0).

    Returns a dict:
        origin_rc     : (row, col) float pixel coords of the L-corner
        horizontal_px : float, pixel length of the horizontal bar
        vertical_px   : float, pixel length of the vertical bar
        px_per_x      : pixels per world x-unit
        px_per_y      : pixels per world y-unit
    """
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("extract_scale_reference expects an RGB image")
    img_u8 = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    magenta = cv2.inRange(hsv, (130, 30, 30), (175, 255, 255)) > 0

    if not magenta.any():
        raise RuntimeError("no magenta pixels found — cannot extract scale reference")

    # The bars are by far the longest straight magenta runs in the image,
    # so the single column / row with the most magenta pixels is each bar.
    col_counts = magenta.sum(axis=0)
    row_counts = magenta.sum(axis=1)
    v_col = int(np.argmax(col_counts))
    h_row = int(np.argmax(row_counts))

    v_rows = np.where(magenta[:, v_col])[0]
    h_cols = np.where(magenta[h_row, :])[0]
    if len(v_rows) < 2 or len(h_cols) < 2:
        raise RuntimeError("magenta bars too short to measure")

    # Bar extents — use floats throughout so no rounding happens on the
    # lengths or the derived pixels-per-unit ratios.
    v_top = float(v_rows.min())
    v_bot = float(v_rows.max())
    h_left = float(h_cols.min())
    h_right = float(h_cols.max())

    vertical_px = v_bot - v_top
    horizontal_px = h_right - h_left

    # Origin = L-corner: the vertical bar's column meets the horizontal
    # bar's row. Report as floats (no casting back to int).
    origin_rc = (float(h_row), float(v_col))

    px_per_x = horizontal_px / float(horizontal_units)
    px_per_y = vertical_px / float(vertical_units)

    print(f"[SCALE] origin(row,col)=({origin_rc[0]:.1f},{origin_rc[1]:.1f})  "
          f"horiz={horizontal_px:.2f}px/{horizontal_units}u  "
          f"vert={vertical_px:.2f}px/{vertical_units}u  "
          f"px_per_x={px_per_x:.4f}  px_per_y={px_per_y:.4f}")

    return {
        "origin_rc": origin_rc,
        "horizontal_px": horizontal_px,
        "vertical_px": vertical_px,
        "px_per_x": px_per_x,
        "px_per_y": px_per_y,
    }


def convert_vertices_to_world(vertices, scale):
    """Convert pixel-space vertices `{vid: (row, col)}` into world-space
    coordinates using the calibration dict from `extract_scale_reference`.

    Returns `{vid: (y_world, x_world)}` — tuple order stays (row-like,
    col-like) so downstream writers that treat index 1 as X and index 0
    as Y keep working. World +x points right, +y points up.
    """
    origin_r, origin_c = scale["origin_rc"]
    px_per_x = scale["px_per_x"]
    px_per_y = scale["px_per_y"]
    out = {}
    for vid, rc in vertices.items():
        row, col = float(rc[0]), float(rc[1])
        x_world = (col - origin_c) / px_per_x
        y_world = (origin_r - row) / px_per_y
        out[vid] = (y_world, x_world)
    return out


def detect_bulb_corners(img, brightness_thresh=240, min_size=5, max_size=200):
    gray = color.rgb2gray(img)
    bright = gray > (brightness_thresh / 255.0)
    labeled_bulbs = measure.label(bright, connectivity=1)
    corners = []
    rejected_small = 0
    rejected_large = 0
    for region in regionprops(labeled_bulbs):
        if region.area <= min_size:
            rejected_small += 1
            continue
        if region.area >= max_size:
            rejected_large += 1
            continue
        corners.append(region.centroid)  # (row, col)
    print(f"[STAGE 0] detect_bulb_corners (brightness>={brightness_thresh}, "
          f"size in ({min_size},{max_size})): {len(corners)} bulbs, "
          f"rejected {rejected_small} too-small + {rejected_large} too-large")
    return np.array(corners)