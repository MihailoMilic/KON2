import cv2
from matplotlib import contour
from matplotlib.patches import Polygon
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
from skimage import img_as_ubyte, measure, morphology
from skimage.measure import approximate_polygon, find_contours, regionprops
from skimage import color
from shapely.geometry import Polygon
from skimage.segmentation import find_boundaries
from skimage.draw import disk
def extract_face_masks(img, min_face_size=500, dilation_radius=1):
    """
    Extract individual face masks from a grayscale mesh image.
    Uses CLAHE + Canny + morphological close + skeletonize
    to create clean barriers, then labels enclosed regions.

    Returns:
        labeled_clean: 2D int array, each pixel labelled with its face ID
        valid_labels:  list of face IDs that are large enough to be real faces
        face_pixels:   dict {face_id: array of [row, col] pixel coords}
    """
    gray = (color.rgb2gray(img) * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(gray)

    edge_mask = cv2.Canny(img_eq, threshold1=60, threshold2=120)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)

    skeleton = morphology.skeletonize(closed > 0)

    bright_mask = (img_eq > 220).astype(np.uint8)
    bright_mask = cv2.dilate(bright_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)))
    img_eq = cv2.inpaint(img_eq, bright_mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
        # After skeleton is applied
    img_eq[skeleton] = 0

    # Now proceed with threshold
    img_eq[img_eq < 150] = 0

    not_black = img_eq > 0
    not_black = morphology.remove_small_objects(not_black, min_size=200)
    labeled_clean = measure.label(not_black, connectivity=1)

    valid_labels = [
        i for i in range(1, labeled_clean.max() + 1)
        if (labeled_clean == i).sum() > min_face_size
    ]

    face_pixels = {i: np.argwhere(labeled_clean == i) for i in valid_labels}

    return labeled_clean, valid_labels, face_pixels

from shapely.geometry import Polygon as ShapelyPolygon

def extract_face_corners(labeled_clean, face_pixels, tolerance=15, small_threshold=200, dilation_radius=15):
    face_corners = {}
    selem = morphology.disk(dilation_radius)
    
    for i, pixels in face_pixels.items():
        mask = (labeled_clean == i)
        area = mask.sum()
        
        if area < small_threshold:
            mask = morphology.binary_dilation(mask, selem)
        
        contours = find_contours(mask, 0.5)
        if not contours:
            continue
        contour = max(contours, key=len)
        contour = contour.astype(np.float32)
        approx = approximate_polygon(contour, tolerance=tolerance)
        approx = approx.reshape(-1, 2)
        
        if area < small_threshold:
            # shrink polygon back — note approx is (row, col) so flip for Shapely (x, y)
            poly = ShapelyPolygon(approx[:, ::-1])
            shrunk = poly.buffer(-dilation_radius, join_style=2)
            if shrunk.is_empty or shrunk.geom_type == 'MultiPolygon':
                # fallback: keep dilated version scaled toward centroid
                cx, cy = np.mean(approx, axis=0)
                equiv_r = np.sqrt(area / np.pi)
                scale = equiv_r / (equiv_r + dilation_radius)
                approx = (approx - [cx, cy]) * scale + [cx, cy]
            else:
                coords = np.array(shrunk.exterior.coords)
                approx = coords[:, ::-1]  # back to (row, col)
        
        face_corners[i] = approx
    return face_corners
def compute_adjacency(labeled_clean, valid_labels, face_pixels,
                      shared_border_threshold=20):
    n = len(valid_labels)
    adjacency = np.zeros((n, n), dtype=bool)
    face_centroids = {i: pixels.mean(axis=0) for i, pixels in face_pixels.items()}

    selem = morphology.disk(5)

    background_faces = set()

    for a, i in enumerate(valid_labels):
        mask_i = (labeled_clean == i)
        dilated_i = morphology.binary_dilation(mask_i, selem)
        border_i = dilated_i & ~mask_i

        border_labels = labeled_clean[border_i]

        for b, j in enumerate(valid_labels):
            if j <= i:
                continue
            count = np.sum(border_labels == j)
            if (i==17 and j ==22) or (i==22 and j==17) or (i==22 and j==31) or (i==31 and j==22):
                print(f"Border between face {i} and {j}: {count} pixels")
            if count > shared_border_threshold:
                adjacency[a, b] = True
                adjacency[b, a] = True

    adjacent_faces = {i: [] for i in valid_labels}
    for a, i in enumerate(valid_labels):
        for b, j in enumerate(valid_labels):
            if adjacency[a, b]:
                adjacent_faces[i].append(j)

    return adjacency, adjacent_faces, face_centroids
def merge_vertices(face_corners, valid_labels, adjacency, merges_per_pair=2):
    """
    For each adjacent face pair, merge the closest corner pairs using Union-Find.
    Each shared edge contributes 2 shared vertices.

    Returns:
        vertices:        dict {vertex_id: [row, col] averaged position}
        face_vertices:   dict {face_id: [list of vertex_ids]}
        all_pts:         flat array of all corner points
        face_pt_indices: dict {face_id: [indices into all_pts]}
    """
    # Build flat list of all corner points
    all_pts = []
    face_pt_indices = {}
    idx = 0
    for i, corners in face_corners.items():
        face_pt_indices[i] = []
        for pt in corners:
            all_pts.append(pt)
            face_pt_indices[i].append(idx)
            idx += 1
    all_pts = np.array(all_pts)

    # Union-Find
    parent = list(range(len(all_pts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    # Merge closest pairs per adjacent face pair
    for a, i in enumerate(valid_labels):
        for b, j in enumerate(valid_labels):
            if not adjacency[a, b] or j <= i:
                continue

            i_indices = face_pt_indices[i]
            j_indices = face_pt_indices[j]

            pairs = sorted([
                (np.linalg.norm(all_pts[pi] - all_pts[pj]), pi, pj)
                for pi in i_indices
                for pj in j_indices
            ])

            used_i, used_j = set(), set()
            merged = 0
            for dist, pi, pj in pairs:
                if pi in used_i or pj in used_j:
                    continue
                union(pi, pj)
                used_i.add(pi)
                used_j.add(pj)
                merged += 1
                if merged == merges_per_pair:
                    break

    # Group points by their root
    groups = defaultdict(list)
    for i in range(len(all_pts)):
        groups[find(i)].append(i)

    # Average positions within each group
    vertices = {}
    pt_to_vertex = {}
    vertex_id = 1
    for root, members in groups.items():
        vertices[vertex_id] = all_pts[members].mean(axis=0)
        for m in members:
            pt_to_vertex[m] = vertex_id
        vertex_id += 1

    # Remap face corners to vertex IDs, deduplicating within each face
    face_vertices = {}
    for i, corners in face_corners.items():
        vids = []
        for k in range(len(corners)):
            vid = pt_to_vertex[face_pt_indices[i][k]]
            if vid not in vids:
                vids.append(vid)
        face_vertices[i] = vids

    return vertices, face_vertices, all_pts, face_pt_indices
def detect_bulb_corners(img, brightness_thresh=240, min_size=5, max_size=200):
    gray = color.rgb2gray(img)
    bright = gray > (brightness_thresh / 255.0)
    labeled_bulbs = measure.label(bright, connectivity=1)
    corners = []
    for region in regionprops(labeled_bulbs):
        if min_size < region.area < max_size:
            corners.append(region.centroid)  # (row, col)
    return np.array(corners)

def fill_bulb_gaps(labeled_clean, bulb_corners, radius=10):
    filled = labeled_clean.copy()
    
    for corner in bulb_corners:
        r, c = int(round(corner[0])), int(round(corner[1]))
        rr, cc = disk((r, c), radius, shape=labeled_clean.shape)
        
        # Only fill background pixels
        zero_mask = filled[rr, cc] == 0
        if not np.any(zero_mask):
            continue
            
        # For each zero pixel, find nearest non-zero pixel's label
        zero_coords = np.column_stack((rr[zero_mask], cc[zero_mask]))
        nonzero_mask = filled[rr, cc] > 0
        if not np.any(nonzero_mask):
            continue
        nonzero_coords = np.column_stack((rr[nonzero_mask], cc[nonzero_mask]))
        nonzero_labels = filled[nonzero_coords[:, 0], nonzero_coords[:, 1]]
        
        dists = cdist(zero_coords, nonzero_coords)
        nearest = dists.argmin(axis=1)
        
        for idx, (pr, pc) in enumerate(zero_coords):
            filled[pr, pc] = nonzero_labels[nearest[idx]]
    
    return filled

def find_dominant_direction(mask, angle_steps=180):
    """
    Sweep angles, project mask onto each direction, find the angle
    where cross-section lengths have minimum variance (most uniform).
    Returns: angle in radians, and the two edge lengths (along, across)
    """
    rows, cols = np.where(mask)
    if len(rows) < 5:
        return None, None, None
    
    points = np.column_stack([cols, rows]).astype(np.float64)  # (x, y)
    best_angle = 0
    best_score = np.inf
    best_lengths = None
    
    for deg in range(0, angle_steps):
        theta = np.radians(deg)
        direction = np.array([np.cos(theta), np.sin(theta)])
        perp = np.array([-np.sin(theta), np.cos(theta)])
        
        # project onto direction and perpendicular
        proj_along = points @ direction
        proj_perp = points @ perp
        
        # bin along the perpendicular axis — each bin is one "slice"
        bins = np.round(proj_perp).astype(int)
        unique_bins = np.unique(bins)
        
        if len(unique_bins) < 3:
            continue
        
        # measure width of each slice
        widths = []
        for b in unique_bins:
            slice_proj = proj_along[bins == b]
            widths.append(slice_proj.max() - slice_proj.min())
        
        widths = np.array(widths)
        # trim top/bottom 10% to ignore ragged edges
        trim = max(1, len(widths) // 10)
        widths_trimmed = np.sort(widths)[trim:-trim] if len(widths) > 2 * trim else widths
        
        score = np.std(widths_trimmed) / (np.mean(widths_trimmed) + 1e-6)  # CV
        
        if score < best_score:
            best_score = score
            best_angle = theta
            best_lengths = (np.mean(widths_trimmed), len(unique_bins))
    
    return best_angle, best_score, best_lengths


def polygon_from_direction(mask, angle, area):
    """
    Given dominant angle, build a rectangle from the bounding box 
    in that rotated frame.
    """
    rows, cols = np.where(mask)
    points = np.column_stack([cols, rows]).astype(np.float64)
    
    direction = np.array([np.cos(angle), np.sin(angle)])
    perp = np.array([-np.sin(angle), np.cos(angle)])
    
    proj_along = points @ direction
    proj_perp = points @ perp
    
    # oriented bounding box
    a_min, a_max = proj_along.min(), proj_along.max()
    p_min, p_max = proj_perp.min(), proj_perp.max()
    
    # 4 corners in original coords
    corners = np.array([
        direction * a_min + perp * p_min,
        direction * a_max + perp * p_min,
        direction * a_max + perp * p_max,
        direction * a_min + perp * p_max,
    ])
    
    # back to (row, col)
    return corners[:, ::-1]