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

    edge_mask = cv2.Canny(img_eq, threshold1=75, threshold2=120)

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
def refine_faces(img, labeled_low, valid_low, face_pixels_low, min_face_size=500):
    gray = (color.rgb2gray(img) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(gray)

    edge_mask = cv2.Canny(img_eq, 300, 400, apertureSize=3, L2gradient=True)
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

    # Match: for each low face, find the high face with max overlap
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

        # Skip if high face is much larger (merged faces)
        if mask_high.sum() > 1.6 * mask_low.sum():
            refined[mask_low] = fid
        else:
            refined[mask_high] = fid
            used_high.add(best)

    valid_labels = valid_low
    face_pixels = {i: np.argwhere(refined == i) for i in valid_labels}
    return refined, valid_labels, face_pixels

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
    debug_pairs = [(23, 25), (15,25)]

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

    return adj_bool, adjacent_faces, face_centroids
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

def find_all_vertices(labeled_clean, valid_labels, face_pixels, bulbs=None, D_max=12, boundary_tolerance=12):
    """
    Phase 1: Junction detection (topology)
    Phase 2: Clustering
    Phase 3: Face ring assignment
    """
    h, w = labeled_clean.shape
    label_set = set(valid_labels)
    padded = np.pad(labeled_clean, 1, mode='constant', constant_values=0)

    # ── Phase 1: junction pixels ──────────────────────────────────────
    junction_pixels = []
    junction_labels = []

    for r in range(1, h + 1):
        for c in range(1, w + 1):
            patch = padded[r-1:r+2, c-1:c+2]
            labels = set(patch.flat)
            faces = labels & label_set
            has_bg = 0 in labels

            is_junction = False
            if len(faces) >= 3:
                is_junction = True
            elif len(faces) >= 2 and has_bg:
                is_junction = True

            if is_junction:
                junction_pixels.append((r - 1, c - 1))
                junction_labels.append(labels & (label_set | {0}))

    # ── Phase 1b: boundary supplement (solo turns) ────────────────────
    mesh_mask = (labeled_clean > 0).astype(float)
    for contour in find_contours(mesh_mask, 0.5):
        approx = approximate_polygon(contour, tolerance=boundary_tolerance)
        for pt in approx:
            pr, pc = int(round(pt[0])), int(round(pt[1]))
            pr = np.clip(pr, 0, h - 1)
            pc = np.clip(pc, 0, w - 1)
            patch = padded[pr:pr+3, pc:pc+3]
            labels = set(patch.flat) & (label_set | {0})
            junction_pixels.append((pr, pc))
            junction_labels.append(labels)

    if not junction_pixels:
        return [], {}, {}

    # ── Phase 2: clustering ───────────────────────────────────────────
    coords = np.array(junction_pixels)
    from scipy.spatial.distance import pdist, squareform
    dist = squareform(pdist(coords))
    visited = np.zeros(len(coords), dtype=bool)

    vertices = []
    for i in range(len(coords)):
        if visited[i]:
            continue
        group = dist[i] <= D_max
        visited |= group
        centroid = coords[group].mean(axis=0).astype(float)
        merged_labels = set()
        for j in np.where(group)[0]:
            merged_labels |= junction_labels[j]
        merged_labels.discard(0)
        vertices.append({
            'pos': centroid,
            'faces': merged_labels,
        })

    # ── Phase 2b: integrate bulbs ─────────────────────────────────────
    if bulbs is not None and len(bulbs) > 0:
        vertex_coords = np.array([v['pos'] for v in vertices]) if vertices else np.empty((0, 2))
        for br, bc in bulbs:
            br2, bc2 = int(round(br)), int(round(bc))
            # check if near boundary
            pr = np.clip(br2, 1, h) 
            pc = np.clip(bc2, 1, w)
            patch = padded[pr-1:pr+2, pc-1:pc+2]
            patch_labels = set(patch.flat) & label_set

            if len(vertex_coords) > 0:
                dists = np.linalg.norm(vertex_coords - [br2, bc2], axis=1)
                nearest = np.argmin(dists)
                if dists[nearest] <= D_max:
                    # merge into existing vertex
                    v = vertices[nearest]
                    v['pos'] = (v['pos'] + np.array([br2, bc2])) / 2
                    v['faces'] |= patch_labels
                    continue

            # new vertex from bulb
            vertices.append({
                'pos': np.array([br2, bc2], dtype=float),
                'faces': patch_labels,
            })
            vertex_coords = np.array([v['pos'] for v in vertices])

    # ── Phase 3: face ring assignment ─────────────────────────────────
    face_centroids = {i: pixels.mean(axis=0) for i, pixels in face_pixels.items()}
    face_vertices = {i: [] for i in valid_labels}

    for vid, v in enumerate(vertices):
        for fid in v['faces']:
            if fid in face_vertices:
                face_vertices[fid].append(vid)

    # angular sort per face
    for fid, vids in face_vertices.items():
        if len(vids) < 2:
            continue
        cr, cc = face_centroids[fid]
        positions = np.array([vertices[vid]['pos'] for vid in vids])
        angles = np.arctan2(positions[:, 0] - cr, positions[:, 1] - cc)
        order = np.argsort(angles)
        face_vertices[fid] = [vids[o] for o in order]

    return vertices, face_vertices, face_centroids