import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
from skimage import measure, morphology
from skimage.measure import approximate_polygon, find_contours


import cv2
import numpy as np
from skimage import measure, morphology


def extract_face_masks(gray, min_face_size=500, dilation_radius=2,
                       clahe_clip=2.0, clahe_tile=(8, 8),
                       blackhat_kernel=11, blackhat_threshold=12):
    """
    Extract individual face masks from a grayscale mesh image.
    Uses CLAHE + black-hat morphology to robustly detect mesh edges,
    even when they are weak or low-contrast.

    Returns:
        labeled_clean: 2D int array, each pixel labelled with its face ID
        valid_labels:  list of face IDs that are large enough to be real faces
        face_pixels:   dict {face_id: array of [row, col] pixel coords}
    """
    # Convert to uint8 for OpenCV
    img_uint8 = (gray * 255).astype(np.uint8)

    # CLAHE — enhance local contrast
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    img_eq = clahe.apply(img_uint8)

    # Black-hat — isolates dark lines smaller than the kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blackhat_kernel, blackhat_kernel))
    blackhat = cv2.morphologyEx(img_eq, cv2.MORPH_BLACKHAT, kernel)

    # Threshold to get edge mask
    _, edge_mask = cv2.threshold(blackhat, blackhat_threshold, 255, cv2.THRESH_BINARY)

    # Also mask out the pure black background (not part of the mesh)
    background_mask = img_uint8 < 20
    edge_mask[background_mask] = 0

    # Face mask = anything not an edge and not background
    not_edge = (edge_mask == 0) & ~background_mask

    # Clean up small fragments
    not_edge = morphology.remove_small_objects(not_edge, min_size=200)

    # Dilate edges slightly to ensure clean separation between faces
    edge_dilated = morphology.binary_dilation(edge_mask > 0, morphology.disk(dilation_radius))
    not_edge = not_edge & ~edge_dilated

    # Label connected regions
    labeled_clean = measure.label(not_edge, connectivity=1)

    valid_labels = [
        i for i in range(1, labeled_clean.max() + 1)
        if (labeled_clean == i).sum() > min_face_size
    ]

    face_pixels = {i: np.argwhere(labeled_clean == i) for i in valid_labels}

    return labeled_clean, valid_labels, face_pixels


def extract_face_corners(labeled_clean, face_pixels, tolerance=15):
    """
    For each face, find its contour and approximate it as a polygon.

    Returns:
        face_corners: dict {face_id: array of [row, col] corner points}
    """
    face_corners = {}
    for i, pixels in face_pixels.items():
        mask = (labeled_clean == i)
        contours = find_contours(mask, 0.5)
        if not contours:
            continue
        contour = max(contours, key=len)
        approx = approximate_polygon(contour, tolerance=tolerance)
        approx = approx[:-1]  # remove duplicate closing point
        face_corners[i] = approx
    return face_corners


def compute_adjacency(labeled_clean, valid_labels, face_pixels,
                      min_dist_threshold=15, shared_border_threshold=10):
    """
    Build a boolean adjacency matrix between faces.
    Two faces are adjacent if:
      - their minimum pixel distance is below min_dist_threshold
      - their shared border length exceeds shared_border_threshold

    Returns:
        adjacency:      2D bool array (n_faces x n_faces)
        adjacent_faces: dict {face_id: [list of adjacent face_ids]}
        face_centroids: dict {face_id: [row, col] centroid}
    """
    n = len(valid_labels)
    adjacency = np.zeros((n, n), dtype=bool)
    face_centroids = {i: pixels.mean(axis=0) for i, pixels in face_pixels.items()}

    for a, i in enumerate(valid_labels):
        mask_i = (labeled_clean == i)
        dilated_i = morphology.binary_dilation(mask_i, morphology.disk(3))
        border_i = dilated_i & ~mask_i

        for b, j in enumerate(valid_labels):
            if j <= i:
                continue
            mask_j = (labeled_clean == j)
            dilated_j = morphology.binary_dilation(mask_j, morphology.disk(3))
            shared_border = np.sum(border_i & dilated_j)

            pts_i = np.argwhere(mask_i)
            pts_j = np.argwhere(mask_j)
            min_dist = cdist(pts_i, pts_j).min()

            if min_dist < min_dist_threshold and shared_border > shared_border_threshold:
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
