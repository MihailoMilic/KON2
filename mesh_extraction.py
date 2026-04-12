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
            continue

        # Skip if the high region substantially reaches into a
        # *different* low face. Without this guard, two distinct low
        # faces that happen to map to the same merged high region
        # would both paint that high region, and the later face would
        # overwrite the earlier one and absorb its pixels (Frankenstein
        # face). The 1.6x size guard above doesn't catch this when the
        # high region is only slightly larger than one of the two low
        # faces it spans.
        low_in_high = labeled_low[mask_high]
        low_in_high = low_in_high[low_in_high > 0]
        foreign_px = int((low_in_high != fid).sum())
        if foreign_px > 0.2 * mask_high.sum():
            refined[mask_low] = fid
            continue

        # Also respect previously used high regions, so two low faces
        # can't claim the same best-high.
        if best in used_high:
            refined[mask_low] = fid
            continue

        refined[mask_high] = fid
        used_high.add(best)

    # Refinement can overwrite one low-res face with another when two
    # faces claim the same high-res region. Drop any face whose refined
    # mask ends up empty so downstream code (extract_face_corners,
    # filter_corners, merge_vertices) never sees a phantom label.
    face_pixels = {}
    valid_labels = []
    for i in valid_low:
        pix = np.argwhere(refined == i)
        if len(pix) == 0:
            continue
        face_pixels[i] = pix
        valid_labels.append(i)
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

    return adj_bool, adjacent_faces, face_centroids, adjacency

import numpy as np
import networkx as nx
from collections import defaultdict

def merge_vertices(face_corners, valid_labels, adjacency, face_pixels,
                               bulbs=None, dilation_radius=15):
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

    # 4. Adjacency graph
    G = nx.Graph()
    for a, i in enumerate(valid_labels):
        for b, j in enumerate(valid_labels):
            if adjacency[a, b] and j > i:
                G.add_edge(i, j)

    # 5. Cycles
    cycles = nx.minimum_cycle_basis(G)

    # 6. Per-cycle junction vertex
    vertices = {}
    pt_to_vertex = {}
    face_extra_vids = defaultdict(list)
    used_bulbs = set()  # bulb indices already materialized as a vertex
    next_vid = 1

    for cycle in cycles:
        if len(cycle) > 6:
            continue

        relevant_indices = np.concatenate([face_pt_indices[fid] for fid in cycle])
        if len(relevant_indices) == 0:
            continue

        # Intersection of dilated masks for every face in this cycle
        inter = dilated_masks[cycle[0]].copy()
        for fid in cycle[1:]:
            inter &= dilated_masks[fid]
        if not inter.any():
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
            hits = np.where(inside)[0]
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
            if fid not in winner_faces:
                face_extra_vids[fid].append(next_vid)

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

    # 7b. Isolated-bulb sweep. Any bulb not yet materialized as a
    # vertex still represents a true corner. For each such bulb, find
    # every face whose dilation contains it, absorb that face's
    # unclaimed corners within `dilation_radius`, and create a vertex
    # pinned to the bulb. Bulbs with no nearby unclaimed corner become
    # dangling vertices only if at least one face's dilation still
    # contains them (otherwise they're noise and ignored).
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
            absorbed = []
            for fid in faces_here:
                for idx in face_pt_indices[fid]:
                    idx = int(idx)
                    if idx in pt_to_vertex:
                        continue
                    if np.linalg.norm(all_pts[idx] - bulb_pos) <= dilation_radius:
                        absorbed.append(idx)
            if not absorbed:
                continue  # bulb with no supporting corner -> skip as noise
            vertices[next_vid] = bulb_pos
            for idx in absorbed:
                pt_to_vertex[idx] = next_vid
            for fid in faces_here:
                if not any(pt_to_face[idx] == fid for idx in absorbed):
                    face_extra_vids[fid].append(next_vid)
            used_bulbs.add(bi)
            next_vid += 1

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

    # 9. Singleton vertices for whatever corners still remain
    for idx in range(len(all_pts)):
        if idx not in pt_to_vertex:
            vertices[next_vid] = all_pts[idx]
            pt_to_vertex[idx] = next_vid
            next_vid += 1

    # 10. Build face_vertices, preserving original corner order, dedup'd
    face_vertices = {}
    for fid in valid_labels:
        seen = []
        for idx in face_pt_indices[fid]:
            vid = pt_to_vertex[int(idx)]
            if vid not in seen:
                seen.append(vid)
        for vid in face_extra_vids[fid]:
            if vid not in seen:
                seen.append(vid)
        face_vertices[fid] = seen

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
