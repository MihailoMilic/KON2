import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

from mesh_extraction import (
    crop_image,
    extract_face_masks,
    extract_face_corners,
    compute_adjacency,
    merge_vertices,
    detect_bulb_corners,
    refine_faces,
    filter_corners,
    filter_flat_corners,
    filter_edge_faces,
    write_nas

)
from visualisation import (
    plot_labeled_faces,
    plot_corners,
    plot_adjacency,
    plot_labeled_faces_with_num,
    plot_merged_vertices,
    plot_connections,
    
)

# ── Load image ────────────────────────────────────────────────────────────────
# Change this path to test a different image without running the full batch.
img = io.imread("jpeg images/hole_080_normalised.jpg")

# ── Stage 0: Detect bright circular bulb corners ──────────────────────────────
bulbs = detect_bulb_corners(img)
print(f"Bulbs detected: {len(bulbs)} → {bulbs}")

# ── Stage 1: Extract face masks ───────────────────────────────────────────────
# lum_fallback_thresh=148 was tuned to correctly classify triangular faces
# (holes 068, 090) that a threshold of 150 misidentified as quads.
labeled_clean, valid_labels, face_pixels = extract_face_masks(img, min_face_size=100, lum_fallback_thresh=148)
print(f"Faces found: {len(valid_labels)}")
labeled_clean, valid_labels, face_pixels = refine_faces(img, labeled_clean, valid_labels, face_pixels)
plot_labeled_faces_with_num(labeled_clean, save_path="labeled.png")

# ── Stage 2: Extract corners per face ─────────────────────────────────────────
face_corners = extract_face_corners(labeled_clean, face_pixels, tolerance=4)
# radius=15 removes corners that land too close to a face boundary pixel.
face_corners = filter_corners(face_corners, labeled_clean, valid_labels, radius=15)
face_corners = filter_flat_corners(face_corners, labeled_clean, valid_labels,
                                   debug_faces={3, 17})
plot_corners(img, face_corners, face_pixels, valid_labels, save_path="corners.png")

# ── Stage 3: Compute face adjacency ───────────────────────────────────────────
# shared_border_threshold=33 was raised from the default to handle hole_054
# where the dilation radius for adjacency detection is 7px.
adjacency, adjacent_faces, face_centroids, adjacency_raw = compute_adjacency(
    labeled_clean, valid_labels, face_pixels,
    shared_border_threshold=33,
    bulbs=bulbs,
)
plot_adjacency(img, labeled_clean, valid_labels,
               face_centroids, adjacent_faces, save_path="adjacency.png")
for face_id in sorted(adjacent_faces.keys()):
    print(f"Face {face_id}: {len(adjacent_faces[face_id])} neighbors → {adjacent_faces[face_id]}")

# ── Stage 4: Merge shared vertices ────────────────────────────────────────────
vertices, face_vertices, all_pts, face_pt_indices = merge_vertices(
    face_corners, valid_labels, adjacency, face_pixels, bulbs=bulbs, img=img
)
plot_merged_vertices(img,
    face_corners,
    all_pts,
    vertices,
    valid_labels,
    adjacency,
    face_pt_indices, save_path="merges.png")

