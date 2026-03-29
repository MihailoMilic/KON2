import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

from mesh_extraction import (
    extract_face_masks,
    extract_face_corners,
    compute_adjacency,
    merge_vertices,
)
from visualisation import (
    plot_labeled_faces,
    plot_corners,
    plot_adjacency,
    plot_merged_vertices,
)

# ── Load image ────────────────────────────────────────────────────────────────
img = io.imread("hole_073_normalised.jpg")
gray = color.rgb2gray(img)

# ── Stage 1: Extract face masks ───────────────────────────────────────────────
labeled_clean, valid_labels, face_pixels = extract_face_masks(gray, min_face_size=500)
print(f"Faces found: {len(valid_labels)}")
plot_labeled_faces(labeled_clean, save_path="labeled.png")

# ── Stage 2: Extract corners per face ─────────────────────────────────────────
face_corners = extract_face_corners(labeled_clean, face_pixels, tolerance=15)
plot_corners(img, face_corners, face_pixels, valid_labels, save_path="corners.png")

# ── Stage 3: Compute face adjacency ───────────────────────────────────────────
adjacency, adjacent_faces, face_centroids = compute_adjacency(
    labeled_clean, valid_labels, face_pixels,
    min_dist_threshold=15, shared_border_threshold=10
)
plot_adjacency(img, labeled_clean, valid_labels, face_corners,
               face_centroids, adjacent_faces, save_path="adjacency.png")

# ── Stage 4: Merge vertices ────────────────────────────────────────────────────
vertices, face_vertices, all_pts, face_pt_indices = merge_vertices(
    face_corners, valid_labels, adjacency
)
print(f"\nUnique vertices: {len(vertices)}")
for face_id, vids in face_vertices.items():
    print(f"  Face {face_id}: {len(vids)} vertices → {vids}")

plot_merged_vertices(img, face_corners, all_pts, vertices,
                     valid_labels, adjacency, face_pt_indices, save_path="merges.png")
