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
import os



img = io.imread(f"jpeg images/hole_009_normalised.jpg")
# img, crop_info = crop_image(img_full)
# print(f"Cropped: {img_full.shape[:2]} → {img.shape[:2]}, offsets: top={crop_info['top']}, left={crop_info['left']}")
# ── Stage 1: Extract face masks ───────────────────────────────────────────────
bulbs = detect_bulb_corners(img)
print(f"Bulbs detected: {len(bulbs)} → {bulbs}")
labeled_clean, valid_labels, face_pixels = extract_face_masks(img, min_face_size=100, lum_fallback_thresh=148)
print(f"Faces found: {len(valid_labels)}")
labeled_clean, valid_labels, face_pixels  = refine_faces(img, labeled_clean, valid_labels, face_pixels)
plot_labeled_faces_with_num(labeled_clean, save_path="labeled.png")
# ── Stage 2: Extract corners per face ─────────────────────────────────────────
face_corners = extract_face_corners(labeled_clean, face_pixels, tolerance=4)
face_corners = filter_corners(face_corners, labeled_clean, valid_labels, radius =15)
face_corners = filter_flat_corners(face_corners, labeled_clean, valid_labels,
                                   debug_faces={3, 17})

plot_corners(img, face_corners, face_pixels, valid_labels, save_path=f"corners.png")
# ── Stage 3: Compute face adjacency ───────────────────────────────────────────
adjacency, adjacent_faces, face_centroids,adjacency_raw = compute_adjacency(
    labeled_clean, valid_labels, face_pixels,
shared_border_threshold=33,
bulbs = bulbs
)
plot_adjacency(img, labeled_clean, valid_labels,
            face_centroids, adjacent_faces, save_path=f"adjacency.png")
for face_id in sorted(adjacent_faces.keys()):
    print(f"Face {face_id}: {len(adjacent_faces[face_id])} neighbors → {adjacent_faces[face_id]}")

vertices, face_vertices, all_pts, face_pt_indices= merge_vertices(face_corners,valid_labels,adjacency, face_pixels, bulbs=bulbs, img=img)
# vertices, face_vertices, valid_labels,face_pixels, labeled = filter_edge_faces(vertices, face_vertices, valid_labels, face_pixels, labeled_clean, img.shape, margin=200)   
plot_merged_vertices(img,
    face_corners,
    all_pts,
    vertices,
    valid_labels,
    adjacency,
    face_pt_indices, save_path=f"merges.png")
print(vertices)
