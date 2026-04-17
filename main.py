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
    filter_edge_faces,

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

folder = "jpeg images"

jpg_files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
print(f"Found {len(jpg_files)} JPEG files: {jpg_files}")
for file in jpg_files:
    print(file)
    img = io.imread(f"jpeg images/{file}")
    # ── Stage 1: Extract face masks ───────────────────────────────────────────────
    bulbs = detect_bulb_corners(img)
    print(f"Bulbs detected: {len(bulbs)} → {bulbs}")
    labeled_clean, valid_labels, face_pixels = extract_face_masks(img, min_face_size=100)
    print(f"Faces found: {len(valid_labels)}")
    labeled_clean, valid_labels, face_pixels  = refine_faces(img, labeled_clean, valid_labels, face_pixels)
    plot_labeled_faces_with_num(labeled_clean, save_path=f"labeled/labeled-{file[:-4]}.png")
    # ── Stage 2: Extract corners per face ─────────────────────────────────────────
    face_corners = extract_face_corners(labeled_clean, face_pixels)
    face_corners = filter_corners(face_corners, labeled_clean, valid_labels, radius =15)

    plot_corners(img, face_corners, face_pixels, valid_labels, save_path=f"corners/corners-{file[:-4]}.png")
    # ── Stage 3: Compute face adjacency ───────────────────────────────────────────
    adjacency, adjacent_faces, face_centroids,adjacency_raw = compute_adjacency(
        labeled_clean, valid_labels, face_pixels,
    shared_border_threshold=35,
    bulbs = bulbs
    )
    plot_adjacency(img, labeled_clean, valid_labels,
                face_centroids, adjacent_faces, save_path=f"adjacency/adjacency-{file[:-4]}.png")
    for face_id in sorted(adjacent_faces.keys()):
        print(f"Face {face_id}: {len(adjacent_faces[face_id])} neighbors → {adjacent_faces[face_id]}")

    vertices, face_vertices, all_pts, face_pt_indices= merge_vertices(face_corners,valid_labels,adjacency, face_pixels, bulbs=bulbs)
    plot_merged_vertices(img,
        face_corners,
        all_pts,
        vertices,
        valid_labels,
        adjacency,
        face_pt_indices, save_path=f"merged/merged-{file[:-4]}.png")