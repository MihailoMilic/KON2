import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

from mesh_extraction import (
    extract_face_masks,
    extract_face_corners,
    compute_adjacency,
    merge_vertices,
    detect_bulb_corners,
    fill_bulb_gaps

)
from visualisation import (
    plot_labeled_faces,
    plot_corners,
    plot_adjacency,
    plot_labeled_faces_with_num,
    plot_merged_vertices,
    plot_connections
)

# ── Load image ────────────────────────────────────────────────────────────────
img = io.imread("hole_081_normalised.jpg")

# ── Stage 1: Extract face masks ───────────────────────────────────────────────
bulbs = detect_bulb_corners(img)
print(f"Bulbs detected: {len(bulbs)} → {bulbs}")
labeled_clean, valid_labels, face_pixels = extract_face_masks(img, min_face_size=100)
print(f"Faces found: {len(valid_labels)}")
plot_labeled_faces(labeled_clean, save_path="labeled.png")
if bulbs.size > 0:
    labeled_clean = fill_bulb_gaps(labeled_clean, bulbs, radius=2)
    plot_labeled_faces(labeled_clean, save_path="labeled_filled.png")
# ── Stage 2: Extract corners per face ─────────────────────────────────────────
face_corners = extract_face_corners(labeled_clean, face_pixels, tolerance=15)
plot_corners(img, face_corners, face_pixels, valid_labels, save_path="corners.png")
# ── Stage 3: Compute face adjacency ───────────────────────────────────────────
adjacency, adjacent_faces, face_centroids = compute_adjacency(
    labeled_clean, valid_labels, face_pixels,
   shared_border_threshold=25
)
plot_adjacency(img, labeled_clean, valid_labels,
               face_centroids, adjacent_faces, save_path="adjacency.png")
for face_id in sorted(adjacent_faces.keys()):
    print(f"Face {face_id}: {len(adjacent_faces[face_id])} neighbors → {adjacent_faces[face_id]}")

# # ── Stage 4: Merge vertices ────────────────────────────────────────────────────
# vertices, face_vertices, all_pts, face_pt_indices = merge_vertices(
#     face_corners, valid_labels, adjacency
# )
# print(f"\nUnique vertices: {len(vertices)}")
# for face_id, vids in face_vertices.items():
#     print(f"  Face {face_id}: {len(vids)} vertices → {vids}")

# plot_merged_vertices(img, face_corners, all_pts, vertices, 
                    #  valid_labels, adjacency, face_pt_indices, save_path="merges.png")
# ── Shape detection for small faces ───────────────────────────────────────────
# from mesh_extraction import find_dominant_direction

# face_shapes = {}
# for i, pixels in face_pixels.items():
#     mask = (labeled_clean == i)
#     area = mask.sum()
#     if area < 200:
#         angle, score, lengths = find_dominant_direction(mask)
#         if score is not None and score < 0.15:
#             face_shapes[i] = 'Q'
#         else:
#             face_shapes[i] = 'T'
#     else:
#         # use vertex count from approx
#         n = len(face_corners.get(i, []))
#         if n <= 4:
#             face_shapes[i] = 'T' if n == 3 else 'Q'
#         else:
#             face_shapes[i] = f'{n}'

# # ── Visualize with shape labels ───────────────────────────────────────────────
# fig, ax = plt.subplots(figsize=(12, 10))
# ax.imshow(img)
# for i, pixels in face_pixels.items():
#     cy, cx = np.mean(pixels, axis=0)
#     label = face_shapes.get(i, '?')
#     color = 'lime' if label == 'Q' else 'red' if label == 'T' else 'yellow'
#     ax.text(cx, cy, label, color=color, fontsize=12, fontweight='bold',
#             ha='center', va='center',
#             path_effects=[__import__('matplotlib.patheffects', fromlist=['withStroke']).withStroke(linewidth=2, foreground='black')])
# ax.set_title("Shape detection: Q=quad, T=tri")
# ax.axis('off')
# fig.savefig("shape_labels.png", dpi=150, bbox_inches='tight')
# print("Saved shape_labels.png")