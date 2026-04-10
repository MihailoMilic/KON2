import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure, morphology, color
from skimage.measure import find_contours, approximate_polygon
from mesh_extraction import detect_bulb_corners, find_boundary_corners, extract_face_masks
from visualisation import plot_labeled_faces

img = io.imread("hole_081_normalised.jpg")

# ── Extract faces ─────────────────────────────────────────────────────────
labeled_clean, valid_labels, face_pixels = extract_face_masks(img, min_face_size=100)
bulbs = detect_bulb_corners(img)
print(f"Bulbs detected: {len(bulbs)}")
print(f"Faces found: {len(valid_labels)}")

# ── Find boundary corners ────────────────────────────────────────────────
corners, corner_faces = find_boundary_corners(img, labeled_clean, valid_labels, face_pixels,
                                               tolerance=12, bulbs=bulbs)
print(f"Boundary corners found: {len(corners)}")

# ── Visualize ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Left: corners on image
axes[0].imshow(img)
for idx, (r, c) in enumerate(corners):
    n_faces = len(corner_faces.get(idx, set()))
    clr = 'lime' if n_faces >= 2 else 'red'  # green=shared, red=solo
    axes[0].plot(c, r, 'o', color=clr, markersize=8, markeredgecolor='white', markeredgewidth=1)
axes[0].set_title(f"Boundary corners: {len(corners)} (green=shared, red=solo)")
axes[0].axis('off')

# Right: corners on labeled map
from matplotlib.colors import ListedColormap
cmap = plt.cm.tab20
axes[1].imshow(labeled_clean, cmap=cmap, interpolation='nearest')
for idx, (r, c) in enumerate(corners):
    faces = corner_faces.get(idx, set())
    label = ','.join(str(f) for f in sorted(faces)) if faces else '?'
    axes[1].plot(c, r, 'o', color='white', markersize=6)
    axes[1].text(c + 5, r, label, color='white', fontsize=7,
                 path_effects=[__import__('matplotlib.patheffects', fromlist=['withStroke']).withStroke(linewidth=2, foreground='black')])
axes[1].set_title("Corners with face assignments")
axes[1].axis('off')

plt.tight_layout()
plt.savefig("boundary_corners.png", dpi=150, bbox_inches='tight')
print("Saved boundary_corners.png")
plt.show()