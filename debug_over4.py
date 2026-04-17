"""
Find images where any face ends up with >4 merged vertices.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import io as _io
from contextlib import redirect_stdout

sys.path.insert(0, ".")
from skimage import io
from mesh_extraction import (
    crop_image, extract_face_masks, extract_face_corners,
    compute_adjacency, merge_vertices, detect_bulb_corners,
    refine_faces, filter_corners,
)

folder = "jpeg images"
files = sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))

for f in files:
    try:
        img_full = io.imread(f"{folder}/{f}")
        img, _ = crop_image(img_full)
        bulbs = detect_bulb_corners(img)
        buf = _io.StringIO()
        with redirect_stdout(buf):
            labeled_clean, valid_labels, face_pixels = extract_face_masks(img, min_face_size=100)
            labeled_clean, valid_labels, face_pixels = refine_faces(
                img, labeled_clean, valid_labels, face_pixels)
            face_corners = extract_face_corners(labeled_clean, face_pixels)
            face_corners = filter_corners(face_corners, labeled_clean, valid_labels, radius=15)
            adjacency, adjacent_faces, face_centroids, adjacency_raw = compute_adjacency(
                labeled_clean, valid_labels, face_pixels,
                shared_border_threshold=35, bulbs=bulbs)
            vertices, face_vertices, all_pts, face_pt_indices = merge_vertices(
                face_corners, valid_labels, adjacency, face_pixels, bulbs=bulbs, img=img)
    except Exception as e:
        print(f"{f}: ERROR {e}")
        continue

    over = {fid: vids for fid, vids in face_vertices.items() if len(vids) > 4}
    if over:
        print(f"== {f} ==")
        for fid, vids in over.items():
            print(f"  face {fid}: {len(vids)} vids → {vids}")
