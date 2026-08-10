import argparse
import logging
import os
import sys
import shutil
import traceback
import contextlib
from pathlib import Path

import numpy as np
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
    write_nas,
    extract_scale_reference,
    convert_vertices_to_world,
)
from visualisation import (
    plot_labeled_faces,
    plot_corners,
    plot_adjacency,
    plot_labeled_faces_with_num,
    plot_merged_vertices,
    plot_combined_visualization
)


class _Tee:
    """Write to every stream given. Used to mirror stdout/stderr into a
    per-image trace file while still showing everything on the console."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


def process_one(img_path: Path, out_dir: Path) -> None:
    """Run the full pipeline for a single image. All image outputs and
    the full stdout/stderr trace land under `out_dir/`.

    Mirrors test.py exactly (same parameters, same stage order) so main
    and test stay in lockstep."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep a copy of the source image inside the per-image folder so the
    # reviewer can see the original next to the derived outputs.
    shutil.copy2(str(img_path), str(out_dir / "original.jpg"))

    log = logging.getLogger(__name__)
    img = io.imread(str(img_path))

    # ── Stage 0: bulbs ───────────────────────────────────────────────────
    bulbs = detect_bulb_corners(img)
    log.debug("Bulbs detected: %d → %s", len(bulbs), bulbs)

    # ── Stage 1: face masks + refinement ─────────────────────────────────
    labeled_clean, valid_labels, face_pixels = extract_face_masks(
        img, min_face_size=100, lum_fallback_thresh=148
    )
    log.debug("Faces after extraction: %d", len(valid_labels))
    labeled_clean, valid_labels, face_pixels = refine_faces(
        img, labeled_clean, valid_labels, face_pixels
    )
    plot_labeled_faces_with_num(labeled_clean, save_path=str(out_dir / "labeled.png"))

    # ── Stage 2: corners + filters ───────────────────────────────────────
    face_corners = extract_face_corners(labeled_clean, face_pixels, tolerance=4)
    face_corners = filter_corners(face_corners, labeled_clean, valid_labels, radius=15)
    face_corners = filter_flat_corners(face_corners, labeled_clean, valid_labels)
    plot_corners(img, face_corners, face_pixels, valid_labels,
                 save_path=str(out_dir / "corners.png"))

    # ── Stage 3: adjacency ───────────────────────────────────────────────
    adjacency, adjacent_faces, face_centroids, adjacency_raw = compute_adjacency(
        labeled_clean, valid_labels, face_pixels,
        shared_border_threshold=33,
        bulbs=bulbs,
    )
    plot_adjacency(img, labeled_clean, valid_labels,
                   face_centroids, adjacent_faces,
                   save_path=str(out_dir / "adjacency.png"))
    for face_id in sorted(adjacent_faces.keys()):
        log.debug("Face %d: %d neighbors → %s",
                  face_id, len(adjacent_faces[face_id]), adjacent_faces[face_id])

    # ── Stage 4: merge vertices ──────────────────────────────────────────
    vertices, face_vertices, all_pts, face_pt_indices = merge_vertices(
        face_corners, valid_labels, adjacency, face_pixels, bulbs=bulbs, img=img
    )
    
    plot_merged_vertices(
        img,
        face_corners,
        all_pts,
        vertices,
        valid_labels,
        adjacency,
        face_pt_indices,
        save_path=str(out_dir / "merged.png"),
    )
    plot_combined_visualization(img, face_corners, vertices, valid_labels, 
                                face_centroids, adjacent_faces, save_path=str(out_dir / "combined_viz.png"))
    # ── Stage 5: write .nas output ───────────────────────────────────────
    # Derive numeric file_id from the image stem (e.g. "hole_043_normalised" → 43)
    stem = img_path.stem
    digits = [p for p in stem.split("_") if p.isdigit()]
    file_id = int(digits[0]) if digits else 101
    nas_dir = out_dir.parent.parent / "nas"
    nas_dir.mkdir(parents=True, exist_ok=True)
    nas_path = nas_dir / f"{stem}.nas"
    scale = extract_scale_reference(img, vertical_units=75.0, horizontal_units=100.0)
    world_vertices = convert_vertices_to_world(vertices, scale)
    write_nas(world_vertices, face_vertices, nas_path, file_id=file_id)


def run_with_trace(img_path: Path, out_dir: Path) -> None:
    """Process `img_path` while teeing stdout/stderr into
    `out_dir/trace.txt`. Catches and reports errors without killing the
    outer batch loop."""
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.txt"
    real_stdout = sys.__stdout__
    real_stderr = sys.__stderr__

    with open(trace_path, "w") as f:
        tee_out = _Tee(real_stdout, f)
        tee_err = _Tee(real_stderr, f)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            print(f"========== {img_path.name} ==========")
            try:
                process_one(img_path, out_dir)
                print(f"========== DONE {img_path.name} ==========")
            except Exception:
                print("[FATAL] pipeline failed with exception:")
                traceback.print_exc()
                print(f"========== FAILED {img_path.name} ==========")


def main():
    parser = argparse.ArgumentParser(description="Batch mesh extraction pipeline")
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging (per-vertex / per-cycle detail)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    folder = Path("jpeg images")
    out_root = Path("output")
    out_root.mkdir(exist_ok=True)

    jpg_files = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".jpg")
    print(f"Found {len(jpg_files)} images to process")

    for i, img_path in enumerate(jpg_files, start=1):
        stem = img_path.stem
        out_dir = out_root / stem
        print(f"[{i:>3}/{len(jpg_files)}] Processing {img_path.name} ...")
        run_with_trace(img_path, out_dir)

    print(f"\nDone. {len(jpg_files)} images processed.")
    print(f"  .nas files  → nas/")
    print(f"  diagnostics → output/")


if __name__ == "__main__":
    main()
