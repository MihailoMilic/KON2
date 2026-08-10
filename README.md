# NIO Hackethon Top 5 Rank Submission — Mesh Extraction Pipeline 

Converts normalised hole images (JPEG) into Nastran bulk-data mesh files (`.nas`).
Each image is segmented into faces, corners are detected and merged into shared vertices,
a scale reference bar converts pixel coordinates to real-world units, and a conforming
quad/tri mesh is written out ready for FEA import.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Input Data](#input-data)
5. [Running the Pipeline](#running-the-pipeline)
6. [Output Files](#output-files)
7. [Pipeline Stages](#pipeline-stages)
8. [Single-Image Development Mode](#single-image-development-mode)

---

## Requirements

| Dependency | Purpose |
|---|---|
| Python ≥ 3.10 | Runtime |
| `numpy` | Array operations throughout |
| `opencv-python` | CLAHE equalisation, Canny edges, HSV conversions |
| `scikit-image` | Segmentation, morphology, corner detection |
| `scipy` | Distance computations, spatial utilities |
| `shapely` | Polygon geometry for corner filtering |
| `scikit-learn` | DBSCAN clustering for vertex merging |
| `simplification` | Visvalingam-Whyatt polyline simplification |
| `matplotlib` | Visualisation outputs |
| `networkx` | Graph operations for adjacency and cycle detection |
| `pillow` | Image I/O backend used by scikit-image |

---

## Installation

```bash
# 1. Unzip the submission and enter the project root
unzip team-name.zip
cd team-name

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip install -r src/requirements.txt
```

> **Conda alternative**
> ```bash
> conda create -n aeon python=3.11
> conda activate aeon
> pip install -r src/requirements.txt
> ```

---

## Project Structure

```
team-name/
├── README.md                     # This file
├── report.pdf                    # One-page approach summary
│
├── nas/                          # OUTPUT — 100 generated .nas files
│   ├── hole_001_flat.nas
│   ├── hole_002_flat.nas
│   └── ...                       # One file per input image, names match exactly
│
└── src/                          # Source code
    ├── main.py                   # Entry point — run this to reproduce all .nas files
    ├── mesh_extraction.py        # Core pipeline: segmentation, corners, adjacency, .nas writer
    ├── visualisation.py          # Matplotlib helpers that produce the diagnostic PNGs
    ├── requirements.txt          # Pinned Python dependencies
    │
    ├── jpeg images/              # INPUT — place the 100 normalised hole JPEGs here
    │   ├── hole_001_normalised.jpg
    │   └── ...
    │
    └── output/                   # Per-image diagnostic visualisations (created automatically)
        └── hole_001_normalised/
            ├── original.jpg      # Copy of the source image
            ├── labeled.png       # Segmented face regions with numeric labels
            ├── corners.png       # Detected polygon corners per face
            ├── adjacency.png     # Face adjacency graph overlay
            ├── merged.png        # Merged/shared vertex positions
            ├── combined_viz.png  # Combined diagnostic view
            └── trace.txt         # Full stdout/stderr log for this image
```

---

## Input Data

Place the normalised hole JPEG images in `src/jpeg images/`.
Expected filename convention: `hole_<NNN>_normalised.jpg` (e.g. `hole_042_normalised.jpg`).

Each image must contain:
- The mesh hole faces with distinct colours or luminosity separations.
- A **magenta L-shaped scale bar** — a vertical bar on the left edge (= 75 world units)
  and a horizontal bar along the bottom edge (= 100 world units).
  The pipeline uses these bars to convert pixel coordinates to real-world coordinates.

---

## Running the Pipeline

### Batch mode — reproduce all `.nas` files

```bash
cd src
python main.py
```

Processes every `.jpg` in `jpeg images/` alphabetically. The `.nas` files are written to
`../nas/` (one level up from `src/`, matching the submission layout). Per-stage diagnostic
PNGs and a full log are saved under `output/<stem>/` for each image.

Errors for one image are caught and logged without stopping the rest of the batch.

### Expected console output (per image)

```
========== hole_001_normalised.jpg ==========
Bulbs detected: 2 → [...]
Faces found: 18
Face 1: 3 neighbors → [2, 5, 7]
...
[SCALE] origin(row,col)=(...)  horiz=...px/100u  vert=...px/75u
[NAS] wrote 42 nodes, 18 elements → ../nas/hole_001_flat.nas
========== DONE hole_001_normalised.jpg ==========
```

---

## Output Files

### `.nas` files — Nastran bulk data

All 100 mesh files are pre-generated and located in `nas/` at the project root.
To regenerate them from scratch, follow [Running the Pipeline](#running-the-pipeline) above.

```
nas/
├── hole_001_flat.nas
├── hole_002_flat.nas
└── ...
```

Each file contains:
- **GRID** cards — one per mesh vertex, with real-world X/Y coordinates (Z = 0).
- **CQUAD4** cards — four-node quad elements for rectangular faces.
- **CTRIA3** cards — three-node triangle elements for triangular faces.

Example excerpt:
```
BEGIN BULK
GRID           1           46.55  30.3570.
GRID           2          54.011  28.6380.
...
CQUAD4         1     101       1       2       8      16
CTRIA3         2     101       3       7      12
...
ENDDATA
```

The property ID on every element defaults to `101`.

### Diagnostic visualisations

For each processed image, `src/output/<stem>/` contains six PNG overlays showing
each pipeline stage (face segmentation, corners, adjacency, merged vertices) plus
a `trace.txt` with the full run log. These are useful for inspecting or debugging
individual results without re-running the full batch.

---

## Pipeline Stages

| Stage | Function(s) | Description |
|---|---|---|
| 0 | `detect_bulb_corners` | Locates bright circular bulbs that mark special corner positions |
| 1 | `extract_face_masks` → `refine_faces` | Segments the image into labelled face regions using HSV or CLAHE grayscale paths |
| 2 | `extract_face_corners` → `filter_corners` → `filter_flat_corners` | Approximates polygon corners per face; removes near-boundary and collinear corners |
| 3 | `compute_adjacency` | Builds a face adjacency graph based on shared border length |
| 4 | `merge_vertices` | Snaps nearby corners shared by adjacent faces into single vertices |
| 5 | `extract_scale_reference` + `convert_vertices_to_world` | Reads the magenta scale bars and maps pixel coordinates to world units |
| 6 | `write_nas` | Writes the final GRID/CQUAD4/CTRIA3 bulk data file |

---

## Single-Image Development Mode

`src/test.py` runs the same pipeline on a single hardcoded image and writes diagnostic
PNGs directly to the working directory — useful for rapid iteration on one image:

```bash
# Edit the image path near the top of test.py first:
#   img = io.imread("jpeg images/hole_054_normalised.jpg")

cd src
python test.py
```

Outputs land in `src/`:
- `labeled.png`, `corners.png`, `adjacency.png`, `merges.png`
