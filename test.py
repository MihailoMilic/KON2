import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure, morphology
from skimage.filters import frangi, sato
from visualisation import plot_labeled_faces
from mesh_extraction import detect_bulb_corners
img = cv2.imread("hole_081_normalised.jpg", cv2.IMREAD_GRAYSCALE)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_eq = clahe.apply(img)

edge_mask = cv2.Canny(img_eq, threshold1=60, threshold2=120)

# Close the double lines into solid lines
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closed = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)

# Skeletonize to single-pixel-wide lines
skeleton = morphology.skeletonize(closed > 0)
img_eq[skeleton] = 0
img_eq[img_eq < 150] = 0
cv2.imwrite("edges_clean.png", img_eq)
not_black = morphology.remove_small_objects(img_eq, min_size=100)
labeled_clean = measure.label(not_black, connectivity=1)
# valid_labels = [
#     i for i in range(1, labeled_clean.max() + 1)
#     if (labeled_clean == i).sum() > 100
# ]
# face_pixels = {i: np.argwhere(labeled_clean == i) for i in valid_labels}
plot_labeled_faces(labeled_clean, save_path="labeled_test.png")
mask_out = (skeleton.astype(np.uint8)) * 255
cv2.imwrite("mask_clean.png", mask_out)
bulbs = detect_bulb_corners(io.imread("hole_081_normalised.jpg"))
print(f"Bulbs detected: {len(bulbs)} → {bulbs}")
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(img_eq, cmap='gray', origin='upper')
axes[0].set_title("White Bulbs Detected")

# Overlay detected bulb centroids (regionprops returns (row, col))
if bulbs.size:
    axes[0].scatter(
        bulbs[:, 1],  # x = col
        bulbs[:, 0],  # y = row
        s=60,
        facecolors='none',
        edgecolors='lime',
        linewidths=1.5,
        label='Bulb corners',
    )
    axes[0].legend(loc='lower right')
axes[0].axis('off')

axes[1].imshow(img, cmap='gray', origin='upper')
axes[1].set_title("Original")
axes[1].axis('off')

plt.tight_layout()
plt.show()

# import cv2
# import numpy as np
# from skimage import color, measure, morphology
# from skimage.util import img_as_ubyte


# def extract_face_masks(
#     img,
#     min_face_size=200,
#     dilation_radius=3,
#     gaussian_sigma=1.2,
#     laplacian_thresh=6,
#     min_region_size=200,
#     save_debug=True,
# ):
#     """
#     Extract individual face masks from a mesh-like image using Laplacian edges
#     as separators between faces.

#     Parameters
#     ----------
#     img : ndarray
#         Input image, RGB or grayscale.
#     min_face_size : int
#         Minimum number of pixels for a region to be considered a valid face.
#     dilation_radius : int
#         Radius used to widen detected edge separators.
#     gaussian_sigma : float
#         Sigma for Gaussian smoothing before Laplacian.
#     laplacian_thresh : int
#         Threshold on absolute Laplacian response. Higher = fewer edges.
#     min_region_size : int
#         Remove connected white regions smaller than this.
#     save_debug : bool
#         If True, writes intermediate images to disk.

#     Returns
#     -------
#     labeled_clean : ndarray[int]
#         2D labeled image where each connected component is a candidate face.
#     valid_labels : list[int]
#         Labels whose area is larger than min_face_size.
#     face_pixels : dict[int, ndarray]
#         Mapping {label: array of [row, col] coordinates}.
#     """

#     # 1) Convert to grayscale uint8 [0, 255]
#     if img.ndim == 3:
#         gray = color.rgb2gray(img)
#         gray = img_as_ubyte(gray)
#     else:
#         if img.dtype != np.uint8:
#             gray = img_as_ubyte(img)
#         else:
#             gray = img.copy()

#     # 2) Improve local contrast
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     gray_eq = clahe.apply(gray)

#     # 3) Smooth before Laplacian to reduce noise sensitivity
#     # OpenCV accepts sigmaX directly with kernel size (0,0)
#     blurred = cv2.GaussianBlur(gray_eq, (0, 0), sigmaX=gaussian_sigma, sigmaY=gaussian_sigma)

#     # 4) Laplacian response
#     lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
#     lap_abs = np.abs(lap)

#     # 5) Threshold Laplacian magnitude to get separator lines
#     edges = lap_abs > laplacian_thresh

#     # 6) Widen separators so they split neighboring faces more reliably
#     expanded_edges = morphology.binary_dilation(edges, morphology.disk(dilation_radius))

#     # 7) Candidate face regions are the non-edge areas
#     candidate_regions = ~expanded_edges

#     # 8) Clean small junk regions
#     candidate_regions = morphology.remove_small_objects(candidate_regions, min_size=min_region_size)

#     # Optional: fill tiny holes inside regions if needed
#     candidate_regions = morphology.remove_small_holes(candidate_regions, area_threshold=64)

#     # 9) Label connected regions
#     labeled_clean = measure.label(candidate_regions, connectivity=1)

#     # 10) Keep only sufficiently large regions
#     valid_labels = [
#         i for i in range(1, labeled_clean.max() + 1)
#         if np.count_nonzero(labeled_clean == i) > min_face_size
#     ]

#     face_pixels = {i: np.argwhere(labeled_clean == i) for i in valid_labels}

#     if save_debug:
#         cv2.imwrite("debug/debug_gray.png", gray)
#         cv2.imwrite("debug/debug_gray_eq.png", gray_eq)
#         cv2.imwrite("debug/debug_blurred.png", blurred)
#         cv2.imwrite(
#             "debug/debug_lap_abs.png",
#             cv2.convertScaleAbs(lap_abs, alpha=255.0 / max(lap_abs.max(), 1))
#         )
#         cv2.imwrite("debug/debug_edges.png", (edges.astype(np.uint8) * 255))
#         cv2.imwrite("debug/debug_expanded_edges.png", (expanded_edges.astype(np.uint8) * 255))
#         cv2.imwrite("debug/debug_candidate_regions.png", (candidate_regions.astype(np.uint8) * 255))

#     return labeled_clean, valid_labels, face_pixels
# img = io.imread("hole_096_normalised.jpg")
# extract_face_masks(img, min_face_size=100)
