import cv2
import numpy as np
from scipy import ndimage
from skan import Skeleton, summarize, draw
import matplotlib.pyplot as plt


def clean_junctions(skeleton_bool):
    skel = skeleton_bool.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbor_count = cv2.filter2D(skel, -1, kernel)
    junction_pixels = (skel > 0) & (neighbor_count >= 3)
    labeled, num = ndimage.label(junction_pixels)
    for i in range(1, num + 1):
        cluster = np.argwhere(labeled == i)
        if len(cluster) <= 1:
            continue
        center = cluster.mean(axis=0).round().astype(int)
        mask = (labeled == i)
        dilated = ndimage.binary_dilation(mask, np.ones((3, 3)))
        border = dilated & ~mask & (skel > 0)
        branch_endpoints = np.argwhere(border)
        skel[mask] = 0
        skel[center[0], center[1]] = 1
        for ep in branch_endpoints:
            draw_line_on_mask(skel, center, ep)
    return skel > 0


def draw_line_on_mask(mask, p1, p2):
    r0, c0 = int(p1[0]), int(p1[1])
    r1, c1 = int(p2[0]), int(p2[1])
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    while True:
        mask[r0, c0] = 1
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += sr
        if e2 < dr:
            err += dr
            c0 += sc




# ── Main pipeline ─────────────────────────────────────────────────────────────
img = cv2.imread("hole_096_normalised.jpg", cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread("hole_096_normalised.jpg")
hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
magenta = cv2.inRange(hsv, (130, 30, 30), (175, 255, 255))
red_low = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
red_high = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
annotations = cv2.dilate(magenta | red_low | red_high,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
img[annotations > 0] = 0

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
edges = cv2.Canny(clahe.apply(img), 60, 120)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
thinned = cv2.ximgproc.thinning(closed)

skeleton = clean_junctions(thinned > 0)

cv2.imwrite("mask_clean.png", skeleton.astype(np.uint8) * 255)