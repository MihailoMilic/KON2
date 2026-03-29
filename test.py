import cv2
import numpy as np

img = cv2.imread("hole_096_normalised.jpg", cv2.IMREAD_GRAYSCALE)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_eq = clahe.apply(img)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
blackhat = cv2.morphologyEx(img_eq, cv2.MORPH_BLACKHAT, kernel)

_, mask = cv2.threshold(blackhat, 12, 255, cv2.THRESH_BINARY)

# overlay edges on original
overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
overlay[mask > 0] = (0, 0, 255)  # red detected edges

cv2.imwrite("overlay.png", overlay)
cv2.imwrite("mask.png", mask)