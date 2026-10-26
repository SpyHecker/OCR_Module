"""Image deskewing and rotation angle correction utilities."""

from __future__ import annotations

import cv2
import numpy as np


def estimate_skew_angle(image: np.ndarray) -> float:
    """Estimate skew angle of text in a grayscale or binary image.

    Uses contour minimum area bounding boxes to find dominant orientation.

    Args:
        image: Grayscale or BGR/RGB numpy array.

    Returns:
        Skew angle in degrees.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Invert colors: text as foreground (white), background as black
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Dilate text lines to merge characters into coherent blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    angles = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 100:
            continue
        rect = cv2.minAreaRect(c)
        angle = rect[-1]
        # Normalize angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        angles.append(angle)

    if not angles:
        return 0.0

    median_angle = float(np.median(angles))
    # Filter out wild outliers if the document is essentially horizontal
    if abs(median_angle) > 45:
        return 0.0
    return median_angle


def rotate_image(image: np.ndarray, angle: float, background_color: int = 255) -> np.ndarray:
    """Rotate an image by a given angle (in degrees) around its center.

    Args:
        image: Input numpy image array.
        angle: Angle to rotate (degrees).
        background_color: Fill color for exposed border areas.

    Returns:
        Rotated image.
    """
    if abs(angle) < 0.1:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Recompute new bounding bounds to prevent clipping
    cos = np.abs(rot_matrix[0, 0])
    sin = np.abs(rot_matrix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    rot_matrix[0, 2] += (new_w / 2) - center[0]
    rot_matrix[1, 2] += (new_h / 2) - center[1]

    border_val = (background_color, background_color, background_color) if len(image.shape) == 3 else background_color
    return cv2.warpAffine(
        image,
        rot_matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val
    )
