"""Thresholding and binarization techniques for document images."""

from __future__ import annotations

import cv2
import numpy as np


def apply_otsu_threshold(gray_image: np.ndarray) -> np.ndarray:
    """Apply Otsu's global binarization after gentle Gaussian blur."""
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def apply_adaptive_threshold(
    gray_image: np.ndarray,
    block_size: int = 21,
    constant_c: int = 10
) -> np.ndarray:
    """Apply Adaptive Gaussian thresholding. Ideal for non-uniform lighting/shadows."""
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        constant_c
    )


def apply_clahe(gray_image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(gray_image)
