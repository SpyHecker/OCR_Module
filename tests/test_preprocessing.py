"""Unit tests for image preprocessing pipeline."""

import numpy as np
from PIL import Image
from ocr_module.preprocessing.preprocessor import ImagePreprocessor
from ocr_module.preprocessing.deskew import estimate_skew_angle, rotate_image
from ocr_module.preprocessing.binarization import (
    apply_otsu_threshold,
    apply_adaptive_threshold,
    apply_clahe,
)
from ocr_module.config import PreprocessingConfig


def test_binarization_functions():
    # Create simple synthetic gradient/noisy grayscale array
    gray = np.linspace(0, 255, 100 * 100, dtype=np.uint8).reshape((100, 100))

    otsu = apply_otsu_threshold(gray)
    assert otsu.shape == (100, 100)
    assert set(np.unique(otsu)).issubset({0, 255})

    adaptive = apply_adaptive_threshold(gray, block_size=11, constant_c=2)
    assert adaptive.shape == (100, 100)
    assert set(np.unique(adaptive)).issubset({0, 255})

    clahe = apply_clahe(gray)
    assert clahe.shape == (100, 100)


def test_deskew_and_rotate():
    # Create an image with an angle
    img = np.ones((200, 200), dtype=np.uint8) * 255
    # Draw horizontal bar
    img[90:110, 30:170] = 0

    rotated = rotate_image(img, 15.0)
    assert rotated.shape[0] > 0
    assert rotated.shape[1] > 0

    angle = estimate_skew_angle(rotated)
    assert isinstance(angle, float)


def test_preprocessor_pipeline():
    test_img = Image.new("RGB", (150, 80), color=(240, 240, 240))
    cfg = PreprocessingConfig(enabled=True, deskew=True, denoise=True, binarize=True)
    preprocessor = ImagePreprocessor(cfg)

    processed, meta = preprocessor.process(test_img)
    assert processed.size[0] > 0
    assert processed.size[1] > 0
    assert meta["preprocessed"] is True
    assert "original_dimensions" in meta
