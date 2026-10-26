"""Image preprocessor coordinating computer vision filters and normalization."""

from __future__ import annotations

from typing import Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image

from ocr_module.config import PreprocessingConfig
from ocr_module.preprocessing.deskew import estimate_skew_angle, rotate_image
from ocr_module.preprocessing.binarization import (
    apply_otsu_threshold,
    apply_adaptive_threshold,
    apply_clahe,
)


class ImagePreprocessor:
    """Pipelines classical computer vision transformations to optimize OCR readability."""

    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()

    def process(self, image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
        """Run configured preprocessing pipeline on an image.

        Args:
            image: PIL Image input.

        Returns:
            Tuple of (preprocessed PIL Image, metadata dictionary).
        """
        if not self.config.enabled:
            return image, {"preprocessed": False}

        # Convert PIL to BGR/Grayscale OpenCV format
        cv_img = np.array(image)
        if len(cv_img.shape) == 2:
            gray = cv_img
        elif cv_img.shape[2] == 4:
            # RGBA to RGB then Gray
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

        metadata: Dict[str, Any] = {
            "preprocessed": True,
            "original_dimensions": [image.width, image.height],
        }

        # 1. CLAHE Contrast Enhancement
        if self.config.clahe_contrast:
            gray = apply_clahe(gray)

        # 2. Denoising
        if self.config.denoise:
            gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

        # 3. Deskew Angle Estimation and Correction
        skew_angle = 0.0
        if self.config.deskew:
            skew_angle = estimate_skew_angle(gray)
            if abs(skew_angle) > 0.5:
                gray = rotate_image(gray, skew_angle)
        metadata["skew_angle_corrected"] = skew_angle

        # 4. Binarization
        if self.config.binarize:
            if self.config.binarize_method.lower() == "adaptive":
                gray = apply_adaptive_threshold(gray)
            else:
                gray = apply_otsu_threshold(gray)

        processed_pil = Image.fromarray(gray)
        metadata["processed_dimensions"] = [processed_pil.width, processed_pil.height]

        return processed_pil, metadata
