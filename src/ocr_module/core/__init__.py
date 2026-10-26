"""Core interfaces and data structures."""

from ocr_module.core.types import BoundingBox, OCRWord, OCRLine, OCRResult
from ocr_module.core.base import BaseOCREngine, ImageInput

__all__ = [
    "BoundingBox",
    "OCRWord",
    "OCRLine",
    "OCRResult",
    "BaseOCREngine",
    "ImageInput",
]
