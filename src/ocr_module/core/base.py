"""Abstract base class defining the contract for all OCR engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image

from ocr_module.core.types import OCRResult


ImageInput = Union[str, Path, Image.Image, np.ndarray]


class BaseOCREngine(ABC):
    """Abstract Base Class for OCR engines.

    Subclasses must implement `process_image` and `is_available`.
    """

    def __init__(self, name: str = "base_engine"):
        self.name = name

    @abstractmethod
    def is_available(self) -> bool:
        """Check if necessary runtime binaries or weights are available."""
        pass

    @abstractmethod
    def process_image(self, image: Image.Image) -> OCRResult:
        """Extract text and bounding data from a PIL Image.

        Args:
            image: Standardized PIL Image instance.

        Returns:
            OCRResult populated with text, words, and metadata.
        """
        pass

    def extract_text(self, image: Image.Image) -> str:
        """Convenience method to return only plain extracted text."""
        result = self.process_image(image)
        return result.text
