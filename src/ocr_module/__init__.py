"""OCR Module - Modular Optical Character Recognition and Document Intelligence."""

from ocr_module.core.types import BoundingBox, OCRWord, OCRLine, OCRResult
from ocr_module.core.base import BaseOCREngine
from ocr_module.config import PipelineConfig, EngineConfig, PreprocessingConfig
from ocr_module.pipeline import OCRPipeline

__version__ = "0.1.0"

__all__ = [
    "OCRPipeline",
    "PipelineConfig",
    "EngineConfig",
    "PreprocessingConfig",
    "BoundingBox",
    "OCRWord",
    "OCRLine",
    "OCRResult",
    "BaseOCREngine",
]
