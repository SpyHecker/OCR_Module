"""Engine factory for dynamically selecting and instantiating OCR engines."""

from __future__ import annotations

import logging
from typing import Optional
from ocr_module.config import EngineConfig
from ocr_module.core.base import BaseOCREngine
from ocr_module.engines.tesseract_engine import TesseractEngine
from ocr_module.engines.vision_engine import VisionTransformerEngine
from ocr_module.engines.mock_engine import MockOCREngine

logger = logging.getLogger(__name__)


class OCREngineFactory:
    """Factory for creating OCR engine instances based on configuration and availability."""

    @staticmethod
    def create(config: Optional[EngineConfig] = None) -> BaseOCREngine:
        cfg = config or EngineConfig()
        req_type = cfg.engine_type.lower()

        if req_type in ("baseline", "tesseract"):
            return TesseractEngine(tesseract_cmd=cfg.tesseract_cmd, languages=cfg.languages)

        elif req_type in ("vision", "transformers", "trocr"):
            return VisionTransformerEngine()

        elif req_type == "mock":
            return MockOCREngine()

        elif req_type == "auto":
            # Auto-detect best available engine
            tess = TesseractEngine(tesseract_cmd=cfg.tesseract_cmd, languages=cfg.languages)
            if tess.is_available():
                return tess

            logger.info("Tesseract binary not detected. Checking deep learning engine...")
            vision = VisionTransformerEngine()
            if vision.is_available():
                # Note: vision transformer will lazy-load weights when called
                return vision

            logger.warning("No heavy OCR engines ready. Defaulting to mock engine.")
            return MockOCREngine()

        raise ValueError(
            f"Unsupported engine type: '{cfg.engine_type}'. "
            f"Supported options: 'auto', 'baseline', 'tesseract', 'transformers', 'mock'."
        )
