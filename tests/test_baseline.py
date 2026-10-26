"""Unit tests for baseline engine wrappers and error handling."""

import pytest
from PIL import Image
from ocr_module.engines.tesseract_engine import TesseractEngine
from ocr_module.engines.factory import OCREngineFactory
from ocr_module.config import EngineConfig


def test_tesseract_availability_check():
    # Pass non-existent executable to verify is_available and runtime check
    engine = TesseractEngine(tesseract_cmd="non_existent_binary_xyz_123")
    assert engine.is_available() is False

    test_img = Image.new("RGB", (50, 50), color=(255, 255, 255))
    with pytest.raises(RuntimeError, match="Tesseract binary is not installed"):
        engine.process_image(test_img)


def test_engine_factory_unknown_type():
    cfg = EngineConfig(engine_type="invalid_engine_name")
    with pytest.raises(ValueError, match="Unsupported engine type"):
        OCREngineFactory.create(cfg)


def test_engine_factory_auto_resolution():
    cfg = EngineConfig(engine_type="auto")
    engine = OCREngineFactory.create(cfg)
    assert engine is not None
    assert hasattr(engine, "process_image")
