"""Unit tests for full OCR pipeline and CLI."""

from pathlib import Path
from PIL import Image
from ocr_module.pipeline import OCRPipeline
from ocr_module.config import PipelineConfig, EngineConfig
from ocr_module.cli import main


def test_pipeline_with_mock_engine():
    img = Image.new("RGB", (300, 100), color=(255, 255, 255))
    cfg = PipelineConfig(
        engine=EngineConfig(engine_type="mock")
    )
    pipeline = OCRPipeline(config=cfg)
    result = pipeline.process(img)

    assert result.engine == "mock"
    assert len(result.text) > 0
    assert len(result.words) > 0
    assert "preprocessing" in result.metadata
    assert "entities" in result.metadata
    assert result.processing_time_ms >= 0


def test_pipeline_entity_extraction():
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    custom_text = "Contact support@example.com or call 555-123-4567 on 2026-05-15. Amount: $199.99"
    cfg = PipelineConfig(
        engine=EngineConfig(engine_type="mock")
    )
    pipeline = OCRPipeline(config=cfg)
    # Inject predefined text
    pipeline.engine.predefined_text = custom_text

    result = pipeline.process(img)
    entities = result.metadata["entities"]

    assert "support@example.com" in entities["emails"]
    assert len(entities["phone_numbers"]) > 0
    assert "2026-05-15" in entities["dates"]
    assert "$199.99" in entities["amounts"]


def test_cli_execution_json(tmp_path):
    sample_img = tmp_path / "test.png"
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    img.save(sample_img)

    out_file = tmp_path / "out.json"
    code = main([
        "-i", str(sample_img),
        "-e", "mock",
        "-f", "json",
        "-o", str(out_file)
    ])
    assert code == 0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert '"engine": "mock"' in content
