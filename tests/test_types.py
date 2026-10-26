"""Unit tests for core data models."""

from ocr_module.core.types import BoundingBox, OCRWord, OCRLine, OCRResult


def test_bounding_box_geometry():
    box = BoundingBox(x=10, y=20, w=100, h=50)
    assert box.x_max == 110
    assert box.y_max == 70
    assert box.center == (60.0, 45.0)
    assert box.to_tuple() == (10, 20, 100, 50)
    assert box.to_ltrb() == (10, 20, 110, 70)


def test_ocr_word_and_line():
    box = BoundingBox(x=5, y=5, w=30, h=15)
    word = OCRWord(text="Hello", confidence=99.0, box=box)
    assert word.text == "Hello"
    assert word.confidence == 99.0

    line = OCRLine(text="Hello World", confidence=95.0, words=[word], box=box)
    assert len(line.words) == 1
    assert line.text == "Hello World"


def test_ocr_result_serialization():
    box = BoundingBox(x=0, y=0, w=10, h=10)
    word = OCRWord(text="Test", box=box)
    result = OCRResult(
        text="Test",
        engine="mock",
        confidence=98.0,
        words=[word],
        processing_time_ms=12.5,
        metadata={"custom": "value"}
    )
    d = result.to_dict()
    assert d["text"] == "Test"
    assert d["engine"] == "mock"
    assert len(d["words"]) == 1
    assert d["metadata"]["custom"] == "value"
