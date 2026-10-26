"""Mock OCR engine for testing, demonstrations, and offline verification."""

from __future__ import annotations

import time
from typing import Optional, List
from PIL import Image

from ocr_module.core.base import BaseOCREngine
from ocr_module.core.types import BoundingBox, OCRLine, OCRResult, OCRWord


class MockOCREngine(BaseOCREngine):
    """Deterministic mock OCR engine used for unit testing and CI."""

    def __init__(self, predefined_text: Optional[str] = None):
        super().__init__(name="mock")
        self.predefined_text = predefined_text

    def is_available(self) -> bool:
        return True

    def process_image(self, image: Image.Image) -> OCRResult:
        start_time = time.perf_counter()
        w, h = image.size

        text_content = self.predefined_text or "OCR MODULE SAMPLE DOCUMENT\nInvoice Number: #98234\nTotal Due: $450.00"
        lines_raw = text_content.splitlines()

        words: List[OCRWord] = []
        lines: List[OCRLine] = []

        line_height = max(20, h // (len(lines_raw) + 2))
        y_offset = 20

        for line_str in lines_raw:
            tokens = line_str.split()
            line_words: List[OCRWord] = []
            x_offset = 20

            for token in tokens:
                token_width = max(30, len(token) * 12)
                box = BoundingBox(x=x_offset, y=y_offset, w=token_width, h=line_height)
                word = OCRWord(text=token, confidence=98.5, box=box)
                words.append(word)
                line_words.append(word)
                x_offset += token_width + 10

            line_box = BoundingBox(
                x=20,
                y=y_offset,
                w=max(50, x_offset - 20),
                h=line_height
            )
            lines.append(OCRLine(
                text=line_str,
                confidence=98.5,
                words=line_words,
                box=line_box
            ))
            y_offset += line_height + 10

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return OCRResult(
            text=text_content,
            engine=self.name,
            confidence=98.5,
            words=words,
            lines=lines,
            processing_time_ms=elapsed_ms,
            metadata={"mock": True, "image_size": [w, h]}
        )
