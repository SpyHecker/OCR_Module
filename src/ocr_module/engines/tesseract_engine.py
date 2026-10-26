"""Tesseract OCR engine implementation using pytesseract."""

from __future__ import annotations

import shutil
import time
from typing import List, Optional
from PIL import Image
import pytesseract

from ocr_module.core.base import BaseOCREngine
from ocr_module.core.types import BoundingBox, OCRLine, OCRResult, OCRWord


class TesseractEngine(BaseOCREngine):
    """Wrapper around Google Tesseract OCR via pytesseract."""

    def __init__(self, tesseract_cmd: Optional[str] = None, languages: Optional[List[str]] = None):
        super().__init__(name="tesseract")
        self.languages = "+".join(languages) if languages else "eng"
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def is_available(self) -> bool:
        """Check if tesseract binary can be executed."""
        try:
            # Check configured path or PATH
            cmd = pytesseract.pytesseract.tesseract_cmd
            if shutil.which(cmd):
                return True
            # Try running version
            _ = pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def process_image(self, image: Image.Image) -> OCRResult:
        """Run Tesseract recognition and extract words, lines, and bounding boxes."""
        if not self.is_available():
            raise RuntimeError(
                "Tesseract binary is not installed or not found on system PATH. "
                "Please install Tesseract OCR from https://github.com/UB-Mannheim/tesseract/wiki "
                "or specify `tesseract_cmd` in EngineConfig."
            )

        start_time = time.perf_counter()
        # image_to_data returns detailed tsv with coordinates, levels, and confidence
        data = pytesseract.image_to_data(
            image,
            lang=self.languages,
            output_type=pytesseract.Output.DICT
        )

        n_boxes = len(data["text"])
        words: List[OCRWord] = []
        confidences: List[float] = []

        # Track line groupings by (block_num, par_num, line_num)
        lines_dict = {}

        for i in range(n_boxes):
            text = data["text"][i].strip()
            conf = float(data["conf"][i])

            # Filter out non-word empty segments or negative confidence tokens
            if not text or conf < 0:
                continue

            confidences.append(conf)
            box = BoundingBox(
                x=int(data["left"][i]),
                y=int(data["top"][i]),
                w=int(data["width"][i]),
                h=int(data["height"][i])
            )
            word = OCRWord(text=text, confidence=conf, box=box)
            words.append(word)

            line_key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            if line_key not in lines_dict:
                lines_dict[line_key] = []
            lines_dict[line_key].append(word)

        lines: List[OCRLine] = []
        full_text_lines = []

        for _, line_words in lines_dict.items():
            if not line_words:
                continue
            line_text = " ".join(w.text for w in line_words)
            full_text_lines.append(line_text)
            avg_conf = sum(w.confidence for w in line_words) / len(line_words)

            # Compute bounding box that spans the whole line
            min_x = min(w.box.x for w in line_words if w.box)
            min_y = min(w.box.y for w in line_words if w.box)
            max_x = max(w.box.x_max for w in line_words if w.box)
            max_y = max(w.box.y_max for w in line_words if w.box)
            line_box = BoundingBox(x=min_x, y=min_y, w=max_x - min_x, h=max_y - min_y)

            lines.append(OCRLine(
                text=line_text,
                confidence=avg_conf,
                words=line_words,
                box=line_box
            ))

        full_text = "\n".join(full_text_lines)
        overall_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return OCRResult(
            text=full_text,
            engine=self.name,
            confidence=overall_conf,
            words=words,
            lines=lines,
            processing_time_ms=elapsed_ms,
            metadata={"languages": self.languages, "total_words": len(words)}
        )
