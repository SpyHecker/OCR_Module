"""Deep learning Vision-Transformer OCR engine (TrOCR/HuggingFace)."""

from __future__ import annotations

import time
from typing import Optional
from PIL import Image

from ocr_module.core.base import BaseOCREngine
from ocr_module.core.types import BoundingBox, OCRLine, OCRResult, OCRWord


class VisionTransformerEngine(BaseOCREngine):
    """Transformer-based end-to-end OCR recognizer using TrOCR architecture."""

    def __init__(self, model_name: str = "microsoft/trocr-small-printed", device: Optional[str] = None):
        super().__init__(name="vision_transformer")
        self.model_name = model_name
        self._device = device
        self._processor = None
        self._model = None
        self._torch = None

    def is_available(self) -> bool:
        """Check if torch and transformers packages are importable."""
        try:
            import torch
            import transformers
            return True
        except ImportError:
            return False

    def _lazy_load(self):
        """Lazy load model weights only upon request."""
        if self._model is not None:
            return

        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self._torch = torch
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._processor = TrOCRProcessor.from_pretrained(self.model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self._device)
        self._model.eval()

    def process_image(self, image: Image.Image) -> OCRResult:
        """Process image using Vision Transformer."""
        if not self.is_available():
            raise RuntimeError("torch and transformers must be installed for VisionTransformerEngine.")

        start_time = time.perf_counter()
        self._lazy_load()

        # Convert to RGB
        rgb_image = image.convert("RGB")
        pixel_values = self._processor(rgb_image, return_tensors="pt").pixel_values.to(self._device)

        with self._torch.no_grad():
            generated_ids = self._model.generate(pixel_values)

        extracted_text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        w, h = image.size

        # Build word tokens
        words = []
        tokens = extracted_text.split()
        for token in tokens:
            words.append(OCRWord(text=token, confidence=95.0, box=BoundingBox(x=0, y=0, w=w, h=h)))

        lines = [
            OCRLine(
                text=extracted_text,
                confidence=95.0,
                words=words,
                box=BoundingBox(x=0, y=0, w=w, h=h)
            )
        ] if extracted_text else []

        return OCRResult(
            text=extracted_text,
            engine=self.name,
            confidence=95.0 if extracted_text else 0.0,
            words=words,
            lines=lines,
            processing_time_ms=elapsed_ms,
            metadata={"model": self.model_name, "device": str(self._device)}
        )
