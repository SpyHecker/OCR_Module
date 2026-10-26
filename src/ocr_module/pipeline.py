"""Main OCR Pipeline coordinating preprocessing, recognition, and postprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import numpy as np
from PIL import Image

from ocr_module.config import PipelineConfig, EngineConfig, PreprocessingConfig
from ocr_module.core.types import OCRResult
from ocr_module.core.base import ImageInput
from ocr_module.engines.factory import OCREngineFactory
from ocr_module.preprocessing.preprocessor import ImagePreprocessor
from ocr_module.postprocessing.cleaners import TextCleaner, EntityExtractor
from ocr_module.postprocessing.layout import sort_words_reading_order


class OCRPipeline:
    """End-to-end OCR processing pipeline."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        engine_type: Optional[str] = None,
        enable_preprocessing: Optional[bool] = None,
    ):
        self.config = config or PipelineConfig()
        if engine_type:
            self.config.engine.engine_type = engine_type
        if enable_preprocessing is not None:
            self.config.preprocessing.enabled = enable_preprocessing

        # Initialize core components
        self.preprocessor = ImagePreprocessor(self.config.preprocessing)
        self.engine = OCREngineFactory.create(self.config.engine)

    def _load_image(self, image_input: ImageInput) -> Image.Image:
        """Convert various input types into a standardized PIL Image."""
        if isinstance(image_input, Image.Image):
            return image_input.copy()
        elif isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            return Image.open(path)
        elif isinstance(image_input, np.ndarray):
            # Check channels
            if len(image_input.shape) == 2:
                return Image.fromarray(image_input)
            elif len(image_input.shape) == 3 and image_input.shape[2] == 3:
                # Assume RGB
                return Image.fromarray(image_input)
            else:
                return Image.fromarray(image_input)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def process(self, image_input: ImageInput) -> OCRResult:
        """Execute end-to-end OCR pipeline on an image.

        Args:
            image_input: File path, Path, PIL Image, or numpy array.

        Returns:
            Structured OCRResult containing extracted text, word coordinates, and metadata.
        """
        raw_image = self._load_image(image_input)

        # 1. Computer Vision Preprocessing
        processed_image, prep_meta = self.preprocessor.process(raw_image)

        # 2. Text Recognition Engine
        result = self.engine.process_image(processed_image)

        # 3. Post-Processing & Layout Ordering
        words = result.words
        if self.config.sort_reading_order and words:
            words = sort_words_reading_order(words)

        # Filter by confidence threshold
        conf_thresh = self.config.engine.confidence_threshold
        if conf_thresh > 0.0:
            words = [w for w in words if w.confidence >= conf_thresh]

        # 4. Text Cleanup
        text = result.text
        if self.config.clean_text:
            text = TextCleaner.clean(text)

        # 5. Extract structured entities
        entities = EntityExtractor.extract_entities(text)

        # Merge metadata
        merged_meta = {
            **result.metadata,
            "preprocessing": prep_meta,
            "entities": entities,
        }

        return OCRResult(
            text=text,
            engine=result.engine,
            confidence=result.confidence,
            words=words,
            lines=result.lines,
            processing_time_ms=result.processing_time_ms,
            metadata=merged_meta,
        )

    def extract_text(self, image_input: ImageInput) -> str:
        """Convenience method returning extracted plain text only."""
        return self.process(image_input).text
