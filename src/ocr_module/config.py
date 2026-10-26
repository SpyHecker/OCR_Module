"""Configuration settings for the OCR module."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class PreprocessingConfig(BaseModel):
    """Image preprocessing parameters."""
    enabled: bool = True
    target_dpi: int = 300
    deskew: bool = True
    denoise: bool = True
    binarize: bool = True
    binarize_method: str = Field(default="otsu", description="Either 'otsu' or 'adaptive'")
    clahe_contrast: bool = True


class EngineConfig(BaseModel):
    """Engine selection and engine-specific parameters."""
    engine_type: str = Field(
        default="auto",
        description="Engine type: 'auto', 'baseline', 'tesseract', 'transformers', or 'mock'"
    )
    languages: List[str] = Field(default_factory=lambda: ["eng"])
    tesseract_cmd: Optional[str] = Field(
        default=None,
        description="Path to tesseract binary if not on system PATH."
    )
    confidence_threshold: float = Field(default=0.0, ge=0.0, le=100.0)


class PipelineConfig(BaseModel):
    """Root configuration for the full OCR pipeline."""
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    clean_text: bool = True
    sort_reading_order: bool = True
