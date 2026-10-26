"""Data structures and type definitions for OCR Module."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box representing a text region in pixel coordinates."""
    x: int = Field(..., description="Top-left x coordinate")
    y: int = Field(..., description="Top-left y coordinate")
    w: int = Field(..., description="Width of bounding box")
    h: int = Field(..., description="Height of bounding box")

    @property
    def x_max(self) -> int:
        return self.x + self.w

    @property
    def y_max(self) -> int:
        return self.y + self.h

    @property
    def center(self) -> Tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Returns (x, y, w, h)."""
        return (self.x, self.y, self.w, self.h)

    def to_ltrb(self) -> Tuple[int, int, int, int]:
        """Returns (left, top, right, bottom)."""
        return (self.x, self.y, self.x_max, self.y_max)


class OCRWord(BaseModel):
    """Single recognized word with spatial coordinates and confidence."""
    text: str = Field(..., description="Recognized word text")
    confidence: float = Field(default=100.0, ge=0.0, le=100.0, description="Confidence score 0-100")
    box: Optional[BoundingBox] = Field(default=None, description="Spatial coordinates")


class OCRLine(BaseModel):
    """Grouped line of text."""
    text: str = Field(..., description="Full line text")
    confidence: float = Field(default=100.0, description="Average confidence score")
    words: List[OCRWord] = Field(default_factory=list, description="Word tokens in the line")
    box: Optional[BoundingBox] = Field(default=None, description="Enclosing bounding box for the line")


class OCRResult(BaseModel):
    """Unified OCR output structure."""
    text: str = Field(..., description="Full reconstructed text")
    engine: str = Field(default="unknown", description="Name of OCR engine used")
    confidence: float = Field(default=100.0, description="Overall document confidence score")
    words: List[OCRWord] = Field(default_factory=list, description="All recognized word tokens")
    lines: List[OCRLine] = Field(default_factory=list, description="Lines ordered by reading flow")
    processing_time_ms: float = Field(default=0.0, description="Inference latency in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Supplementary image & processing metadata")

    def to_dict(self) -> Dict[str, Any]:
        """Export as plain dictionary."""
        return self.model_dump()
