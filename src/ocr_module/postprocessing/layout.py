"""Spatial layout analysis and reading order reconstruction."""

from __future__ import annotations

from typing import List
from ocr_module.core.types import OCRWord


def sort_words_reading_order(words: List[OCRWord], line_threshold_px: int = 10) -> List[OCRWord]:
    """Sort recognized words into natural human reading order (top-to-bottom, left-to-right).

    Groups words into approximate horizontal lines using line_threshold_px, then sorts
    each line by its horizontal x coordinate.

    Args:
        words: List of OCRWord tokens with bounding boxes.
        line_threshold_px: Maximum vertical gap before starting a new line bucket.

    Returns:
        Sorted list of OCRWord instances.
    """
    if not words:
        return []

    # Filter words with boxes
    boxed = [w for w in words if w.box is not None]
    unboxed = [w for w in words if w.box is None]

    # Primary sort by top y coordinate
    boxed.sort(key=lambda w: w.box.y)

    lines: List[List[OCRWord]] = []
    current_line: List[OCRWord] = []
    current_y: float = -1.0

    for word in boxed:
        box = word.box
        if current_y < 0 or abs(box.y - current_y) <= line_threshold_px:
            current_line.append(word)
            current_y = (current_y + box.y) / 2.0 if current_y >= 0 else box.y
        else:
            # Sort previous line left-to-right
            current_line.sort(key=lambda w: w.box.x)
            lines.append(current_line)
            current_line = [word]
            current_y = box.y

    if current_line:
        current_line.sort(key=lambda w: w.box.x)
        lines.append(current_line)

    flattened = [w for line in lines for w in line]
    return flattened + unboxed
