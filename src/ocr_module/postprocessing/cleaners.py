"""Text cleaning and information extraction postprocessors."""

from __future__ import annotations

import re
from typing import Dict, List


class TextCleaner:
    """Cleans up common OCR artifacts and whitespace anomalies."""

    @staticmethod
    def clean(text: str) -> str:
        """Normalize line breaks, multiple spaces, and unicode quotes."""
        if not text:
            return ""

        # Normalize diverse whitespace
        cleaned = re.sub(r"[ \t]+", " ", text)
        # Normalize excessive newlines
        cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
        # Replace fancy typographic quotes with standard ASCII
        cleaned = (
            cleaned.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
            .replace("`", "'")
        )
        return cleaned.strip()


class EntityExtractor:
    """Extracts common structured entities from OCR text."""

    EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    PHONE_PATTERN = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
    DATE_PATTERN = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b")
    CURRENCY_PATTERN = re.compile(r"[$€£₹]\s*\d+(?:[.,]\d{2})?")

    @classmethod
    def extract_entities(cls, text: str) -> Dict[str, List[str]]:
        """Extract email, phone, date, and monetary figures."""
        return {
            "emails": cls.EMAIL_PATTERN.findall(text),
            "phone_numbers": cls.PHONE_PATTERN.findall(text),
            "dates": cls.DATE_PATTERN.findall(text),
            "amounts": cls.CURRENCY_PATTERN.findall(text),
        }
