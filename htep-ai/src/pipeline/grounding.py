# src/pipeline/grounding.py
"""
Checks whether a value read by the vision model appears in the OCR text.

A VLM can invent plausible medicines or numbers; OCR cannot. A VLM value that
also appears (allowing small OCR errors) in the OCR text is "grounded" and
treated as verified. Anything else is kept, but flagged for human review.
"""

import re

from rapidfuzz import fuzz

FUZZY_THRESHOLD = 85


def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w/.%]+", " ", text)  # keep units like mg/ml, 0.5, 5%
    return " ".join(text.split())


def is_grounded(value, ocr_text: str, threshold: int = FUZZY_THRESHOLD) -> bool:
    """True if `value` is found in `ocr_text`, exactly or with minor OCR noise."""
    if value is None:
        return False
    needle = _normalise(str(value))
    haystack = _normalise(ocr_text or "")
    if not needle or not haystack:
        return False

    # Short values ("45", "M", "OD") only count as whole-word matches:
    # fuzzy matching them would hit almost anything.
    if len(needle) <= 3:
        return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack) is not None

    if needle in haystack:
        return True
    return fuzz.partial_ratio(needle, haystack) >= threshold
