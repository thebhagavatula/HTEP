# src/pipeline/fusion.py
# OCR + ICR confidence-based correction

import re


class OCRICRFusion:
    """
    Applies lightweight rule-based correction.
    (ICR-assisted logic comes later)
    """

    def correct_text(self, text: str) -> str:
        if not text:
            return ""

        words = text.split()
        corrected_words = []

        for word in words:
            # Numeric context
            if re.match(r'^\d+(\.\d+)?$', word):
                corrected = (
                    word.replace('O', '0')
                        .replace('l', '1')
                        .replace('I', '1')
                )
            else:
                corrected = (
                    word.replace('0', 'O')
                        .replace('1', 'l')
                )

            corrected_words.append(corrected)

        return " ".join(corrected_words)
