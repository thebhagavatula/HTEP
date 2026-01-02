# src/icr/inference.py
# High-level ICR inference wrapper (BLOCK now, CURSIVE later)

import cv2
from typing import List, Dict
from src.recognition.icr_block_engine import BlockICREngine

# Future import
# from src.recognition.icr_cursive_engine import CursiveICREngine


class ICRInference:
    """
    Unified interface for ICR inference.
    Decides which model to use and standardizes output.
    """

    def __init__(self):
        self.block_engine = BlockICREngine()

        # Placeholder for future
        self.cursive_engine = None
        # self.cursive_engine = CursiveICREngine()

    # --------------------------------------------------
    # SINGLE CHARACTER
    # --------------------------------------------------

    def predict_char(self, char_img, style: str = "block") -> Dict:
        """
        Predict a single character image.

        style: 'block' | 'cursive'
        """
        if style == "block":
            return self.block_engine.predict_char(char_img)

        elif style == "cursive":
            if not self.cursive_engine:
                raise NotImplementedError("Cursive ICR not available yet")
            return self.cursive_engine.predict_char(char_img)

        else:
            raise ValueError(f"Unknown ICR style: {style}")

    # --------------------------------------------------
    # WORD (LIST OF CHAR IMAGES)
    # --------------------------------------------------

    def predict_word(self, char_images: List, style: str = "block") -> Dict:
        """
        Predict a word from segmented character images.
        """
        if style == "block":
            return self.block_engine.predict_word(char_images)

        elif style == "cursive":
            if not self.cursive_engine:
                raise NotImplementedError("Cursive ICR not available yet")
            return self.cursive_engine.predict_word(char_images)

        else:
            raise ValueError(f"Unknown ICR style: {style}")

    # --------------------------------------------------
    # IMAGE PATH (CONVENIENCE)
    # --------------------------------------------------

    def predict_char_from_path(self, image_path: str, style: str = "block") -> Dict:
        """
        Convenience wrapper: load image → predict.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        return self.predict_char(img, style=style)
