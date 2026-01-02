# tests/test_icr_block.py

import cv2
from src.recognition.icr_block_engine import BlockICREngine


def test_single_character():
    engine = BlockICREngine()
    img = cv2.imread("tests/sample_char.png", cv2.IMREAD_GRAYSCALE)

    result = engine.predict_char(img)
    assert "character" in result
    assert "confidence" in result
