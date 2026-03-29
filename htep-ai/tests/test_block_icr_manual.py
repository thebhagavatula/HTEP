import cv2
import numpy as np
from pathlib import Path

from src.recognition.icr_block_engine import BlockICREngine


def _load_or_create_char_image():
	fixture = Path("data/icr_training/scanned/words/Hello.png")

	if fixture.exists():
		img = cv2.imread(str(fixture), cv2.IMREAD_GRAYSCALE)
		if img is not None:
			return img

	# Fallback synthetic glyph image to keep test self-contained.
	canvas = np.full((64, 64), 255, dtype=np.uint8)
	cv2.putText(canvas, "A", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
	return canvas


def test_block_icr_manual_predict_char():
	engine = BlockICREngine()
	img = _load_or_create_char_image()

	result = engine.predict_char(img)

	assert "character" in result
	assert "confidence" in result
