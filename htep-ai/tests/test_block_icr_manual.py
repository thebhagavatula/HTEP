import cv2
from src.recognition.icr_block_engine import BlockICREngine

engine = BlockICREngine()

img = cv2.imread("tests/A.png", cv2.IMREAD_GRAYSCALE)
result = engine.predict_char(img)

print(result)
