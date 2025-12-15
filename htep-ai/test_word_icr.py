# test_word_icr.py

import cv2
import numpy as np
from pathlib import Path
from src.icr.inference import ICRPredictor

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MODEL_PATH = "models/icr_model"
WORDS_DIR = Path("data/icr_training/scanned/words")
TARGET_SIZE = (28, 28)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

predictor = ICRPredictor(model_path=MODEL_PATH)

# --------------------------------------------------
# CHARACTER SEGMENTATION (WORD LEVEL)
# --------------------------------------------------

def segment_characters(word_img):
    gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 200:
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])

    return [word_img[y:y+h, x:x+w] for (x, y, w, h) in boxes]

# --------------------------------------------------
# LIGHT PREPROCESS (NO RE-CROPPING)
# --------------------------------------------------

def preprocess_char_from_word(char_img, target_size=(28, 28)):
    gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if np.sum(thresh == 0) < np.sum(thresh == 255):
        thresh = 255 - thresh

    final = cv2.resize(thresh, target_size, interpolation=cv2.INTER_AREA)
    return final

# --------------------------------------------------
# WORD PREDICTION
# --------------------------------------------------

def predict_word(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"SKIP (cannot read): {image_path.name}")
        return None

    char_images = segment_characters(img)
    predicted_chars = []

    debug_dir = Path("data/processed/word_chars") / image_path.stem
    debug_dir.mkdir(parents=True, exist_ok=True)

    for i, char_img in enumerate(char_images):
        processed = preprocess_char_from_word(char_img, TARGET_SIZE)
        pred = predictor.predict_array(processed)

        predicted_chars.append(pred["character"])
        cv2.imwrite(str(debug_dir / f"char_{i}.png"), processed)

    return "".join(predicted_chars)

# --------------------------------------------------
# RUN ON ALL WORD IMAGES
# --------------------------------------------------

if __name__ == "__main__":

    if not WORDS_DIR.exists():
        raise FileNotFoundError(f"Words directory not found: {WORDS_DIR}")

    print(f"\nRunning word ICR on: {WORDS_DIR}\n")

    for img_path in WORDS_DIR.iterdir():
        if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            continue

        word = predict_word(img_path)

        if word is not None:
            print(f"{img_path.name}  →  {word}")
