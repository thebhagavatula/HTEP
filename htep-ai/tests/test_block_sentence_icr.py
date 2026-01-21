import cv2
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.recognition.icr_block_engine import BlockICREngine
from src.segmentation.word_segmenter import segment_words
from src.segmentation.char_segmenter import segment_characters


def predict_sentence(image_path):
    """
    Predict a full BLOCK-written sentence using:
    OpenCV → Word segmentation → Character segmentation → CNN
    """
    engine = BlockICREngine()

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    words = segment_words(img)

    sentence = []

    for word_img in words:
        chars = segment_characters(word_img)

        if not chars:
            continue

        word_pred = engine.predict_word(chars)
        sentence.append(word_pred["text"])

    return " ".join(sentence)


if __name__ == "__main__":
    IMAGE_PATH = "tests/shrey614.jpg"  # 👈 put your sentence image here

    result = predict_sentence(IMAGE_PATH)

    print("\n==============================")
    print("PREDICTED SENTENCE:")
    print(result)
    print("==============================\n")