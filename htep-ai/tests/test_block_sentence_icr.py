import cv2
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.recognition.icr_block_engine import BlockICREngine
from src.nlp.block_parser import BlockTextParser
from src.segmentation.word_segmenter import segment_words


PARSER_TERMS = [
    "hello", "my", "name", "patient", "aspirin", "diabetes", "metformin",
    "hypertension", "discharge", "report", "prescription", "diagnosis",
]


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
        word_pred = engine.predict_word(word_img)
        sentence.append(word_pred["text"])

    return " ".join(sentence)


if __name__ == "__main__":
    IMAGE_PATH = "data/icr_training/scanned/sentences/hellomynameisnilesh.png"
    parser = BlockTextParser(dictionary_terms=PARSER_TERMS)

    raw_text = predict_sentence(IMAGE_PATH)
    parsed = parser.parse(raw_text)

    print("\n==============================")
    print("RAW SENTENCE:")
    print(raw_text)
    print("\nPARSED SENTENCE:")
    print(parsed["corrected_text"])
    print("\nPARSER BACKEND:", parsed["backend"])
    print("DICTIONARY MATCHES:", parsed["dictionary_matches"])
    print("CORRECTIONS:", parsed["corrections"])
    print("==============================\n")