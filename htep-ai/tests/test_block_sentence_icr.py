import cv2
import sys
from pathlib import Path
from typing import List

# Ensure project root is on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.recognition.icr_block_engine import BlockICREngine
from src.nlp.block_parser import BlockTextParser
from src.nlp.lexicon_beam_decoder import LexiconBeamDecoder
from src.segmentation.word_segmenter import segment_words
from src.segmentation.char_segmenter import segment_characters
from src.icr.preprocessing import preprocess_single_char


PARSER_TERMS = [
    "hello", "my", "name", "patient", "aspirin", "diabetes", "metformin",
    "hypertension", "discharge", "report", "prescription", "diagnosis",
]

def decode_word_with_lexicon(word_img, engine, decoder, top_k=4):
    char_images = segment_characters(word_img)
    if not char_images:
        return ""

    candidate_lists: List[List[dict]] = []
    for char_img in char_images:
        processed = preprocess_single_char(char_img)
        candidates = engine.predict_char_candidates(processed, top_k=top_k)
        if candidates:
            candidate_lists.append(candidates)

    if not candidate_lists:
        return ""

    decoded = decoder.decode_word(candidate_lists, beam_width=25)
    return decoded["decoded_word"]


def predict_sentence(image_path):
    """
    Predict a full BLOCK-written sentence using:
    OpenCV → Word segmentation → Character segmentation → CNN
    """
    engine = BlockICREngine()
    parser = BlockTextParser(
        dictionary_terms=PARSER_TERMS,
        english_terms=PARSER_TERMS,
        similarity_cutoff=0.75,
        english_similarity_cutoff=0.8,
    )

    lexicon_terms = list(PARSER_TERMS)
    decoder = LexiconBeamDecoder(
        lexicon_terms,
        primary_terms=PARSER_TERMS,
        max_edit_distance=2,
        replacement_confidence_threshold=0.95,
        replacement_min_char_confidence_threshold=0.90,
    )

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    words = segment_words(img)

    sentence = []

    for word_img in words:
        word = decode_word_with_lexicon(word_img, engine, decoder, top_k=4)
        if word:
            sentence.append(word)

    raw_text = " ".join(sentence)
    parsed = parser.parse(raw_text)
    return raw_text, parsed


if __name__ == "__main__":
    IMAGE_PATH = "data/icr_training/scanned/sentences/hellomynameisnilesh.png"

    raw_text, parsed = predict_sentence(IMAGE_PATH)

    print("\n==============================")
    print("RAW SENTENCE:")
    print(raw_text)
    print("\nPARSED SENTENCE:")
    print(parsed["corrected_text"])
    print("\nPARSER BACKEND:", parsed["backend"])
    print("DICTIONARY MATCHES:", parsed["dictionary_matches"])
    print("CORRECTIONS:", parsed["corrections"])
    print("==============================\n")