# test_paragraph_icr.py - Test ICR on full paragraphs with multiple lines

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import sys
import numpy as np
from pathlib import Path
from typing import List

# Ensure project root is on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.icr.inference import ICRPredictor
from src.recognition.icr_block_engine import BlockICREngine
from src.nlp.block_parser import BlockTextParser
from src.nlp.lexicon_beam_decoder import LexiconBeamDecoder, load_english_words
from src.icr.preprocessing import preprocess_single_char

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MODEL_PATH = "models/icr_model"
PARAGRAPHS_DIR = Path("data/icr_training/scanned/paragraphs")
TARGET_SIZE = (28, 28)
SPACE_THRESHOLD = 1.5  # Gap multiplier to detect spaces between words
LINE_HEIGHT_THRESHOLD = 0.5  # Multiplier for detecting new lines
MIN_CHAR_AREA = 60
TOP_K_CANDIDATES = 6
BEAM_WIDTH = 50
PARAGRAPH_LEXICON_TERMS = [
    "patient",
    "aspirin",
    "diabetes",
    "metformin",
    "hypertension",
    "discharge",
    "prescription",
    "diagnosis",
    "report",
    "hospital",
    "instructions",
    "treatment",
    "medication",
    "viral",
    "uti",
    "tablet",
    "capsule",
    "daily",
    "twice",
    "mg",
    "ml",
    "blood",
    "pressure",
    "follow",
    "review",
]

PARAGRAPH_COMMON_TERMS = [
    "the", "and", "for", "with", "is", "in", "to", "of", "on", "a", "an",
    "take", "after", "before", "morning", "night", "days", "weeks", "pain",
    "fever", "cough", "doctor", "advice", "report", "test", "result",
]

ENGLISH_WORDS_PATH = Path("data/dictionaries/english_words_alpha.txt")
BROAD_ENGLISH_TERMS = load_english_words(
    ENGLISH_WORDS_PATH,
    max_words=50000,
    min_len=3,
    max_len=12,
)

ENGLISH_TERMS = sorted(set(PARAGRAPH_COMMON_TERMS + BROAD_ENGLISH_TERMS))

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

predictor = ICRPredictor(model_path=MODEL_PATH)
parser = BlockTextParser(
    dictionary_terms=PARAGRAPH_LEXICON_TERMS,
    english_terms=ENGLISH_TERMS,
    similarity_cutoff=0.76,
    english_similarity_cutoff=0.86,
    english_fuzzy_requires_ocr_noise=True,
)
engine = BlockICREngine()

lexicon_terms = list(PARAGRAPH_LEXICON_TERMS) + list(ENGLISH_TERMS)
decoder = LexiconBeamDecoder(
    lexicon_terms,
    primary_terms=PARAGRAPH_LEXICON_TERMS,
    max_edit_distance=2,
    replacement_confidence_threshold=0.90,
    replacement_min_char_confidence_threshold=0.74,
    non_primary_replacement_min_char_confidence=0.42,
)

# --------------------------------------------------
# LINE SEGMENTATION
# --------------------------------------------------

def segment_lines(paragraph_img):
    """
    Segment paragraph into individual text lines.
    
    Returns:
        List of (line_image, y_position) tuples
    """
    gray = cv2.cvtColor(paragraph_img, cv2.COLOR_BGR2GRAY)
    
    # Get horizontal projection (sum of pixels in each row)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horizontal_projection = np.sum(thresh, axis=1)
    
    # Find gaps between lines (rows with very few pixels)
    threshold = np.max(horizontal_projection) * 0.1
    
    # Find line boundaries
    in_line = False
    line_start = 0
    lines = []
    
    for i, val in enumerate(horizontal_projection):
        if val > threshold and not in_line:
            # Start of a line
            line_start = i
            in_line = True
        elif val <= threshold and in_line:
            # End of a line
            line_end = i
            if line_end - line_start > 10:  # Minimum line height
                line_img = paragraph_img[line_start:line_end, :]
                lines.append((line_img, line_start))
            in_line = False
    
    # Handle last line if still in a line
    if in_line and len(horizontal_projection) - line_start > 10:
        line_img = paragraph_img[line_start:, :]
        lines.append((line_img, line_start))
    
    return lines

# --------------------------------------------------
# CHARACTER SEGMENTATION WITH SPACE DETECTION
# --------------------------------------------------

def segment_line(line_img, space_threshold=SPACE_THRESHOLD, min_char_area=MIN_CHAR_AREA):
    """
    Segment characters from a single line and detect spaces.
    
    Returns:
        List of tuples: (character_image, is_space, x_position)
    """
    gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)

    # Threshold
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Get bounding boxes
    boxes = []
    areas = [cv2.contourArea(c) for c in contours]
    area_threshold = min_char_area
    if areas:
        area_threshold = max(MIN_CHAR_AREA, int(np.percentile(np.array(areas), 20) * 0.6))

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > area_threshold:  # Filter noise with adaptive threshold
            boxes.append((x, y, w, h))

    # Sort by x position
    boxes.sort(key=lambda b: b[0])
    
    if not boxes:
        return []
    
    # Calculate average character width
    avg_width = np.mean([w for (x, y, w, h) in boxes])
    
    # Extract characters and detect spaces
    results = []
    for i, (x, y, w, h) in enumerate(boxes):
        char_img = line_img[y:y+h, x:x+w]
        results.append((char_img, False, x))
        
        # Check gap to next character
        if i < len(boxes) - 1:
            next_x = boxes[i + 1][0]
            gap = next_x - (x + w)
            
            # Detect space
            if gap > space_threshold * avg_width:
                results.append((None, True, x + w))
    
    return results

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

def preprocess_char(char_img, target_size=(28, 28)):
    """Preprocess character to match training data format."""
    return preprocess_single_char(char_img, target_size=target_size)


def decode_line_with_strategy(line_img, line_idx, debug_dir, space_threshold, min_char_area):
    """Decode one line using a specific segmentation strategy and return text + confidence."""
    char_data = segment_line(line_img, space_threshold=space_threshold, min_char_area=min_char_area)
    if not char_data:
        return "", 0.0

    line_text = []
    char_index = 0
    current_word_candidates: List[List[dict]] = []
    confidence_samples: List[float] = []

    def flush_word_candidates():
        if not current_word_candidates:
            return
        decoded = decoder.decode_word(current_word_candidates, beam_width=BEAM_WIDTH)
        line_text.append(decoded["decoded_word"])
        confidence_samples.append(float(decoded.get("top1_mean_confidence", 0.0)))
        current_word_candidates.clear()

    for char_img, is_space, _ in char_data:
        if is_space:
            flush_word_candidates()
            line_text.append(" ")
        else:
            processed = preprocess_char(char_img, TARGET_SIZE)
            candidates = engine.predict_char_candidates(processed, top_k=TOP_K_CANDIDATES)
            if candidates:
                current_word_candidates.append(candidates)

            cv2.imwrite(
                str(debug_dir / f"line{line_idx:02d}_char{char_index:03d}.png"),
                processed,
            )
            char_index += 1

    flush_word_candidates()

    text = "".join(line_text)
    avg_conf = (sum(confidence_samples) / len(confidence_samples)) if confidence_samples else 0.0
    return text, avg_conf

# --------------------------------------------------
# PARAGRAPH PREDICTION
# --------------------------------------------------

def predict_paragraph(image_path: Path, space_threshold=SPACE_THRESHOLD):
    """Predict text from paragraph image."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: Cannot read image {image_path}")
        return None

    # Segment into lines
    lines = segment_lines(img)
    
    if not lines:
        print(f"WARNING: No lines detected in {image_path.name}")
        return ""
    
    print(f"Detected {len(lines)} lines")
    
    predicted_lines = []
    
    # Debug directory
    debug_dir = Path("data/processed/paragraph_chars") / image_path.stem
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Process each line
    for line_idx, (line_img, y_pos) in enumerate(lines):
        strategies = [
            (space_threshold, MIN_CHAR_AREA),
            (max(1.2, space_threshold - 0.2), max(40, MIN_CHAR_AREA - 15)),
            (space_threshold + 0.25, MIN_CHAR_AREA + 15),
        ]

        best_text = ""
        best_conf = -1.0

        for strategy_space, strategy_area in strategies:
            text, conf = decode_line_with_strategy(
                line_img,
                line_idx,
                debug_dir,
                space_threshold=strategy_space,
                min_char_area=strategy_area,
            )
            if conf > best_conf:
                best_text = text
                best_conf = conf

        if best_text:
            predicted_lines.append(best_text)
    
    return "\n".join(predicted_lines)

# --------------------------------------------------
# RUN ON ALL PARAGRAPH IMAGES
# --------------------------------------------------

if __name__ == "__main__":
    
    if not PARAGRAPHS_DIR.exists():
        print(f"ERROR: Paragraphs directory not found: {PARAGRAPHS_DIR}")
        print("Please create the directory and add paragraph images.")
        exit(1)
    
    # Get all image files
    image_files = list(PARAGRAPHS_DIR.glob("*.png")) + \
                  list(PARAGRAPHS_DIR.glob("*.jpg")) + \
                  list(PARAGRAPHS_DIR.glob("*.jpeg"))
    
    if not image_files:
        print(f"No images found in {PARAGRAPHS_DIR}")
        exit(1)
    
    # Optional CLI selector:
    # - latest : run only the most recently modified file
    # - <filename> : run only that file in paragraph folder
    # - <absolute/relative path> : run only that path if it exists
    selected = None
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        if arg.lower() == "latest":
            selected = max(image_files, key=lambda p: p.stat().st_mtime)
        else:
            candidate = Path(arg)
            if candidate.exists() and candidate.is_file():
                selected = candidate
            else:
                named = PARAGRAPHS_DIR / arg
                if named.exists() and named.is_file():
                    selected = named

    if selected is not None:
        image_files = [selected]

    print(f"\n{'='*60}")
    print(f"PARAGRAPH ICR TEST")
    print(f"{'='*60}")
    print(f"Directory: {PARAGRAPHS_DIR}")
    print(f"Images found: {len(image_files)}")
    print(f"Space threshold: {SPACE_THRESHOLD}x average character width")
    print(f"{'='*60}\n")
    
    for img_path in sorted(image_files):
        print(f"\nImage: {img_path.name}")
        print("-" * 60)
        
        result = predict_paragraph(img_path, SPACE_THRESHOLD)
        
        if result is not None:
            print("Predicted text:")
            print(result)
            parsed = parser.parse(result)
            print("\nParsed text:")
            print(parsed["corrected_text"])
            print("\nParser backend:", parsed["backend"])
            print("Dictionary matches:", parsed["dictionary_matches"])
            print("Corrections:", parsed["corrections"])
            print(f"\nTotal lines: {len(result.split(chr(10)))}")
            print(f"Total characters: {len(result)}")
    
    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"Debug images saved to: data/processed/paragraph_chars/")
    print(f"{'='*60}\n")
