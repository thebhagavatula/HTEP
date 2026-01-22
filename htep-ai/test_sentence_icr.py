# test_sentence_icr.py - Test ICR on full sentences with space detection

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
from pathlib import Path
from src.icr.inference import ICRPredictor

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MODEL_PATH = "models/icr_model"
SENTENCES_DIR = Path("data/icr_training/scanned/sentences")
TARGET_SIZE = (28, 28)
SPACE_THRESHOLD = 1.5  # Gap multiplier to detect spaces

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

predictor = ICRPredictor(model_path=MODEL_PATH)

# --------------------------------------------------
# CHARACTER SEGMENTATION WITH SPACE DETECTION
# --------------------------------------------------

def segment_sentence(sentence_img, space_threshold=SPACE_THRESHOLD):
    """
    Segment characters from sentence and detect spaces based on gaps.
    
    Returns:
        List of tuples: (character_image, is_space, x_position)
    """
    gray = cv2.cvtColor(sentence_img, cv2.COLOR_BGR2GRAY)

    # Threshold to get binary image
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Get bounding boxes
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 100:  # Filter out noise
            boxes.append((x, y, w, h))

    # Sort by x position (left to right)
    boxes.sort(key=lambda b: b[0])
    
    if not boxes:
        return []
    
    # Calculate average character width for space detection
    avg_width = np.mean([w for (x, y, w, h) in boxes])
    
    # Extract characters and detect spaces
    results = []
    for i, (x, y, w, h) in enumerate(boxes):
        char_img = sentence_img[y:y+h, x:x+w]
        results.append((char_img, False, x))  # Add character
        
        # Check gap to next character
        if i < len(boxes) - 1:
            next_x = boxes[i + 1][0]
            gap = next_x - (x + w)
            
            # If gap is larger than threshold * avg_width, insert space
            if gap > space_threshold * avg_width:
                results.append((None, True, x + w))  # Add space marker
    
    return results

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

def preprocess_char(char_img, target_size=(28, 28)):
    """Preprocess character to match training data format."""
    gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if np.sum(thresh == 0) < np.sum(thresh == 255):
        thresh = 255 - thresh

    final = cv2.resize(thresh, target_size, interpolation=cv2.INTER_AREA)
    return final

# --------------------------------------------------
# SENTENCE PREDICTION
# --------------------------------------------------

def predict_sentence(image_path: Path, space_threshold=SPACE_THRESHOLD):
    """Predict text from sentence image."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: Cannot read image {image_path}")
        return None

    # Segment characters with space detection
    char_data = segment_sentence(img, space_threshold)
    
    if not char_data:
        print(f"WARNING: No characters detected in {image_path.name}")
        return ""
    
    predicted_text = []
    
    # Debug directory
    debug_dir = Path("data/processed/sentence_chars") / image_path.stem
    debug_dir.mkdir(parents=True, exist_ok=True)

    char_index = 0
    for char_img, is_space, x_pos in char_data:
        if is_space:
            predicted_text.append(" ")
        else:
            # Preprocess and predict
            processed = preprocess_char(char_img, TARGET_SIZE)
            pred = predictor.predict_array(processed)
            
            predicted_text.append(pred["character"])
            
            # Save debug image
            cv2.imwrite(str(debug_dir / f"char_{char_index:03d}_x{x_pos}.png"), processed)
            char_index += 1

    return "".join(predicted_text)

# --------------------------------------------------
# RUN ON ALL SENTENCE IMAGES
# --------------------------------------------------

if __name__ == "__main__":
    
    if not SENTENCES_DIR.exists():
        print(f"ERROR: Sentences directory not found: {SENTENCES_DIR}")
        print("Please create the directory and add sentence images.")
        exit(1)
    
    # Get all image files
    image_files = list(SENTENCES_DIR.glob("*.png")) + \
                  list(SENTENCES_DIR.glob("*.jpg")) + \
                  list(SENTENCES_DIR.glob("*.jpeg"))
    
    if not image_files:
        print(f"No images found in {SENTENCES_DIR}")
        exit(1)
    
    print(f"\n{'='*60}")
    print(f"SENTENCE ICR TEST")
    print(f"{'='*60}")
    print(f"Directory: {SENTENCES_DIR}")
    print(f"Images found: {len(image_files)}")
    print(f"Space threshold: {SPACE_THRESHOLD}x average character width")
    print(f"{'='*60}\n")
    
    for img_path in sorted(image_files):
        print(f"\nImage: {img_path.name}")
        print("-" * 60)
        
        result = predict_sentence(img_path, SPACE_THRESHOLD)
        
        if result is not None:
            print(f"Predicted: {result}")
            print(f"Length: {len(result)} characters")
    
    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"Debug images saved to: data/processed/sentence_chars/")
    print(f"{'='*60}\n")
