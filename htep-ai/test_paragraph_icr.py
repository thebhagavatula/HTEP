# test_paragraph_icr.py - Test ICR on full paragraphs with multiple lines

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
PARAGRAPHS_DIR = Path("data/icr_training/scanned/paragraphs")
TARGET_SIZE = (28, 28)
SPACE_THRESHOLD = 1.5  # Gap multiplier to detect spaces between words
LINE_HEIGHT_THRESHOLD = 0.5  # Multiplier for detecting new lines

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

predictor = ICRPredictor(model_path=MODEL_PATH)

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

def segment_line(line_img, space_threshold=SPACE_THRESHOLD):
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
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 100:  # Filter noise
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
    gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if np.sum(thresh == 0) < np.sum(thresh == 255):
        thresh = 255 - thresh

    final = cv2.resize(thresh, target_size, interpolation=cv2.INTER_AREA)
    return final

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
        # Segment characters in line
        char_data = segment_line(line_img, space_threshold)
        
        if not char_data:
            continue
        
        line_text = []
        char_index = 0
        
        for char_img, is_space, x_pos in char_data:
            if is_space:
                line_text.append(" ")
            else:
                # Preprocess and predict
                processed = preprocess_char(char_img, TARGET_SIZE)
                pred = predictor.predict_array(processed)
                
                line_text.append(pred["character"])
                
                # Save debug image
                cv2.imwrite(
                    str(debug_dir / f"line{line_idx:02d}_char{char_index:03d}.png"), 
                    processed
                )
                char_index += 1
        
        predicted_lines.append("".join(line_text))
    
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
            print(f"\nTotal lines: {len(result.split(chr(10)))}")
            print(f"Total characters: {len(result)}")
    
    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"Debug images saved to: data/processed/paragraph_chars/")
    print(f"{'='*60}\n")
