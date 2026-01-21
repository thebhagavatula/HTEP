# src/segmentation/char_segmenter.py
# Character segmentation using OpenCV (BLOCK text)

import cv2


def segment_characters(word_img, min_area=200):
    """
    Segment characters from a word image.
    SAME logic as your old test_word_icr.py
    """

    gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)

    # Binary (invert for white text on black bg)
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Merge strokes slightly
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > min_area:
            boxes.append((x, y, w, h))

    # Sort left → right
    boxes.sort(key=lambda b: b[0])

    # Crop characters
    char_images = [
        word_img[y:y + h, x:x + w]
        for (x, y, w, h) in boxes
    ]

    return char_images

def segment_characters_with_spaces(word_img, space_threshold=1.0):
    """
    Segment characters and detect spaces between them.
    Returns list of (char_img | None, is_space)
    """

    gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 100:
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])
    if not boxes:
        return []

    avg_width = sum(w for (_, _, w, _) in boxes) / len(boxes)

    results = []

    for i, (x, y, w, h) in enumerate(boxes):
        char_img = word_img[y:y+h, x:x+w]
        results.append((char_img, False))

        if i < len(boxes) - 1:
            next_x = boxes[i+1][0]
            gap = next_x - (x + w)
            if gap > space_threshold * avg_width:
                results.append((None, True))

    return results
