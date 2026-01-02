# src/segmentation/word_segmenter.py
# Word segmentation using OpenCV (BLOCK text)

import cv2
import numpy as np


def segment_words(sentence_img, min_area=500):
    """
    Segment words from a sentence image.
    SAME contour-based logic you used earlier.
    """

    gray = cv2.cvtColor(sentence_img, cv2.COLOR_BGR2GRAY)

    # Threshold
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Join letters into words
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

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

    # Sort words left → right
    boxes.sort(key=lambda b: b[0])

    word_images = [
        sentence_img[y:y + h, x:x + w]
        for (x, y, w, h) in boxes
    ]

    return word_images
