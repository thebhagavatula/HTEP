import cv2
import numpy as np


def segment_lines(paragraph_img, min_line_height=10):
    """
    Segment paragraph image into individual text lines.
    Returns list of line images (top → bottom)
    """

    gray = cv2.cvtColor(paragraph_img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    horizontal_projection = np.sum(thresh, axis=1)
    threshold = np.max(horizontal_projection) * 0.1

    lines = []
    in_line = False
    start = 0

    for i, val in enumerate(horizontal_projection):
        if val > threshold and not in_line:
            start = i
            in_line = True
        elif val <= threshold and in_line:
            end = i
            if end - start >= min_line_height:
                lines.append(paragraph_img[start:end, :])
            in_line = False

    if in_line and paragraph_img.shape[0] - start >= min_line_height:
        lines.append(paragraph_img[start:, :])

    return lines
