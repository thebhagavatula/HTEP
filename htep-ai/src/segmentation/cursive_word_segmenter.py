import cv2
import numpy as np


def segment_cursive_words(
    line_img,
    min_gap_ratio=0.25,
    min_word_width=15
):
    """
    Segment a cursive line image into word images
    using vertical projection gap analysis.

    Args:
        line_img: BGR or grayscale image of ONE text line
        min_gap_ratio: fraction of avg stroke width to treat as word gap
        min_word_width: minimum width of a word crop

    Returns:
        List of word images (left → right)
    """

    if line_img is None:
        return []

    # Convert to grayscale
    if line_img.ndim == 3:
        gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = line_img.copy()

    # Binarize (invert text to white)
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Vertical projection
    projection = np.sum(thresh, axis=0)

    # Normalize projection
    projection = projection / np.max(projection)

    # Columns considered "empty"
    empty_cols = projection < 0.05

    words = []
    in_word = False
    start = 0

    # Estimate average stroke width
    non_zero_cols = np.where(projection > 0.2)[0]
    if len(non_zero_cols) < 2:
        return [line_img]

    avg_stroke = np.mean(np.diff(non_zero_cols))
    min_gap = max(int(avg_stroke / min_gap_ratio), 8)

    gap_count = 0

    for i, is_empty in enumerate(empty_cols):
        if not is_empty and not in_word:
            start = i
            in_word = True
            gap_count = 0

        elif is_empty and in_word:
            gap_count += 1
            if gap_count >= min_gap:
                end = i - gap_count
                if end - start >= min_word_width:
                    words.append(line_img[:, start:end])
                in_word = False

        elif not is_empty and in_word:
            gap_count = 0

    # Handle last word
    if in_word:
        end = line_img.shape[1]
        if end - start >= min_word_width:
            words.append(line_img[:, start:end])

    return words
