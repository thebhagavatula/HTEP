# src/icr/preprocessing.py

import cv2
import numpy as np

# ---------------- CONFIG (same values as before) ---------------- #

TARGET_SIZE = (28, 28)
USE_FULL_IMAGE_RESIZE = False
AUTO_INVERT_POLARITY = True

MIN_CONTOUR_AREA = 50
MARGIN = 18
MORPH_KERNEL = (3, 3)

# --------------------------------------------------------------- #

def resize_preserve_pad_from_gray(gray, target_size=(28, 28), bg=255):
    """Aspect-preserving resize + pad for a grayscale image (no contouring)."""
    th, tw = target_size
    h, w = gray.shape

    if h == 0 or w == 0:
        return np.ones(target_size, dtype=np.uint8) * bg

    scale = min(tw / w, th / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w = tw - new_w
    pad_h = th - new_h

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    final = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=bg
    )

    _, final = cv2.threshold(final, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return final


def preprocess_single_char(
    img,
    target_size=(28, 28),
    min_contour_area=MIN_CONTOUR_AREA,
    margin=MARGIN,
    morph_kernel=MORPH_KERNEL
):
    """
    Preprocess a single character image into a 28x28 binary image.
    FORCED resize (no aspect ratio preservation).
    """

    if img is None:
        raise ValueError("Input image is None")

    # Convert to grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Auto invert polarity
    if AUTO_INVERT_POLARITY:
        if np.mean(gray) > 127:
            gray = 255 - gray

    # Skip contouring
    if USE_FULL_IMAGE_RESIZE:
        return resize_preserve_pad_from_gray(gray, target_size)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_cl = clahe.apply(gray)

    # Blur
    blur = cv2.GaussianBlur(gray_cl, (3, 3), 0)

    # Thresholds
    thr_adapt = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 9
    )
    _, thr_otsu = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    thresh = thr_adapt if np.sum(thr_adapt == 0) > np.sum(thr_otsu == 0) else thr_otsu

    # Ensure black strokes on white bg
    if np.sum(thresh == 0) < np.sum(thresh == 255):
        thresh = 255 - thresh

    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Contours
    contours, _ = cv2.findContours(
        255 - cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        meaningful = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
        if meaningful:
            xs, ys, xe, ye = [], [], [], []
            for c in meaningful:
                x, y, w, h = cv2.boundingRect(c)
                xs.append(x)
                ys.append(y)
                xe.append(x + w)
                ye.append(y + h)

            crop = gray[
                max(0, min(ys) - margin):min(gray.shape[0], max(ye) + margin),
                max(0, min(xs) - margin):min(gray.shape[1], max(xe) + margin)
            ]
        else:
            crop = gray
    else:
        crop = gray

    # Guard
    if crop.size == 0:
        return np.ones(target_size, dtype=np.uint8) * 255

    # FORCE resize (NO aspect ratio preservation)
    final = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)

    _, final = cv2.threshold(final, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return final
