# scripts/test_scanned_icr_preproc.py

from pathlib import Path
import json
import cv2
import numpy as np
from src.icr.inference import ICRPredictor

MODEL_PATH = "models/icr_model"
SCANNED_DIR = Path("data/icr_training/scanned")
OUTPUT_JSON = Path("data/processed/icr_scanned_results.json")
TARGET_SIZE = (28, 28)

# Toggle: if True, do NOT run contour detection — just aspect-preserving resize+pad
USE_FULL_IMAGE_RESIZE = False

# Auto invert polarity (detect white background → invert)
AUTO_INVERT_POLARITY = True

# Contour/cleanup parameters
MIN_CONTOUR_AREA = 50
MARGIN = 18
MORPH_KERNEL = (3, 3)

predictor = ICRPredictor(model_path=MODEL_PATH)
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)


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


def preprocess_single_char(img, target_size=(28, 28),
                           min_contour_area=MIN_CONTOUR_AREA,
                           margin=MARGIN, morph_kernel=MORPH_KERNEL):
    """
    Robust preprocessing:
      - If USE_FULL_IMAGE_RESIZE → skip contouring.
      - Else use CLAHE, thresholding, morphology, contour union, crop, resize, pad.
    Returns a 28×28 uint8 image with black strokes on white background.
    """

    if img is None:
        raise ValueError("Input image is None")

    # Convert to grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Auto invert polarity (if background is white)
    if AUTO_INVERT_POLARITY:
        mean_intensity = float(np.mean(gray))
        if mean_intensity > 127:
            gray = 255 - gray

    # Skip contouring
    if USE_FULL_IMAGE_RESIZE:
        return resize_preserve_pad_from_gray(gray, target_size)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_cl = clahe.apply(gray)

    # Blur
    blur = cv2.GaussianBlur(gray_cl, (3, 3), 0)

    # Two thresholds: adaptive + Otsu
    thr_adapt = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 9
    )
    _, thr_otsu = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Choose threshold with more ink (more black pixels)
    nonzero_adapt = np.sum(thr_adapt == 0)
    nonzero_otsu = np.sum(thr_otsu == 0)
    thresh = thr_adapt if nonzero_adapt > nonzero_otsu else thr_otsu

    # Ensure black strokes, white background
    if np.sum(thresh == 0) < np.sum(thresh == 255):
        thresh = 255 - thresh

    # Morphology (open, then close)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours (invert because strokes = black)
    contours, _ = cv2.findContours(
        255 - cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    crop = None

    if contours:
        meaningful = [
            c for c in contours if cv2.contourArea(c) >= min_contour_area
        ]

        if meaningful:
            xs, ys, xs_w, ys_h = [], [], [], []

            for c in meaningful:
                x, y, w, h = cv2.boundingRect(c)
                xs.append(x)
                ys.append(y)
                xs_w.append(x + w)
                ys_h.append(y + h)

            x0 = max(0, min(xs) - margin)
            y0 = max(0, min(ys) - margin)
            x1 = min(cleaned.shape[1], max(xs_w) + margin)
            y1 = min(cleaned.shape[0], max(ys_h) + margin)

            crop = gray[y0:y1, x0:x1]

        else:
            # Fallback center crop
            H, W = gray.shape
            s = min(H, W)
            cy, cx = H // 2, W // 2
            half = s // 2
            crop = gray[
                max(0, cy - half):min(H, cy + half),
                max(0, cx - half):min(W, cx + half)
            ]
    else:
        # No contours → fallback
        H, W = gray.shape
        s = min(H, W)
        cy, cx = H // 2, W // 2
        half = s // 2
        crop = gray[
            max(0, cy - half):min(H, cy + half),
            max(0, cx - half):min(W, cx + half)
        ]

    # Guard against empty crop
    ch, cw = crop.shape[:2]
    if ch == 0 or cw == 0:
        return np.ones(target_size, dtype=np.uint8) * 255

    # Resize and pad
    th, tw = target_size
    scale = min(tw / cw, th / ch)

    new_w = max(1, int(cw * scale))
    new_h = max(1, int(ch * scale))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w = tw - new_w
    pad_h = th - new_h

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    final = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=255
    )

    _, final = cv2.threshold(
        final, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return final


# -------- PROCESS ALL IMAGES -------- #

results = []

for img_path in SCANNED_DIR.rglob("*"):
    if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        continue

    try:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"SKIP (cannot read): {img_path}")
            continue

        processed = preprocess_single_char(bgr, target_size=TARGET_SIZE)

        # Save processed image
        processed_save_dir = Path("data/processed/processed_icr")
        processed_save_dir.mkdir(parents=True, exist_ok=True)
        save_name = processed_save_dir / f"processed_{img_path.stem}.png"
        cv2.imwrite(str(save_name), processed)

        # Predict
        pred = predictor.predict_array(processed)

        result = {
            "file": str(img_path),
            "character": pred["character"],
            "confidence": float(pred["confidence"])
        }

        results.append(result)

        print(f"{img_path.name} -> {pred['character']} (conf {pred['confidence']:.3f})")

    except Exception as e:
        print(f"ERROR processing {img_path}: {e}")

# Save results
with OUTPUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} predictions to {OUTPUT_JSON}")
