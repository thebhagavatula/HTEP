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
MARGIN = 6                  # smaller margin -> tighter crop
MORPH_KERNEL = (1, 1)       # smaller kernel to avoid closing thin strokes


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

    final = cv2.copyMakeBorder(resized, top, bottom, left, right,
                           borderType=cv2.BORDER_CONSTANT, value=255)

    # Final binarize (ensure clean 0/255)
    _, final = cv2.threshold(final, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Center by centroid of foreground (black strokes) to match training centering ---
    # foreground mask: stroke pixels are 0 (black) after threshold, so invert for moments
    fg = (final == 0).astype('uint8')  # 1 where stroke, 0 otherwise
    mom = cv2.moments(fg)

    if mom['m00'] != 0:
        cx = int(mom['m10'] / mom['m00'])
        cy = int(mom['m01'] / mom['m00'])
        # shift to image center
        shift_x = (tw // 2) - cx
        shift_y = (th // 2) - cy
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        final = cv2.warpAffine(final, M, (tw, th), borderValue=255)
    
    # Ensure returned image is binary uint8
    _, final = cv2.threshold(final, 0, 255, cv2.THRESH_BINARY)
    final = final.astype('uint8')
    
    return final




import os

def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def debug_process_and_predict(img_path: Path):
    """
    For a single image, run multiple preprocessing variants, save intermediates,
    and print predictions. Helps find where C/D get corrupted.
    """
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print("SKIP (cannot read):", img_path)
        return

    stem = img_path.stem
    debug_root = Path("data/processed/debug") / stem
    _ensure_dir(debug_root)

    # 1) basic grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr.copy()
    cv2.imwrite(str(debug_root / "01_gray.png"), gray)

    # 2) CLAHE + blur
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_cl = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_cl, (3,3), 0)
    cv2.imwrite(str(debug_root / "02_clahe_blur.png"), blur)

    # 3) thresholds
    thr_adapt = cv2.adaptiveThreshold(blur, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 31, 9)
    _, thr_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(str(debug_root / "03_thr_adapt.png"), thr_adapt)
    cv2.imwrite(str(debug_root / "04_thr_otsu.png"), thr_otsu)

    # Save inverted versions too (to see effect)
    cv2.imwrite(str(debug_root / "05_thr_adapt_inv.png"), 255 - thr_adapt)
    cv2.imwrite(str(debug_root / "06_thr_otsu_inv.png"), 255 - thr_otsu)

    # 4) cleaned with current kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL)
    cleaned = cv2.morphologyEx(thr_otsu if np.sum(thr_otsu==0) >= np.sum(thr_adapt==0) else thr_adapt,
                               cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imwrite(str(debug_root / "07_cleaned.png"), cleaned)

    # 5) create multiple variants and predict
    variants = []

    # Variant A = current pipeline final
    final_current = preprocess_single_char(bgr, target_size=TARGET_SIZE,
                                           min_contour_area=MIN_CONTOUR_AREA,
                                           margin=MARGIN, morph_kernel=MORPH_KERNEL)
    variants.append(("current", final_current))

    # Variant B = no morphology (skip open/close)
    def preprocess_no_morph(img):
        # copy of preprocess_single_char but without morphology
        if img.ndim == 3:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            g = img.copy()
        if AUTO_INVERT_POLARITY:
            if float(np.mean(g)) > 127:
                g = 255 - g
        clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe2.apply(g)
        bl = cv2.GaussianBlur(cl, (3,3), 0)
        thr_a = cv2.adaptiveThreshold(bl, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 31, 9)
        _, thr_o = cv2.threshold(bl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = thr_a if np.sum(thr_a==0) > np.sum(thr_o==0) else thr_o
        if np.sum(th==0) < np.sum(th==255):
            th = 255 - th
        # skip morphology, proceed to contours and crop (as original)
        contours, _ = cv2.findContours(255 - th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        crop = None
        if contours:
            meaningful = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
            if meaningful:
                xs, ys, xs_w, ys_h = [], [], [], []
                for c in meaningful:
                    x, y, w, h = cv2.boundingRect(c)
                    xs.append(x); ys.append(y); xs_w.append(x+w); ys_h.append(y+h)
                x0 = max(0, min(xs) - MARGIN); y0 = max(0, min(ys) - MARGIN)
                x1 = min(th.shape[1], max(xs_w) + MARGIN); y1 = min(th.shape[0], max(ys_h) + MARGIN)
                crop = g[y0:y1, x0:x1]
        if crop is None:
            H, W = g.shape; s = min(H,W); cy, cx = H//2, W//2; half = s//2
            crop = g[max(0,cy-half):min(H,cy+half), max(0,cx-half):min(W,cx+half)]
        # resize + pad to target
        tht, twt = TARGET_SIZE
        ch, cw = crop.shape
        scale = min(twt / cw, tht / ch)
        nw, nh = max(1,int(cw*scale)), max(1,int(ch*scale))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        pad_w = twt - nw; pad_h = tht - nh
        top = pad_h//2; bottom = pad_h - top; left = pad_w//2; right = pad_w - left
        final = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
        _, final = cv2.threshold(final, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return final

    variants.append(("no_morph", preprocess_no_morph(bgr)))

    # Variant C = larger margin
    variants.append(("large_margin", preprocess_single_char(bgr, target_size=TARGET_SIZE,
                                                           min_contour_area=MIN_CONTOUR_AREA,
                                                           margin=max(4, MARGIN+12),
                                                           morph_kernel=MORPH_KERNEL)))

    # Variant D = smaller morph kernel
    variants.append(("small_morph", preprocess_single_char(bgr, target_size=TARGET_SIZE,
                                                           min_contour_area=MIN_CONTOUR_AREA,
                                                           margin=MARGIN,
                                                           morph_kernel=(1,1))))

    # Variant E = use full-image-resize (no contour crop)
    variants.append(("full_resize", resize_preserve_pad_from_gray(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), target_size=TARGET_SIZE)))

    # Save & predict each
    for name, imgv in variants:
        fname = debug_root / f"{name}.png"
        cv2.imwrite(str(fname), imgv)
        pred = predictor.predict_array(imgv)
        print(f"{img_path.name} [{name}] -> {pred['character']} (conf {pred['confidence']:.3f})  saved: {fname}")

    print("Debug images saved to:", debug_root)


# -------- PROCESS ALL IMAGES -------- #

results = []

for img_path in SCANNED_DIR.rglob("*"):
    if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        continue
    debug_process_and_predict(img_path)


# Save results
with OUTPUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} predictions to {OUTPUT_JSON}")
