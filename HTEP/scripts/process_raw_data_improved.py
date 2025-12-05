"""
Improved raw data processing with optional preprocessing steps for OCR

Usage:
    python scripts/process_raw_data_improved.py [--dpi DPI] [--upscale FACTOR] [--deskew] [--no-save-images]

This script processes files in data/raw and writes results to data/processed.
It also saves preprocessed debug images to data/processed/debug_images/ for inspection.
"""
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

# Ensure we can import src package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ocr.extractor import OCRExtractor

# Simple deskew
def deskew_image_cv(img_cv_gray):
    # Compute moments and angle
    coords = np.column_stack(np.where(img_cv_gray > 0))
    if coords.size == 0:
        return img_cv_gray, 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = angle + 90
    (h, w) = img_cv_gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_cv_gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def improve_image(pil_img: Image.Image, upscale: int = 2, denoise: bool = True, deskew: bool = True):
    # Convert to CV format
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # Upscale
    if upscale and upscale > 1:
        new_w = int(gray.shape[1] * upscale)
        new_h = int(gray.shape[0] * upscale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Denoise using bilateral filter which preserves edges
    if denoise:
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive thresholding for potential uneven illumination
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 25, 15)

    # Optional deskew
    angle = 0.0
    if deskew:
        try:
            gray, angle = deskew_image_cv(gray)
        except Exception:
            pass

    # Additional morphology if needed (opening)
    kernel = np.ones((1, 1), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    return Image.fromarray(gray), angle


def process_file_improved(extractor, file_path: Path, args):
    result = {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "processed_at": datetime.utcnow().isoformat() + "Z",
        "status": "success",
        "pages": {},
        "error": None,
    }

    try:
        if file_path.suffix.lower() == ".pdf":
            pages = convert_from_path(str(file_path), dpi=args.dpi)
            for i, page in enumerate(pages, start=1):
                # Preprocess
                processed_img, angle = improve_image(page, upscale=args.upscale, denoise=args.denoise, deskew=args.deskew)

                # Save debug image
                if args.save_images:
                    debug_dir = ROOT / "data" / "processed" / "debug_images" / file_path.stem
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    processed_img.save(debug_dir / f"page_{i}.png")

                text = extractor._extract_from_image(processed_img)  # use internal method to avoid re-preprocessing
                result["pages"][f"page_{i}"] = text

        elif file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            img = Image.open(str(file_path))
            processed_img, angle = improve_image(img, upscale=args.upscale, denoise=args.denoise, deskew=args.deskew)
            if args.save_images:
                debug_dir = ROOT / "data" / "processed" / "debug_images" / file_path.stem
                debug_dir.mkdir(parents=True, exist_ok=True)
                processed_img.save(debug_dir / f"page_1.png")
            text = extractor._extract_from_image(processed_img)
            result["pages"]["page_1"] = text

        else:
            result["status"] = "skipped"
            result["error"] = f"Unsupported file type: {file_path.suffix}"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=400, help="DPI to use for PDF conversion (default 400)")
    parser.add_argument("--upscale", type=int, default=2, help="Upscale factor for image resizing (default 2)")
    parser.add_argument("--denoise", action="store_true", default=True, help="Apply bilateral denoising (default True)")
    parser.add_argument("--no-denoise", dest="denoise", action="store_false", help="Disable denoising")
    parser.add_argument("--deskew", action="store_true", default=True, help="Apply deskew (default True)")
    parser.add_argument("--no-deskew", dest="deskew", action="store_false", help="Disable deskew")
    parser.add_argument("--save-images", action="store_true", default=True, help="Save debug preprocessed images (default True)")
    parser.add_argument("--no-save-images", dest="save_images", action="store_false", help="Do not save debug images")
    args = parser.parse_args()

    print("Running improved OCR processing with args:", args)

    extractor = OCRExtractor()

    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = list(raw_dir.glob("**/*.*"))
    if not files:
        print("No files found in data/raw/")
        return

    results = {}
    for file in files:
        print(f"Processing (improved): {file.name}")
        rslt = process_file_improved(extractor, file, args)
        results[file.name] = rslt
        with open(processed_dir / f"improved_{file.stem}.json", "w", encoding="utf-8") as f:
            json.dump(rslt, f, indent=2, ensure_ascii=False)

    with open(processed_dir / "processing_results_improved.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Done. Improved results written to data/processed/processing_results_improved.json")


if __name__ == "__main__":
    main()
