"""
Process all files in data/raw/ using OCRExtractor and write results to data/processed/processing_results.json

Usage:
    python scripts/process_raw_data.py

"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Ensure src is on sys.path so we can import modules when executing as a script
ROOT = Path(__file__).resolve().parents[1]
# Put workspace root on sys.path so we can import package named 'src'
SRC_PATH = str(ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from src.ocr.extractor import OCRExtractor

# Paths
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = PROCESSED_DIR / "processing_results.json"


def process_file(extractor: OCRExtractor, file_path: Path):
    """Process a single file and return a dict of results."""
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
            extracted = extractor.extract_from_pdf(str(file_path))
            result["pages"] = extracted
        elif file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            extracted = extractor.extract_from_image(str(file_path))
            result["pages"] = {"page_1": extracted}
        else:
            result["status"] = "skipped"
            result["error"] = f"Unsupported file type: {file_path.suffix}"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def main():
    print("Processing raw document files...")

    extractor = OCRExtractor()

    files = list(RAW_DIR.glob("**/*.*"))
    if not files:
        print("No files found in data/raw/")
        return

    processed_results = {}

    for file in files:
        print(f"Processing: {file.name}")
        res = process_file(extractor, file)
        processed_results[file.name] = res
        # Save an interim per-file JSON to processed to keep results
        with open(PROCESSED_DIR / f"{file.stem}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

    # Save aggregated results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_results, f, indent=2, ensure_ascii=False)

    print(f"Done. Results written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
