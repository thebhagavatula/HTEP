# app/api.py

from pathlib import Path
import sys
import time
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import traceback
import cv2

# -------------------------------
# PATHS
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import WEB_DIR, RAW_DATA_DIR

UPLOAD_DIR = RAW_DATA_DIR
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# OCR ENGINE (configured in src/config.py)
# -------------------------------

# -------------------------------
# IMPORT ENGINES
# -------------------------------

from src.ocr.extractor import OCRExtractor
from src.recognition.icr_block_engine import BlockICREngine
from src.nlp.block_parser import BlockTextParser
from src.recognition.icr_llava_engine import LlavaICREngine
from src.nlp.ocr_postprocessor import OCRPostProcessor
from src.nlp.medical_extractor import MedicalDocExtractor

# -------------------------------
# FLASK APP
# -------------------------------

app = Flask(
    __name__,
    static_folder=str(WEB_DIR),
    static_url_path=""
)

# -------------------------------
# LOAD MODELS ONCE
# -------------------------------

ocr_engine = OCRExtractor()
block_icr = BlockICREngine()
block_parser = BlockTextParser()

llava_icr = None

try:
    llava_icr = LlavaICREngine()       # LLaVa Integration
except Exception as e:
    print(f"LlavaICREngine disabled: {e}")

# OCR Post-Processor (drug/disease dictionaries + fuzzy matching)
ocr_postprocessor = None
try:
    ocr_postprocessor = OCRPostProcessor()
    print("[OK] OCRPostProcessor loaded")
except Exception as e:
    print(f"OCRPostProcessor disabled: {e}")

medical_extractor = None
if ocr_postprocessor:
    try:
        medical_extractor = MedicalDocExtractor(ocr_postprocessor.drugs, ocr_postprocessor.diseases)
        print("[OK] MedicalDocExtractor loaded")
    except Exception as e:
        print(f"MedicalDocExtractor disabled: {e}")

print("Backend ready")

# -------------------------------
# WEBSITE ROUTES
# -------------------------------

@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(WEB_DIR, path)

# -------------------------------
# API ROUTE
# -------------------------------

def _preview(text: str, limit: int = 180) -> str:
    if not text:
        return ""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        filename = secure_filename(file.filename)
        file_path = UPLOAD_DIR / filename
        file.save(file_path)

        print(f"File received: {filename}")
        request_start = time.time()

        suffix = file_path.suffix.lower()

        # ---------------- OCR ----------------
        t0 = time.time()
        if suffix == ".pdf":
            pages = ocr_engine.extract_from_pdf(str(file_path))
            ocr_text = "\n".join(pages.values())
        else:
            ocr_text = ocr_engine.extract_from_image(str(file_path))
        print(f"OCR completed in {time.time() - t0:.2f}s")

        # ---------------- BLOCK ICR (IMAGES ONLY) ----------------
        block_text = ""
        block_text_raw = ""
        block_parse_result = None
        llava_text = ""

        if suffix in [".png", ".jpg", ".jpeg"]:
            image = cv2.imread(str(file_path))
            if image is not None:

                # -------- BLOCK ICR --------
                t0 = time.time()
                block_text = block_icr.predict_paragraph(image)

                # Fallback to sentence if single-line
                if "\n" not in block_text:
                    block_text = block_icr.predict_sentence(image)
                print(f"Block ICR completed in {time.time() - t0:.2f}s")

                block_text_raw = block_text

                # -------- BLOCK PARSER (SPACY + SCISPACY DICTIONARY) --------
                if block_text.strip():
                    try:
                        t0 = time.time()
                        block_parse_result = block_parser.parse(block_text)
                        block_text = block_parse_result.get("corrected_text", block_text)
                        print(f"Block parser completed in {time.time() - t0:.2f}s")
                    except Exception as e:
                        print("Block parser failed:", e)

                # -------- LLAVA ICR (EXPERIMENTAL) --------
                if llava_icr is not None:
                    try:
                        t0 = time.time()
                        llava_result = llava_icr.predict_paragraph(image)
                        llava_text = llava_result.get("text", "")
                        llava_conf = llava_result.get("confidence", 0.0)
                        print(f"LLaVA ICR completed in {time.time() - t0:.2f}s")
                    except Exception as e:
                        print("LLaVa ICR failed:", e)
                        llava_text = ""

        # ---------------- MERGE OUTPUT ----------------
        final_text = ocr_text.strip()

        # ---------------- OCR POST-PROCESSING ----------------
        post_result = None
        matched_drugs = []
        matched_diseases = []
        post_corrections = []

        if ocr_postprocessor is not None and final_text:
            try:
                t0 = time.time()
                post_result = ocr_postprocessor.process(final_text)
                final_text = post_result.get("corrected_text", final_text)
                matched_drugs = post_result.get("matched_drugs", [])
                matched_diseases = post_result.get("matched_diseases", [])
                post_corrections = post_result.get("corrections", [])
                print(f"Post-processing completed in {time.time() - t0:.2f}s")
                print(f"  Drugs matched: {matched_drugs}")
                print(f"  Diseases matched: {matched_diseases}")
                print(f"  Corrections applied: {len(post_corrections)}")
            except Exception as e:
                print(f"Post-processing failed (falling back to raw OCR): {e}")
                final_text = ocr_text.strip()

        # ---------------- JSON EXTRACTION ----------------
        extracted_data = {}
        if medical_extractor is not None and final_text:
            try:
                t0 = time.time()
                extracted_data = medical_extractor.extract(final_text)
                print(f"JSON extraction completed in {time.time() - t0:.2f}s")
            except Exception as e:
                print(f"JSON extraction failed: {e}")

        # ---------------- DEBUG LOGS ----------------
        print(f"OCR len={len(ocr_text)} preview={_preview(ocr_text)!r}")
        print(f"BLOCK len={len(block_text)} preview={_preview(block_text)!r}")
        print(f"LLAVA len={len(llava_text)} preview={_preview(llava_text)!r}")
        print(f"FINAL len={len(final_text)} preview={_preview(final_text)!r}")
        print(f"Total request time: {time.time() - request_start:.2f}s")

        response = {
            "text": final_text,
            "file": filename,
            "ocr_text": ocr_text,
            "corrected_text": final_text,
            "matched_drugs": matched_drugs,
            "matched_diseases": matched_diseases,
            "corrections": post_corrections,
            "extracted_data": extracted_data,
        }

        if block_text_raw.strip() and block_parse_result is not None:
            response["block_text_raw"] = block_text_raw
            response["block_text_parsed"] = block_text
            response["block_parser"] = {
                "backend": block_parse_result.get("backend", "unknown"),
                "dictionary_matches": block_parse_result.get("dictionary_matches", []),
                "dictionary_layers": block_parse_result.get(
                    "dictionary_layers", {"medical": [], "english": []}
                ),
                "entities": block_parse_result.get("entities", []),
                "corrections": block_parse_result.get("corrections", []),
            }

        return jsonify(response)

    except Exception as e:
        print("ERROR DURING PROCESSING")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -------------------------------
# RUN
# -------------------------------

if __name__ == "__main__":
    app.run(debug=True)
