# app/api.py

from pathlib import Path
import sys
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import traceback
import cv2
import pytesseract

# -------------------------------
# PATHS
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import WEB_DIR, RAW_DATA_DIR, TESSERACT_CMD_PATH

UPLOAD_DIR = RAW_DATA_DIR
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# TESSERACT PATH
# -------------------------------

if TESSERACT_CMD_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH

# -------------------------------
# IMPORT ENGINES
# -------------------------------

from src.ocr.extractor import OCRExtractor
from src.recognition.icr_block_engine import BlockICREngine
from src.nlp.block_parser import BlockTextParser
from src.recognition.icr_llava_engine import LlavaICREngine

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
    llava_icr = LlavaICREngine()       # ✅ LLaVa Integration
except Exception as e:
    print(f"⚠️ LlavaICREngine disabled: {e}")

print("✅ Backend ready")

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

        print(f"📥 File received: {filename}")

        suffix = file_path.suffix.lower()

        # ---------------- OCR ----------------
        if suffix == ".pdf":
            pages = ocr_engine.extract_from_pdf(str(file_path))
            ocr_text = "\n".join(pages.values())
        else:
            ocr_text = ocr_engine.extract_from_image(str(file_path))

        # ---------------- BLOCK ICR (IMAGES ONLY) ----------------
        block_text = ""
        block_text_raw = ""
        block_parse_result = None
        llava_text = ""

        if suffix in [".png", ".jpg", ".jpeg"]:
            image = cv2.imread(str(file_path))
            if image is not None:

                # -------- BLOCK ICR --------
                block_text = block_icr.predict_paragraph(image)

                # Fallback to sentence if single-line
                if "\n" not in block_text:
                    block_text = block_icr.predict_sentence(image)

                block_text_raw = block_text

                # -------- BLOCK PARSER (SPACY + SCISPACY DICTIONARY) --------
                if block_text.strip():
                    try:
                        block_parse_result = block_parser.parse(block_text)
                        block_text = block_parse_result.get("corrected_text", block_text)
                    except Exception as e:
                        print("⚠️ Block parser failed:", e)

                # -------- LLAVA ICR (EXPERIMENTAL) --------
                if llava_icr is not None:
                    try:
                        llava_result = llava_icr.predict_paragraph(image)
                        llava_text = llava_result.get("text", "")
                        llava_conf = llava_result.get("confidence", 0.0)
                    except Exception as e:
                        print("⚠️ LLaVa ICR failed:", e)
                        llava_text = ""

        # ---------------- MERGE OUTPUT ----------------
        final_text = ocr_text.strip()

        if block_text.strip():
            final_text += "\n\n[Block Handwritten]\n" + block_text.strip()

        # ⚠️ LLaVa is shown but NOT trusted
        if llava_text and llava_text.strip():
            final_text += (
                "\n\n[LLaVa VLM - Experimental]\n"
                + llava_text.strip()
            )

        # ---------------- DEBUG LOGS ----------------
        print(f"🧾 OCR len={len(ocr_text)} preview={_preview(ocr_text)!r}")
        print(f"🧾 BLOCK len={len(block_text)} preview={_preview(block_text)!r}")
        print(f"🧾 LLAVA len={len(llava_text)} preview={_preview(llava_text)!r}")
        print(f"🧾 FINAL len={len(final_text)} preview={_preview(final_text)!r}")

        response = {
            "text": final_text,
            "file": filename,
            "llava_text": llava_text,
            "ocr_text": ocr_text,
            "block_text": block_text
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
