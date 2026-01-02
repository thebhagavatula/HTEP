# app/api.py

from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import traceback
import cv2
import pytesseract

# -------------------------------
# PATHS
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"
UPLOAD_DIR = BASE_DIR / "data" / "raw"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# TESSERACT PATH (WINDOWS)
# -------------------------------

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# -------------------------------
# IMPORT ENGINES
# -------------------------------

from src.ocr.extractor import OCRExtractor
from src.recognition.icr_block_engine import BlockICREngine

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
        icr_text = ""

        if suffix in [".png", ".jpg", ".jpeg"]:
            image = cv2.imread(str(file_path))
            if image is not None:
                icr_text = block_icr.predict_sentence(image)

        # ---------------- MERGE ----------------
        final_text = ocr_text.strip()

        if icr_text.strip():
            final_text += "\n\n[Handwritten]\n" + icr_text.strip()

        return jsonify({
            "text": final_text,
            "file": filename
        })

    except Exception as e:
        print("❌ ERROR DURING PROCESSING")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -------------------------------
# RUN
# -------------------------------

if __name__ == "__main__":
    app.run(debug=True)
