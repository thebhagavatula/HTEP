# app/api.py

from pathlib import Path
import sys
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import traceback
import sys
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
from src.nlp.classifier import MedicalDocumentClassifier

# -------------------------------
# FLASK APP
# -------------------------------

app = Flask(
    __name__,
    static_folder=str(WEB_DIR),
    static_url_path=""
)
CORS(app)

# -------------------------------
# LAZY LOAD MODELS
# -------------------------------
# Loading ML models at the module level blocks gunicorn from binding to the port
# on Cloud Run, causing startup timeouts. We lazy-load them instead.

engines_loaded = False
ocr_engine = None
block_icr = None
block_parser = None
llava_icr = None
ocr_postprocessor = None
medical_extractor = None
doc_classifier = None

def load_engines():
    global engines_loaded, ocr_engine, block_icr, block_parser, llava_icr
    global ocr_postprocessor, medical_extractor, doc_classifier
    
    if engines_loaded:
        return

    print("Initializing ML engines...", flush=True)
    t0 = time.time()
    
    ocr_engine = OCRExtractor()
    block_icr = BlockICREngine()
    block_parser = BlockTextParser()

    try:
        llava_icr = LlavaICREngine()       # LLaVa Integration
    except Exception as e:
        print(f"LlavaICREngine disabled: {e}", flush=True)

    try:
        ocr_postprocessor = OCRPostProcessor()
        print("[OK] OCRPostProcessor loaded", flush=True)
    except Exception as e:
        print(f"OCRPostProcessor disabled: {e}", flush=True)

    if ocr_postprocessor:
        try:
            medical_extractor = MedicalDocExtractor(ocr_postprocessor.drugs, ocr_postprocessor.diseases)
            print("[OK] MedicalDocExtractor loaded", flush=True)
        except Exception as e:
            print(f"MedicalDocExtractor disabled: {e}", flush=True)

    doc_classifier = MedicalDocumentClassifier()
    print("[OK] MedicalDocumentClassifier loaded", flush=True)
    
    engines_loaded = True
    print(f"All engines initialized in {round(time.time() - t0, 2)}s", flush=True)

print("Backend ready", flush=True)

# -------------------------------
# WEBSITE ROUTES
# -------------------------------

@app.route("/status")
def status():
    """Health-check endpoint — shows what's loaded."""
    load_engines()
    import psutil
    proc = psutil.Process()
    mem = proc.memory_info()
    return jsonify({
        "status": "ok",
        "ocr_engine": ocr_engine.engine_name if ocr_engine else None,
        "block_icr": block_icr.model is not None if (block_icr and hasattr(block_icr, 'model')) else False,
        "ocr_postprocessor": ocr_postprocessor is not None,
        "medical_extractor": medical_extractor is not None,
        "llava_icr": llava_icr is not None,
        "memory_mb": round(mem.rss / 1024 / 1024, 1),
    })

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
        load_engines()
        
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        filename = secure_filename(file.filename)
        file_path = UPLOAD_DIR / filename
        file.save(file_path)

        print(f"File received: {filename}", flush=True)
        request_start = time.time()
        timings = {}

        suffix = file_path.suffix.lower()

        # ---------------- OCR ----------------
        t0 = time.time()
        if suffix == ".pdf":
            pages = ocr_engine.extract_from_pdf(str(file_path))
            ocr_text = "\n".join(pages.values())
        else:
            ocr_text = ocr_engine.extract_from_image(str(file_path))
        ocr_time = round(time.time() - t0, 2)
        timings["ocr"] = ocr_time
        print(f"OCR completed in {ocr_time}s", flush=True)

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
                icr_time = round(time.time() - t0, 2)
                timings["block_icr"] = icr_time
                print(f"Block ICR completed in {icr_time}s", flush=True)

                block_text_raw = block_text

                # -------- BLOCK PARSER (SPACY + SCISPACY DICTIONARY) --------
                if block_text.strip():
                    try:
                        t0 = time.time()
                        block_parse_result = block_parser.parse(block_text)
                        block_text = block_parse_result.get("corrected_text", block_text)
                        parser_time = round(time.time() - t0, 2)
                        timings["block_parser"] = parser_time
                        print(f"Block parser completed in {parser_time}s")
                    except Exception as e:
                        print("Block parser failed:", e)

                # -------- LLAVA ICR (EXPERIMENTAL) --------
                if llava_icr is not None:
                    try:
                        t0 = time.time()
                        llava_result = llava_icr.predict_paragraph(image)
                        llava_text = llava_result.get("text", "")
                        llava_conf = llava_result.get("confidence", 0.0)
                        llava_time = round(time.time() - t0, 2)
                        timings["llava_icr"] = llava_time
                        print(f"LLaVA ICR completed in {llava_time}s")
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
                postproc_time = round(time.time() - t0, 2)
                timings["post_processing"] = postproc_time
                print(f"Post-processing completed in {postproc_time}s")
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
                extraction_time = round(time.time() - t0, 2)
                timings["json_extraction"] = extraction_time
                print(f"JSON extraction completed in {extraction_time}s")
            except Exception as e:
                print(f"JSON extraction failed: {e}")

        # ---------------- DEBUG LOGS ----------------
        print(f"OCR len={len(ocr_text)} preview={_preview(ocr_text)!r}")
        print(f"BLOCK len={len(block_text)} preview={_preview(block_text)!r}")
        print(f"LLAVA len={len(llava_text)} preview={_preview(llava_text)!r}")
        print(f"FINAL len={len(final_text)} preview={_preview(final_text)!r}")
        total_time = round(time.time() - request_start, 2)
        timings["total"] = total_time
        print(f"Total request time: {total_time}s")

        # ---------------- DOCUMENT CLASSIFICATION ----------------
        classification = {}
        try:
            class_result = doc_classifier.classify_document(final_text)
            urgency_level, urgency_conf = doc_classifier.get_document_urgency(final_text)
            classification = {
                "document_type": class_result.document_type,
                "confidence": round(class_result.confidence, 2),
                "urgency": urgency_level,
                "urgency_confidence": round(urgency_conf, 2),
                "keywords_found": class_result.keywords_found[:10],
            }
        except Exception as e:
            print(f"Classification failed: {e}")

        response = {
            "text": final_text,
            "file": filename,
            "ocr_text": ocr_text,
            "corrected_text": final_text,
            "matched_drugs": matched_drugs,
            "matched_diseases": matched_diseases,
            "corrections": post_corrections,
            "extracted_data": extracted_data,
            "classification": classification,
            "timings": timings,
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
        print("ERROR DURING PROCESSING", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        return jsonify({"error": str(e)}), 500

# -------------------------------
# RUN
# -------------------------------

if __name__ == "__main__":
    app.run(debug=True)
