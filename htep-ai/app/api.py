# app/api.py
#
# One Flask app for both places it runs:
#   - Vercel: deployed as a single Python Function (see pyproject.toml and
#     vercel.json); public/ is served by Vercel's CDN.
#   - Local development: python -c "from app.api import app; app.run()"
#     also serves public/ itself.

from pathlib import Path
import sys
import tempfile
import threading
import time
import traceback
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

# -------------------------------
# PATHS
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import MAX_UPLOAD_MB, WEB_DIR

# The only writable place on serverless hosts. Uploads live here just long
# enough to be processed, then are deleted.
UPLOAD_DIR = Path(tempfile.gettempdir()) / "htep-uploads"

ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}

from src.ocr.extractor import DocumentTooLargeError
from src.pipeline.digitizer import Digitizer

# -------------------------------
# FLASK APP
# -------------------------------

app = Flask(
    __name__,
    static_folder=str(WEB_DIR),
    static_url_path=""
)
app.config["MAX_CONTENT_LENGTH"] = int(MAX_UPLOAD_MB * 1024 * 1024)
CORS(app)

# -------------------------------
# LAZY LOAD MODELS
# -------------------------------
# Loading the OCR models at import time would slow every cold start, even for
# requests that don't need them. The page's status call loads them early.

digitizer = None
_digitizer_lock = threading.Lock()

def load_engines():
    global digitizer
    if digitizer is not None:
        return

    # Concurrent requests (e.g. the frontend's status warm-up and an upload)
    # must not load the models twice: that doubles memory.
    with _digitizer_lock:
        if digitizer is None:
            print("Initializing engines...", flush=True)
            t0 = time.time()
            digitizer = Digitizer()
            print(f"Engines initialized in {round(time.time() - t0, 2)}s", flush=True)

# -------------------------------
# API ROUTES
# -------------------------------

@app.route("/api/status")
def status():
    """Health check; also loads the engines so the first upload is quick."""
    load_engines()
    import psutil
    mem = psutil.Process().memory_info()
    vlm = digitizer.vlm
    return jsonify({
        "status": "ok",
        "ocr_engine": digitizer.ocr.engine_name,
        "vlm": {
            "enabled": vlm.enabled,
            "model": vlm.model if vlm.enabled else None,
            "reachable": vlm.is_reachable(),
        },
        "limits": {"max_upload_mb": MAX_UPLOAD_MB},
        "memory_mb": round(mem.rss / 1024 / 1024, 1),
    })


def _save_upload():
    """Save the posted file. Returns (file_path, filename) or (None, error_response)."""
    if "file" not in request.files:
        return None, (jsonify({"error": "No file uploaded"}), 400)

    file = request.files["file"]
    if file.filename == "":
        return None, (jsonify({"error": "Empty filename"}), 400)

    filename = secure_filename(file.filename)
    if Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
        return None, (jsonify({"error": "Upload a PDF, PNG or JPEG file"}), 400)

    # A unique name per upload, so two uploads of "scan.jpg" never overwrite each other
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{filename}"
    file.save(file_path)
    return file_path, filename


# The engines are not thread-safe, and two documents at once can exhaust
# memory, so files are processed one at a time.
_process_lock = threading.Lock()

def _process_file(file_path: Path, filename: str) -> dict:
    """Digitize a saved upload, then delete it: medical documents are not kept."""
    load_engines()
    try:
        with _process_lock:
            result = digitizer.process(file_path)
    finally:
        file_path.unlink(missing_ok=True)

    result["file"] = filename
    # No document text in the logs: it is patient data
    print(f"Processed a {Path(filename).suffix} file in {result['timings']['total']}s "
          f"(ocr_confidence={result['ocr_confidence']}, vlm_used={result['vlm']['used']}, "
          f"fields_to_check={len(result['review_required'])})", flush=True)
    return result


@app.route("/api/digitize", methods=["POST"])
def digitize():
    """Digitize one prescription. Synchronous: the response is the record."""
    try:
        file_path, filename = _save_upload()
        if file_path is None:
            return filename
        return jsonify(_process_file(file_path, filename))

    except DocumentTooLargeError as e:
        return jsonify({"error": str(e)}), 413

    except RequestEntityTooLarge:
        raise  # answered by upload_too_large()

    except Exception as e:
        print("ERROR DURING PROCESSING", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        return jsonify({"error": str(e)}), 500


@app.errorhandler(RequestEntityTooLarge)
def upload_too_large(_e):
    return jsonify({"error": f"The file is larger than {MAX_UPLOAD_MB:g} MB. "
                             "Upload a smaller photo or scan."}), 413

# -------------------------------
# WEBSITE ROUTES (local development; Vercel serves public/ from its CDN)
# -------------------------------

@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(WEB_DIR, path)

# -------------------------------
# RUN
# -------------------------------

if __name__ == "__main__":
    app.run(debug=True)
