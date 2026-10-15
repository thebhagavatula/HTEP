# app/api.py

from pathlib import Path
import sys
import threading
import time
import traceback
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# -------------------------------
# PATHS
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import WEB_DIR, RAW_DATA_DIR

UPLOAD_DIR = RAW_DATA_DIR
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}

from src.pipeline.digitizer import Digitizer

# -------------------------------
# FLASK APP
# -------------------------------

app = Flask(
    __name__,
    static_folder=str(WEB_DIR),
    static_url_path=""
)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
CORS(app)

# -------------------------------
# LAZY LOAD MODELS
# -------------------------------
# Loading the OCR models at import time blocks gunicorn from binding to the
# port on Cloud Run, causing startup timeouts. We lazy-load them instead.

digitizer = None
_digitizer_lock = threading.Lock()

def load_engines():
    global digitizer
    if digitizer is not None:
        return

    # Concurrent requests (e.g. the frontend's /status warm-up and an upload)
    # must not load the models twice: that doubles memory.
    with _digitizer_lock:
        if digitizer is None:
            print("Initializing engines...", flush=True)
            t0 = time.time()
            digitizer = Digitizer()
            print(f"Engines initialized in {round(time.time() - t0, 2)}s", flush=True)

print("Backend ready", flush=True)

# -------------------------------
# WEBSITE ROUTES
# -------------------------------

@app.route("/status")
def status():
    """Health-check endpoint — shows what's loaded."""
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
            print(f"File received: {filename}", flush=True)
            result = digitizer.process(file_path)
    finally:
        file_path.unlink(missing_ok=True)

    result["file"] = filename
    print(f"Processed {filename} in {result['timings']['total']}s "
          f"(ocr_confidence={result['ocr_confidence']}, vlm={result['vlm']}, "
          f"review={result['review_required']})", flush=True)
    return result


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        file_path, filename = _save_upload()
        if file_path is None:
            return filename
        return jsonify(_process_file(file_path, filename))

    except Exception as e:
        print("ERROR DURING PROCESSING", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        return jsonify({"error": str(e)}), 500

# -------------------------------
# ASYNC JOBS
# -------------------------------
# A multi-page PDF can take minutes, longer than proxies such as Cloudflare
# tunnels (100s) keep a request open. POST /jobs returns at once and the
# client polls GET /jobs/<id> for the result.

JOB_TTL_S = 3600
_jobs = {}
_jobs_lock = threading.Lock()

def _run_job(job_id: str, file_path: Path, filename: str):
    try:
        update = {"status": "done", "result": _process_file(file_path, filename)}
    except Exception as e:
        print("ERROR DURING PROCESSING", flush=True)
        traceback.print_exc()
        update = {"status": "error", "error": str(e)}
    with _jobs_lock:
        _jobs[job_id].update(update)

@app.route("/jobs", methods=["POST"])
def create_job():
    file_path, filename = _save_upload()
    if file_path is None:
        return filename

    job_id = uuid.uuid4().hex
    now = time.time()
    with _jobs_lock:
        for old in [k for k, j in _jobs.items()
                    if j["status"] != "pending" and now - j["created"] > JOB_TTL_S]:
            del _jobs[old]
        _jobs[job_id] = {"status": "pending", "created": now}

    threading.Thread(target=_run_job, args=(job_id, file_path, filename), daemon=True).start()
    return jsonify({"job_id": job_id}), 202

@app.route("/jobs/<job_id>")
def get_job(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return jsonify({"error": "Unknown job — the server may have restarted"}), 404
        body = {k: v for k, v in job.items() if k != "created"}
    return jsonify(body)

# -------------------------------
# RUN
# -------------------------------

if __name__ == "__main__":
    app.run(debug=True)
