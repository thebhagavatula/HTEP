# HTEP: Prescription Digitizer

HTEP turns a photo or scan of a prescription into a structured digital record:
patient and doctor details, diagnosis, and a medicine list with dose, frequency
and duration. The record shows exactly what was extracted and cannot be edited
on the page; fields that need a closer look are highlighted. It can be
exported as CSV or JSON, or printed.

Live site: <https://htep-ai.vercel.app>

## How it works

```
upload (PDF / PNG / JPEG)
  -> OCR in the user's browser: PaddleOCR's PP-OCRv6 medium models on WebGPU
     (fallback without WebGPU: the server's lighter OCR)
  -> dictionary correction of drug and disease names
  -> rule-based field extraction
  -> [only if OCR looks unreliable] vision-language model reads the image
  -> grounding check: is each VLM value present in the OCR text?
  -> review page: unverified values are marked "Check"
```

- **OCR** reads the page. It runs in the browser (`public/ocr.js`) with the
  same PP-OCRv6 medium models PaddleOCR uses, through ONNX Runtime Web on
  WebGPU: about 2 s per image on an integrated laptop GPU, where one server CPU
  needs about 80 s. The models (~140 MB) download once per browser. Only the
  recognized text is sent to the server (`POST /api/extract`), so the document
  never leaves the device. `ocr.js` is a step-by-step port of RapidOCR's Python
  pipeline; on test images it gives identical text for print and differs only
  on borderline handwritten characters.
- Browsers without WebGPU upload the file to `POST /api/digitize`, where the
  server reads it with smaller models (PP-OCRv6 small + English PP-OCRv5),
  which are fast enough for the free hosting plan but weaker on handwriting.
  The review page says which reader was used.
- Values the rules extract from OCR text are taken from the page itself, so
  they are trusted.
- **Vision-language model (VLM)** is a fallback reader for handwriting and messy
  layouts (default model: Qwen2.5-VL 7B). It runs only when the OCR confidence is
  low, there is little OCR text, or no medicines were found. It returns JSON
  that follows a fixed schema.
- **Grounding** (`src/pipeline/grounding.py`): a VLM value counts as verified
  only if it also appears in the OCR text, allowing small OCR errors. VLMs
  can invent plausible medicines; those are kept but flagged for a person to
  check, never silently trusted.
- Uploaded files are deleted as soon as they are processed.

HTEP helps with data entry. It does not give medical advice, and a person
should always check the record against the original.

## Repository layout

```
htep-ai/
  app/api.py              Flask API: /api/extract (browser OCR text), /api/digitize
                          (server OCR), /api/merge (VLM answer), /api/status
  src/pipeline/           digitizer.py (orchestration), grounding.py
  src/ocr/extractor.py    OCR engines: rapid (default), paddle, tesseract
  src/nlp/                field extraction, dictionary correction, classifier
  src/vlm/client.py       VLM client (Ollama or OpenAI-compatible API)
  public/                 static frontend (served by Vercel's CDN)
  public/ocr.js           in-browser OCR (ONNX Runtime Web + OpenCV.js + pdf.js)
  public/models/ocr/      PP-OCRv6 medium models for the browser (~140 MB)
  models/ocr/             the server's small OCR models (~18 MB)
  pyproject.toml          runtime dependencies + Vercel entrypoint
  vercel.json             function limits (300 s) and bundle exclusions
  tests/test_digitizer.py pipeline, grounding and API tests
  src/icr, src/recognition, src/segmentation, models/icr_block/,
  requirements-research.txt
                          research: handwriting ICR experiments, not used by the server
```

## Run locally

Python 3.12. PDFs are rendered by pypdfium2, so no system packages are needed.

```bash
cd htep-ai
python -m venv .venv
.venv/Scripts/activate          # Linux/macOS: source .venv/bin/activate
pip install flask flask-cors psutil python-dotenv rapidocr onnxruntime opencv-python-headless numpy pypdfium2 pillow rapidfuzz -c constraints-deploy.txt
python -c "from app.api import app; app.run(port=5000)"
```

(The package list is `[project].dependencies` in `pyproject.toml`.)

Open <http://127.0.0.1:5000>. To digitize one file from the command line:
`python main.py path/to/prescription.jpg`.

Tests: `pip install pytest && python -m pytest tests/test_digitizer.py`.

## Vision model

### On-device (built; switched off until verified)

`ON_DEVICE_AI_ENABLED` in `public/output.html` is `false` until Qwen2-VL's
handwriting quality and speed have been checked on real devices.

When the OCR looks unreliable (little text, low confidence, or no medicines
found), the API returns `vlm.recommended: true` and a small preview of the
first page (`page_image`). The review page then offers **Read with on-device
AI**:

1. `public/vlm.js` loads Qwen2-VL 2B (`onnx-community/Qwen2-VL-2B-Instruct`,
   about 2.5 GB, cached by the browser after the first run) with
   Transformers.js on WebGPU and reads the preview in the browser. The image
   never leaves the user's device.
2. The browser sends only the model's JSON answer, the current form and the
   OCR text to `POST /api/merge`, which applies the same grounding rules as
   the server: AI values only fill empty fields, and any value not found in
   the OCR text is marked **Check**.

It needs a browser with WebGPU (a recent Chrome or Edge on a computer); other
browsers keep the OCR-only result.

### Server-side (optional, needs a GPU server)

A server-side VLM can be used instead. It runs as a separate service, so the
web server stays small. Set:

| Variable | Default | Meaning |
|---|---|---|
| `VLM_URL` | empty (off) | e.g. `http://gpu-box:11434` (Ollama) or `http://gpu-box:8000` (vLLM) |
| `VLM_API` | `ollama` | `ollama`, or `openai` for vLLM / other OpenAI-compatible servers |
| `VLM_MODEL` | `qwen2.5vl:7b` | model name on that server (vLLM: `Qwen/Qwen2.5-VL-7B-Instruct`) |
| `VLM_MODE` | `fallback` | `fallback`, `always` or `off` |
| `VLM_MIN_OCR_CONFIDENCE` | `0.85` | below this OCR confidence, ask the VLM |
| `VLM_TIMEOUT_S` | `90` | per-request timeout |

Hardware: Ollama 0.17 reserves about 6.7 GiB for Qwen2.5-VL's vision
encoder on top of the weights, so plan on a GPU with 16 GB or more (for
example an L4 or A10G), or serve it with vLLM. It does not fit an 8 GB laptop
GPU through Ollama.

```bash
ollama pull qwen2.5vl:7b
# or
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --max-model-len 8192
```

## Deploy

The site and the API run in one Vercel project on the free Hobby plan, which
has no payment method, so hosting costs $0 and nothing runs on a personal
computer.

- Vercel project settings: **Root Directory = `htep-ai`**. Vercel reads
  `pyproject.toml` (dependencies, and `app.api:app` as the entrypoint) and
  `vercel.json` (300 s limit, files kept out of the bundle).
- `public/` is served from Vercel's CDN; `/api/*` runs in one Python Function
  (2 GB, 1 vCPU). Pushing to `main` deploys both.
- Limits that keep a request on the free plan: 4 MB uploads (Vercel accepts
  4.5 MB), 10 PDF pages. Hobby allows 4 active-CPU hours a month, roughly
  3,600 to 7,200 single-page documents.
- Measured: bundle about 375 MB of the 500 MB allowed, cold start about 2 s,
  peak memory about 650 MB.
- Hobby is for personal, non-commercial use. A commercial launch needs either
  a paid plan or phase 2 (OCR in the browser).
- `htep-ai/Dockerfile` still builds a container for any other host.

## Measured on `tests/ground_truth.json` (37 documents)

| | Before (PaddleOCR + TensorFlow + LLaVA) | Now (RapidOCR) |
|---|---|---|
| Character error rate | 1.45% | 0.63% |
| Time per image (CPU) | ~18 s | ~1.5–2 s |
| Engine start-up | ~110 s | ~2 s |
| Peak memory | 2.5 to 4.7 GB | about 650 MB |

## Roadmap

1. **OCR in the browser:** run the same ONNX models with ONNX Runtime Web, so
   there is no backend at all.
2. **Lab reports:** test, value, unit and reference range, with out-of-range
   flags. The classifier already recognizes lab reports; they need their own
   extractor in `src/pipeline/digitizer.py`.
3. **Personal health records:** a patient's prescriptions and reports over
   time, with a medicine schedule and lab trends. This needs accounts and
   storage.
