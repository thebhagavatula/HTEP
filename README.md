# HTEP: Prescription Digitizer

HTEP turns a photo or scan of a prescription into an editable, checked digital
record: patient and doctor details, diagnosis, and a medicine list with dose,
frequency and duration. Staff check the highlighted fields and export the
record as CSV or JSON, or print it.

Live site: <https://htep-ai.vercel.app>

## How it works

```
upload (PDF / PNG / JPEG)
  -> OCR (RapidOCR: PP-OCR models on ONNX Runtime)
  -> dictionary correction of drug and disease names
  -> rule-based field extraction
  -> [only if OCR looks unreliable] vision-language model reads the image
  -> grounding check: is each VLM value present in the OCR text?
  -> review page: unverified values are marked "Check"
```

- **OCR** reads the page. Values the rules extract from OCR text are taken
  from the page itself, so they are trusted.
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
  app/api.py              Flask API: /jobs (async), /upload (sync), /status
  src/pipeline/           digitizer.py (orchestration), grounding.py
  src/ocr/extractor.py    OCR engines: rapid (default), paddle, tesseract
  src/nlp/                field extraction, dictionary correction, classifier
  src/vlm/client.py       VLM client (Ollama or OpenAI-compatible API)
  web/                    static frontend (deployed to Vercel)
  tests/test_digitizer.py pipeline, grounding and API tests
  src/icr, src/recognition, src/segmentation, models/
                          research: handwriting ICR experiments, not used by the server
```

## Run locally

Python 3.12. PDF support needs Poppler (`poppler-utils` on Linux, or the
Windows build on `PATH`).

```bash
cd htep-ai
python -m venv .venv
.venv/Scripts/activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements-deploy.txt -c constraints-deploy.txt
python -c "from app.api import app; app.run(port=5000)"
```

Open <http://127.0.0.1:5000>. To digitize one file from the command line:
`python main.py path/to/prescription.jpg`.

Tests: `pip install pytest && python -m pytest tests/test_digitizer.py`.

## Vision model (optional)

The VLM runs as a separate service, so the web server stays small. Set:

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

- **Backend:** `htep-ai/Dockerfile` (Python 3.12, about 300 MB of Python
  packages). One instance with **1 GiB** of memory is enough: it peaked at
  about 650 MB on a 2-page PDF. It runs on Cloud Run, Render (a paid instance;
  the free 512 MB plan is too small) or any Docker host.
- **Frontend:** `htep-ai/web` on Vercel. Point `HTEP_BACKEND_URL` in
  `web/app.js` at the backend.

## Measured on `tests/ground_truth.json` (37 documents)

| | Before (PaddleOCR + TensorFlow + LLaVA) | Now (RapidOCR) |
|---|---|---|
| Character error rate | 1.45% | 0.92% |
| Time per image (CPU) | ~18 s | 1.6 s |
| Engine start-up | ~110 s | under 1 s |
| Peak memory | 2.5 to 4.7 GB | about 650 MB |

## Roadmap

1. **Lab reports:** test, value, unit and reference range, with out-of-range
   flags. The classifier already recognizes lab reports; they need their own
   extractor in `src/pipeline/digitizer.py`.
2. **Personal health records:** a patient's prescriptions and reports over
   time, with a medicine schedule and lab trends. This needs accounts and
   storage.
