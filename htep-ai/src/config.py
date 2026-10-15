import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Core directories
DATA_DIR = Path(os.getenv("HTEP_DATA_DIR", BASE_DIR / "data"))
MODELS_DIR = Path(os.getenv("HTEP_MODELS_DIR", BASE_DIR / "models"))
WEB_DIR = Path(os.getenv("HTEP_WEB_DIR", BASE_DIR / "web"))

# Specific sub-directories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ICR_TRAINING_DIR = DATA_DIR / "icr_training"
DICTIONARY_DIR = DATA_DIR / "dictionaries"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

# Dictionary file paths (for OCR post-processing)
DRUG_DICT_PATH = Path(os.getenv("HTEP_DRUG_DICT", DICTIONARY_DIR / "drugs.txt"))
DISEASE_DICT_PATH = Path(os.getenv("HTEP_DISEASE_DICT", DICTIONARY_DIR / "diseases.txt"))

# Fuzzy matching thresholds (0-100, higher = stricter)
DRUG_FUZZY_THRESHOLD = int(os.getenv("HTEP_DRUG_FUZZY_THRESHOLD", "85"))
DISEASE_FUZZY_THRESHOLD = int(os.getenv("HTEP_DISEASE_FUZZY_THRESHOLD", "85"))

# "rapid" (default, ONNX Runtime), "paddle" (needs paddlepaddle) or "tesseract"
OCR_ENGINE = os.getenv("OCR_ENGINE", "rapid")

# Tesseract Configuration
TESSERACT_CMD_PATH = os.getenv("TESSERACT_CMD_PATH", "tesseract")

# Vision-language model (optional, runs as a separate service; see src/vlm/client.py)
# VLM_URL empty = disabled. VLM_API is "ollama" or "openai" (vLLM and other
# OpenAI-compatible servers).
VLM_URL = os.getenv("VLM_URL", "")
VLM_API = os.getenv("VLM_API", "ollama")
VLM_MODEL = os.getenv("VLM_MODEL", "qwen2.5vl:7b")
VLM_TIMEOUT_S = float(os.getenv("VLM_TIMEOUT_S", "90"))
VLM_MAX_IMAGE_SIDE = int(os.getenv("VLM_MAX_IMAGE_SIDE", "1280"))
# "fallback": ask the VLM only when OCR looks unreliable (low confidence, or no
# medicines found). "always": every image/first PDF page. "off": never.
VLM_MODE = os.getenv("VLM_MODE", "fallback")
VLM_MIN_OCR_CONFIDENCE = float(os.getenv("VLM_MIN_OCR_CONFIDENCE", "0.85"))

# Ensure required minimal directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
DICTIONARY_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

