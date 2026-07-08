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

OCR_ENGINE = "paddle"  # switch to "tesseract" anytime to roll back instantly

# Tesseract Configuration
TESSERACT_CMD_PATH = os.getenv("TESSERACT_CMD_PATH", "tesseract")

# Model configurations
LLAVA_MODEL_TAG = os.getenv("LLAVA_MODEL_TAG", "llava")

# Ensure required minimal directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
DICTIONARY_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

