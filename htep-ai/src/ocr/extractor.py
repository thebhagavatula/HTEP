# src/ocr/extractor.py

import os

# ── PaddlePaddle 3.3.x oneDNN compatibility fix ──────────────────────
# PaddlePaddle 3.3.1 has a known bug where the PIR executor fails to
# convert certain attributes for the oneDNN (MKLDNN) backend, raising:
#   NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
#   [pir::ArrayAttribute<pir::DoubleAttribute>]
# Disabling MKLDNN before any paddle import side-steps this entirely.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")

# Skip network connectivity check on PaddleX model download
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from pathlib import Path
from typing import Dict
import cv2
import numpy as np
from pdf2image import convert_from_path, pdfinfo_from_path

# PaddleX's default text detection only enforces a *minimum* side (64px) and
# lets images up to 4000px through at full size. On a 12 MP phone photo or a
# 300 DPI page the detector alone needs >3 GB and Cloud Run OOM-kills the
# container. Cap the longest side used for detection; recognition still reads
# text crops from the full-resolution image.
DET_MAX_SIDE = 1600

# Import config toggle
try:
    from src.config import OCR_ENGINE
except ImportError:
    OCR_ENGINE = "paddle"  # default if not set in config


class OCRExtractor:
    """
    Unified OCR extractor. Public interface matches what api.py expects:
      - extract_from_pdf(path) -> Dict[int, str]   {page_num: text}
      - extract_from_image(path) -> str
    Internally routes to PaddleOCR (primary) or Tesseract (backup)
    based on OCR_ENGINE config in src/config.py.
    """

    def __init__(self, lang: str = "en", device: str = "cpu"):
        self.engine_name = OCR_ENGINE

        if self.engine_name == "paddle":
            from paddleocr import PaddleOCR

            # PaddleOCR v3.x API:
            #   - `use_gpu` removed → use `device` ("cpu", "gpu:0", etc.)
            #   - disable heavy pre-processing models we don't need for
            #     straightforward scanned documents (saves ~30s startup)
            self._paddle = PaddleOCR(
                lang=lang,
                device=device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_limit_side_len=DET_MAX_SIDE,
                text_det_limit_type="max",
            )

        elif self.engine_name == "tesseract":
            import pytesseract
            from src.config import TESSERACT_CMD_PATH
            if TESSERACT_CMD_PATH:
                pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH
            self._pytesseract = pytesseract

        else:
            raise ValueError(f"Unknown OCR_ENGINE: {self.engine_name}")

        print(f"[OK] OCRExtractor initialized with engine: {self.engine_name}")

    # ──────────────── INTERNAL HELPERS ────────────────

    def _paddle_ocr_array(self, image_input) -> str:
        """
        Run PaddleOCR on an image (path string or numpy array).
        PaddleOCR v3.x .predict() returns a list of OCRResult objects.
        Each OCRResult is dict-like with keys: rec_texts, rec_scores, etc.
        """
        result = self._paddle.predict(image_input)

        if not result:
            return ""

        page = result[0]  # first (and only) page result
        texts = page.get("rec_texts", [])

        if not texts:
            return ""

        return "\n".join(texts)

    def _tesseract_ocr_array(self, image_array: np.ndarray) -> str:
        """Run Tesseract on a numpy image array."""
        gray = (
            cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            if len(image_array.shape) == 3
            else image_array
        )
        return self._pytesseract.image_to_string(gray)

    def _ocr_array(self, image_input) -> str:
        """Route to the active engine."""
        if self.engine_name == "paddle":
            return self._paddle_ocr_array(image_input)
        else:
            return self._tesseract_ocr_array(image_input)

    # ──────────────── PUBLIC API ────────────────

    def extract_from_image(self, image_path: str) -> str:
        """
        Called by api.py as: ocr_text = ocr_engine.extract_from_image(str(file_path))
        Returns plain text string.
        """
        if self.engine_name == "paddle":
            # PaddleOCR v3.x can accept file paths directly
            return self._paddle_ocr_array(str(image_path)).strip()

        # Tesseract path: load image as array
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return self._ocr_array(image).strip()

    def extract_from_pdf(self, pdf_path: str, dpi: int = 300) -> Dict[int, str]:
        """
        Called by api.py as:
            pages = ocr_engine.extract_from_pdf(str(file_path))
            ocr_text = "\\n".join(pages.values())
        Returns dict: {1: "page 1 text", 2: "page 2 text", ...}
        """
        # Render one page at a time so multi-page PDFs don't hold every
        # 300 DPI page (~26 MB each) in memory at once.
        page_count = pdfinfo_from_path(str(pdf_path))["Pages"]

        pages = {}
        for i in range(1, page_count + 1):
            page_img = convert_from_path(str(pdf_path), dpi=dpi, first_page=i, last_page=i)[0]
            page_array = cv2.cvtColor(np.array(page_img), cv2.COLOR_RGB2BGR)
            pages[i] = self._ocr_array(page_array).strip()

        return pages