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

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
import pypdfium2 as pdfium

# PaddleX's default text detection only enforces a *minimum* side (64px) and
# lets images up to 4000px through at full size. On a 12 MP phone photo or a
# 300 DPI page the detector alone needs >3 GB and Cloud Run OOM-kills the
# container. Cap the longest side used for detection; recognition still reads
# text crops from the full-resolution image.
DET_MAX_SIDE = 1600

from src.config import MAX_PDF_PAGES, OCR_DET_MODEL_PATH, OCR_ENGINE, OCR_REC_MODEL_PATH


class DocumentTooLargeError(ValueError):
    """The upload is over a processing limit, e.g. too many PDF pages."""


def render_pdf_pages(pdf_path, dpi: int = 200, first_n: int = None):
    """Yield the PDF's pages as BGR images, one at a time to keep memory flat.

    pypdfium2 is a self-contained wheel; pdf2image needed the Poppler program,
    which serverless hosts such as Vercel don't have.
    """
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        count = len(pdf) if first_n is None else min(first_n, len(pdf))
        for i in range(count):
            page = pdf[i]
            image = page.render(scale=dpi / 72).to_pil().convert("RGB")
            page.close()
            yield cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    finally:
        pdf.close()


def pdf_page_count(pdf_path) -> int:
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return len(pdf)
    finally:
        pdf.close()


@dataclass
class OCRPage:
    """Text of one image or PDF page, with the mean recognition confidence (0-1)."""
    text: str
    confidence: float


class OCRExtractor:
    """
    Unified OCR extractor. Public interface matches what api.py expects:
      - extract_from_pdf(path) -> Dict[int, str]   {page_num: text}
      - extract_from_image(path) -> str
    Internally routes to RapidOCR (default), PaddleOCR or Tesseract
    based on OCR_ENGINE config in src/config.py.
    """

    def __init__(self, lang: str = "en", device: str = "cpu"):
        self.engine_name = OCR_ENGINE

        if self.engine_name == "rapid":
            from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

            # PP-OCRv6 small detection + the English PP-OCRv5 mobile recognizer
            # (the one PaddleOCR used with lang="en") on ONNX Runtime: no
            # PaddlePaddle install and several times faster on CPU. On
            # tests/ground_truth.json the CER is 0.63% (PaddleOCR: 1.45%). The
            # default multilingual recognizer scored 0.92% there but misread
            # labels on real handwriting ("neme", "601F"), so it is not used.
            # The models are committed under models/ocr/ and loaded by path:
            # RapidOCR would otherwise download them into its package folder,
            # which is read-only on serverless hosts.
            self._rapid = RapidOCR(params={
                "Global.use_cls": False,
                "Det.model_path": str(OCR_DET_MODEL_PATH),
                "Rec.model_path": str(OCR_REC_MODEL_PATH),
                "Rec.lang_type": LangRec.EN,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Rec.model_type": ModelType.MOBILE,
            })
            # Models load on first use; load them now so the first upload
            # doesn't pay for it.
            warmup = np.full((48, 160, 3), 255, dtype=np.uint8)
            cv2.putText(warmup, "Rx 10mg", (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            self._rapid(warmup)

        elif self.engine_name == "paddle":
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

    def _rapid_ocr_page(self, image_input) -> OCRPage:
        """Run RapidOCR on an image (path string or numpy array)."""
        result = self._rapid(image_input)
        texts = list(result.txts or [])
        scores = [float(s) for s in (result.scores or [])]
        confidence = sum(scores) / len(scores) if scores else 0.0
        return OCRPage("\n".join(texts).strip(), round(confidence, 3))

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

    def _ocr_page(self, image_input) -> OCRPage:
        """Route to the active engine. Only RapidOCR reports a confidence."""
        if self.engine_name == "rapid":
            return self._rapid_ocr_page(image_input)
        if self.engine_name == "paddle":
            return OCRPage(self._paddle_ocr_array(image_input).strip(), 1.0)
        if isinstance(image_input, str):
            image_input = cv2.imread(image_input)
            if image_input is None:
                raise ValueError("Could not read image")
        return OCRPage(self._tesseract_ocr_array(image_input).strip(), 1.0)

    def _ocr_array(self, image_input) -> str:
        """Route to the active engine."""
        return self._ocr_page(image_input).text

    # ──────────────── PUBLIC API ────────────────

    def read_image(self, image_path: str) -> OCRPage:
        return self._ocr_page(str(image_path))

    def read_pdf(self, pdf_path: str, dpi: int = 200) -> List[OCRPage]:
        # 200 DPI reads the same text as 300 DPI on our samples, faster and in less memory
        page_count = pdf_page_count(pdf_path)
        if page_count > MAX_PDF_PAGES:
            raise DocumentTooLargeError(
                f"This PDF has {page_count} pages; the limit is {MAX_PDF_PAGES}. "
                "Upload the prescription pages only."
            )
        return [self._ocr_page(image) for image in render_pdf_pages(pdf_path, dpi)]

    def extract_from_image(self, image_path: str) -> str:
        """Plain-text OCR of an image (used by the evaluation scripts)."""
        return self.read_image(image_path).text

    def extract_from_pdf(self, pdf_path: str, dpi: int = 200) -> Dict[int, str]:
        """Plain-text OCR of a PDF: {page_num: text}."""
        return {i: page.text for i, page in enumerate(self.read_pdf(pdf_path, dpi), start=1)}
