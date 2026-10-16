# src/pipeline/digitizer.py
"""
Prescription digitizer: one uploaded file in, one reviewable record out.

    OCR (RapidOCR) -> dictionary correction -> rule-based field extraction
        -> [optional] VLM reading, checked against the OCR text -> merged record

Every field in the record says where it came from ("ocr" or "vlm") and whether
it is verified. Rule-based values come straight from the OCR text, so they are
verified. A VLM value is verified only if it can also be found in the OCR text;
otherwise it is kept but listed in `review_required` for a person to check.

When the OCR looks unreliable the response says so (`vlm.recommended`) and
includes a small preview of the first page, so the browser can run a vision
model on the user's own device and send its reading to `merge_vlm_result()`.

Lab reports and personal health records are planned to plug in here as more
document types; today every document goes through the prescription extractor.
"""

import base64
import copy
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from rapidfuzz import fuzz

from src.config import VLM_MIN_OCR_CONFIDENCE, VLM_MODE
from src.nlp.classifier import MedicalDocumentClassifier
from src.nlp.medical_extractor import MedicalDocExtractor
from src.nlp.ocr_postprocessor import OCRPostProcessor
from src.ocr.extractor import OCRExtractor, OCRPage, render_pdf_pages
from src.pipeline.grounding import is_grounded
from src.vlm.client import VLMClient

logger = logging.getLogger(__name__)

SCALAR_FIELDS = ("patient_name", "age", "gender", "date", "doctor_name", "hospital")
LIST_FIELDS = ("diagnosis", "instructions")
MEDICINE_FIELDS = ("dosage", "frequency", "duration")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# Less OCR text than this means the OCR mostly failed (e.g. handwriting).
MIN_USEFUL_OCR_CHARS = 40

# The page preview sent back for the in-browser vision model
PREVIEW_MAX_SIDE = 1280


class Digitizer:
    def __init__(self, ocr: Optional[OCRExtractor] = None, vlm: Optional[VLMClient] = None):
        self.ocr = ocr or OCRExtractor()
        self.postprocessor = OCRPostProcessor()
        self.extractor = MedicalDocExtractor(self.postprocessor.drugs, self.postprocessor.diseases)
        self.classifier = MedicalDocumentClassifier()
        self.vlm = vlm or VLMClient()

    # ------------------------------------------------------------------

    def process(self, file_path: Path) -> Dict[str, Any]:
        """OCR a file on the server, then extract the record."""
        file_path = Path(file_path)
        start = time.time()
        pages = self._read(file_path)
        timings = {"ocr": round(time.time() - start, 2)}
        return self._build(pages, timings, start, file_path=file_path, name=file_path.name)

    def process_text(self, pages: List[OCRPage], name: str = "") -> Dict[str, Any]:
        """Extract the record from text the user's browser already recognized.

        The browser runs the OCR models itself (public/ocr.js) and keeps the
        image, so there is no server-side VLM call or page preview here.
        """
        return self._build(pages, {}, time.time(), file_path=None, name=name)

    def _build(self, pages: List[OCRPage], timings: Dict[str, float], start: float,
               file_path: Optional[Path], name: str) -> Dict[str, Any]:
        ocr_text = "\n".join(p.text for p in pages).strip()
        ocr_confidence = round(min((p.confidence for p in pages), default=0.0), 3)

        t0 = time.time()
        post = self.postprocessor.process(ocr_text) if ocr_text else {}
        text = post.get("corrected_text", ocr_text)
        rules = self.extractor.extract(text) if text else {}
        timings["extraction"] = round(time.time() - t0, 2)

        record, sources = _from_rules(rules)

        reason = _vlm_reason(ocr_text, ocr_confidence, record)
        vlm_info: Dict[str, Any] = {"used": False, "recommended": reason is not None,
                                    "reason": reason, "model": None}
        page_image = None
        server_vlm = file_path is not None and self.vlm.enabled and VLM_MODE != "off"
        if server_vlm and (reason or VLM_MODE == "always"):
            t0 = time.time()
            vlm_data = self.vlm.extract_prescription(self._first_page_image(file_path))
            timings["vlm"] = round(time.time() - t0, 2)
            vlm_info.update(used=vlm_data is not None, reason=reason or "always", model=self.vlm.model)
            if vlm_data:
                _merge_vlm(record, sources, vlm_data, ocr_text)
        if reason and not vlm_info["used"] and file_path is not None:
            # No server-side VLM: let the browser read the page on the user's device
            page_image = _preview_data_url(self._first_page_image(file_path))

        classification = self._classify(text)
        timings["total"] = round(time.time() - start, 2)

        return {
            "file": name,
            "document_type": classification.get("document_type", "unknown"),
            "text": text,
            "ocr_text": ocr_text,
            "corrected_text": text,
            "ocr_confidence": ocr_confidence,
            "extracted_data": record,
            "field_sources": sources,
            "review_required": _review_required(record, sources),
            "matched_drugs": post.get("matched_drugs", []),
            "matched_diseases": post.get("matched_diseases", []),
            "corrections": post.get("corrections", []),
            "classification": classification,
            "vlm": vlm_info,
            "page_image": page_image,
            "timings": timings,
        }

    # ------------------------------------------------------------------

    def _read(self, file_path: Path) -> List[OCRPage]:
        if file_path.suffix.lower() == ".pdf":
            return self.ocr.read_pdf(str(file_path))
        return [self.ocr.read_image(str(file_path))]

    def _first_page_image(self, file_path: Path):
        if file_path.suffix.lower() in IMAGE_SUFFIXES:
            return cv2.imread(str(file_path))
        return next(render_pdf_pages(file_path, dpi=150, first_n=1), None)

    def _classify(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}
        try:
            result = self.classifier.classify_document(text)
            urgency, urgency_conf = self.classifier.get_document_urgency(text)
            return {
                "document_type": result.document_type,
                "confidence": round(result.confidence, 2),
                "urgency": urgency,
                "urgency_confidence": round(urgency_conf, 2),
                "keywords_found": result.keywords_found[:10],
            }
        except Exception:
            logger.exception("Classification failed")
            return {}


# ----------------------------------------------------------------------
# When to ask a vision model
# ----------------------------------------------------------------------

def _vlm_reason(ocr_text: str, ocr_confidence: float, record: Dict[str, Any]) -> Optional[str]:
    """Why a VLM should read this document, or None if the OCR looks reliable."""
    if len(ocr_text) < MIN_USEFUL_OCR_CHARS:
        return "little_ocr_text"
    if ocr_confidence < VLM_MIN_OCR_CONFIDENCE:
        return "low_ocr_confidence"
    if not record["medicines"]:
        return "no_medicines_found"
    return None


def _preview_data_url(image) -> Optional[str]:
    if image is None:
        return None
    h, w = image.shape[:2]
    scale = PREVIEW_MAX_SIDE / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


# ----------------------------------------------------------------------
# Merging
# ----------------------------------------------------------------------

def merge_vlm_result(record: Dict[str, Any], sources: Dict[str, Any], vlm_data: Dict[str, Any],
                     ocr_text: str) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """Merge a VLM reading made elsewhere (the user's browser) into a record.

    Same rules as the server-side VLM: VLM values only fill empty fields, and a
    value counts as verified only if it is found in the OCR text.
    """
    record = copy.deepcopy(record) if isinstance(record, dict) else {}
    sources = copy.deepcopy(sources) if isinstance(sources, dict) else {}
    for f in SCALAR_FIELDS:
        record[f] = _clean(record.get(f))
    for f in LIST_FIELDS:
        record[f] = [str(x) for x in record.get(f) or [] if _clean(x)]
    record["vitals"] = record.get("vitals") if isinstance(record.get("vitals"), dict) else {}
    record["medicines"] = [
        {"name": _clean(m.get("name")), **{k: _clean(m.get(k)) for k in MEDICINE_FIELDS},
         "source": m.get("source", "ocr"), "verified": bool(m.get("verified", True))}
        for m in record.get("medicines") or [] if isinstance(m, dict) and _clean(m.get("name"))
    ]
    sources = {f: {"source": str(v.get("source", "ocr")), "verified": bool(v.get("verified", True))}
               for f, v in sources.items() if isinstance(v, dict)}
    if isinstance(vlm_data, dict):
        _merge_vlm(record, sources, vlm_data, ocr_text or "")
    return record, sources, _review_required(record, sources)


def _from_rules(rules: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    record: Dict[str, Any] = {f: rules.get(f) for f in SCALAR_FIELDS}
    for f in LIST_FIELDS:
        record[f] = list(rules.get(f) or [])
    record["vitals"] = dict(rules.get("vitals") or {})
    record["medicines"] = [
        {**{k: m.get(k) for k in ("name",) + MEDICINE_FIELDS}, "source": "ocr", "verified": True}
        for m in rules.get("medicines") or []
    ]
    sources = {f: {"source": "ocr", "verified": True} for f in SCALAR_FIELDS + LIST_FIELDS if record[f]}
    return record, sources


def _clean(value) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _same_name(a: str, b: str) -> bool:
    return fuzz.ratio(a.lower(), b.lower()) >= 85 or a.lower() in b.lower() or b.lower() in a.lower()


def _merge_vlm(record, sources, vlm: Dict[str, Any], ocr_text: str) -> None:
    """Fill gaps in the rule-based record with VLM values, checked against the OCR text."""
    for f in SCALAR_FIELDS:
        value = _clean(vlm.get(f))
        if value and not record[f]:
            record[f] = value
            sources[f] = {"source": "vlm", "verified": is_grounded(value, ocr_text)}

    for f in LIST_FIELDS:
        added_unverified = False
        for item in vlm.get(f) or []:
            item = _clean(item)
            if not item or any(_same_name(item, existing) for existing in record[f]):
                continue
            record[f].append(item)
            added_unverified |= not is_grounded(item, ocr_text)
            if f in sources and sources[f]["source"] == "ocr":
                sources[f]["source"] = "ocr+vlm"
            sources.setdefault(f, {"source": "vlm", "verified": True})
        if added_unverified:
            sources[f]["verified"] = False

    for vm in vlm.get("medicines") or []:
        name = _clean(vm.get("name")) if isinstance(vm, dict) else None
        if not name:
            continue
        fields = {k: _clean(vm.get(k)) for k in MEDICINE_FIELDS}
        match = next((m for m in record["medicines"] if _same_name(m["name"], name)), None)
        if match:
            for k, v in fields.items():
                if v and not match.get(k):
                    match[k] = v
                    if match.get("source") == "ocr":
                        match["source"] = "ocr+vlm"
                    if not is_grounded(v, ocr_text):
                        match["verified"] = False
        else:
            verified = is_grounded(name, ocr_text) and all(
                v is None or is_grounded(v, ocr_text) for v in fields.values()
            )
            record["medicines"].append({"name": name, **fields, "source": "vlm", "verified": verified})


def _review_required(record, sources) -> List[str]:
    review = [f for f, meta in sources.items() if not meta["verified"]]
    review += [
        f"medicines[{i}]" for i, m in enumerate(record["medicines"]) if not m.get("verified", True)
    ]
    return review
