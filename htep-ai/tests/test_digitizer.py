# tests/test_digitizer.py
"""Digitizer, grounding and job API tests. Fake OCR and VLM objects keep these fast."""

import time

import pytest

from src.nlp.medical_extractor import MedicalDocExtractor
from src.ocr.extractor import OCRPage
from src.pipeline.digitizer import Digitizer
from src.pipeline.grounding import is_grounded

PRESCRIPTION = """CityCare Hospital
Dr. Ramesh Rao MBBS
Name: Mr. K. Suresh   Age: 45 Sex: M
Date: 12/10/2026
Diagnosis: Viral fever
Rx
1. Tab. Dolo 650mg 1-0-1 after food x 5 days
2. Cap. Omez 20 mg OD before breakfast
Advice: Drink plenty of fluids"""


class FakeOCR:
    engine_name = "fake"

    def __init__(self, text, confidence=0.95):
        self.page = OCRPage(text, confidence)

    def read_image(self, _path):
        return self.page

    def read_pdf(self, _path):
        return [self.page]


class FakeVLM:
    model = "fake-vlm"

    def __init__(self, data=None, enabled=True):
        self.data = data
        self.enabled = enabled
        self.calls = 0

    def extract_prescription(self, _image):
        self.calls += 1
        return self.data


# ---------------------------------------------------------------- grounding

@pytest.mark.parametrize("value, expected", [
    ("Dolo 650mg", True),            # exact
    ("Amoxicillin", True),           # OCR read "Amoxicil1in"
    ("45", True),                    # short value, whole word
    ("4", False),                    # short value must not match inside "45"
    ("Azithromycin 500mg", False),   # not on the page
    (None, False),
])
def test_is_grounded(value, expected):
    ocr = "Tab. Dolo 650mg 1-0-1\nCap Amoxicil1in 250mg\nAge: 45"
    assert is_grounded(value, ocr) is expected


# ---------------------------------------------------------------- rules

def test_rules_parse_duration_and_dosage_forms():
    data = MedicalDocExtractor([], []).extract(PRESCRIPTION)
    dolo = next(m for m in data["medicines"] if m["name"] == "Dolo")
    assert dolo == {"name": "Dolo", "dosage": "650mg", "frequency": "1-0-1", "duration": "5 days"}
    assert data["patient_name"] == "K. Suresh"
    assert data["doctor_name"] == "Dr. Ramesh Rao"


# ---------------------------------------------------------------- digitizer

def test_confident_ocr_skips_vlm():
    vlm = FakeVLM({"medicines": [{"name": "Invented"}]})
    result = Digitizer(ocr=FakeOCR(PRESCRIPTION), vlm=vlm).process("rx.png")

    assert vlm.calls == 0
    assert result["vlm"]["used"] is False
    assert result["review_required"] == []
    names = [m["name"] for m in result["extracted_data"]["medicines"]]
    assert names == ["Dolo", "Omez"]
    assert all(m["source"] == "ocr" and m["verified"] for m in result["extracted_data"]["medicines"])


def test_vlm_fills_gaps_and_flags_values_not_in_ocr_text():
    # Poor OCR of a handwritten prescription: low confidence, no structure
    ocr_text = "Suresh\nAmoxicil1in 250mg tds\nparacetamol"
    vlm = FakeVLM({
        "patient_name": "Suresh",
        "age": "45",                                   # not in the OCR text
        "medicines": [
            {"name": "Amoxicillin", "dosage": "250mg", "frequency": "TDS", "duration": None},
            {"name": "Azithromycin", "dosage": "500mg"},  # invented
        ],
        "diagnosis": [],
    })
    result = Digitizer(ocr=FakeOCR(ocr_text, confidence=0.6), vlm=vlm).process("rx.png")
    data, sources = result["extracted_data"], result["field_sources"]

    assert result["vlm"] == {"used": True, "reason": "low_ocr_confidence", "model": "fake-vlm"}
    assert data["patient_name"] == "Suresh"
    assert sources["patient_name"] == {"source": "vlm", "verified": True}
    assert sources["age"] == {"source": "vlm", "verified": False}

    meds = {m["name"]: m for m in data["medicines"]}
    assert meds["Amoxicillin"]["verified"] is True
    assert meds["Azithromycin"]["verified"] is False
    assert "age" in result["review_required"]
    assert any(r.startswith("medicines[") for r in result["review_required"])


def test_vlm_failure_falls_back_to_ocr_only():
    vlm = FakeVLM(data=None)
    result = Digitizer(ocr=FakeOCR("Rx\nTab. Dolo 650mg", confidence=0.5), vlm=vlm).process("rx.png")
    assert vlm.calls == 1
    assert result["vlm"]["used"] is False
    assert [m["name"] for m in result["extracted_data"]["medicines"]] == ["Dolo"]


def test_disabled_vlm_is_never_called():
    vlm = FakeVLM({"patient_name": "X"}, enabled=False)
    Digitizer(ocr=FakeOCR("", confidence=0.0), vlm=vlm).process("rx.png")
    assert vlm.calls == 0


# ---------------------------------------------------------------- API

class FakeDigitizer:
    class vlm:
        enabled = False
        model = None

        @staticmethod
        def is_reachable():
            return False

    class ocr:
        engine_name = "fake"

    def process(self, file_path):
        assert file_path.exists()
        return {"extracted_data": {"medicines": []}, "timings": {"total": 0.0},
                "ocr_confidence": 1.0, "vlm": {"used": False}, "review_required": []}


@pytest.fixture
def client(monkeypatch):
    from app import api
    monkeypatch.setattr(api, "digitizer", FakeDigitizer())
    return api.app.test_client(), api


def test_job_runs_and_deletes_upload(client, tmp_path, monkeypatch):
    test_client, api = client
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path)

    import io
    resp = test_client.post("/jobs", data={"file": (io.BytesIO(b"fake"), "rx.png")})
    assert resp.status_code == 202
    job_id = resp.get_json()["job_id"]

    for _ in range(50):
        job = test_client.get(f"/jobs/{job_id}").get_json()
        if job["status"] != "pending":
            break
        time.sleep(0.05)

    assert job["status"] == "done"
    assert job["result"]["file"] == "rx.png"
    assert list(tmp_path.iterdir()) == []   # upload deleted after processing


def test_rejects_unsupported_file_type(client):
    import io
    test_client, _ = client
    resp = test_client.post("/jobs", data={"file": (io.BytesIO(b"x"), "notes.docx")})
    assert resp.status_code == 400
