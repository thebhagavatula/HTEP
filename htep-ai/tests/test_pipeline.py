# tests/test_pipeline.py

from src.pipeline.controller import PipelineController


def test_pipeline():
    pipeline = PipelineController()

    # Avoid external OCR binaries and fixture files in this unit test.
    pipeline.ocr.extract_from_image = lambda _: "Patient has diabete5 and takes asp1rin daily"
    result = pipeline.process_image("unused_input.png")

    assert "final_text" in result
    assert "segments" in result
