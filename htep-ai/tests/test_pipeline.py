# tests/test_pipeline.py

from src.pipeline.controller import PipelineController


def test_pipeline():
    pipeline = PipelineController()
    result = pipeline.process_image("tests/sample_doc.png")

    assert "final_text" in result
    assert "segments" in result
