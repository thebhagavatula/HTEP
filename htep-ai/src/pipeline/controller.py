# src/pipeline/controller.py
# OCR + ICR + NLP master pipeline

from typing import Dict
from src.ocr.extractor import OCRExtractor
from src.icr.inference import ICRInference
from src.pipeline.fusion import OCRICRFusion
from src.segmentation.medical_segmenter import MedicalDocumentSegmenter


class PipelineController:
    """
    Orchestrates OCR → ICR → Fusion → Segmentation
    """

    def __init__(self):
        self.ocr = OCRExtractor()
        self.icr = ICRInference()
        self.fusion = OCRICRFusion()
        self.segmenter = MedicalDocumentSegmenter()

    def process_image(self, image_path: str) -> Dict:
        """
        Full pipeline for an image document.
        """
        # 1️⃣ OCR
        ocr_text = self.ocr.extract_from_image(image_path)

        # 2️⃣ (Optional) ICR-based correction
        fused_text = self.fusion.correct_text(ocr_text)

        # 3️⃣ Medical segmentation
        segments = self.segmenter.segment_document(fused_text)

        return {
            "raw_ocr_text": ocr_text,
            "final_text": fused_text,
            "segments": [
                {
                    "type": s.segment_type,
                    "content": s.content,
                    "confidence": s.confidence
                } for s in segments
            ]
        }
