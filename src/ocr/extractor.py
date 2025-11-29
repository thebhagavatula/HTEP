# src/ocr/extractor.py

import os
import cv2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import logging
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRExtractor:
    """
    OCR Extractor for medical documents.
    Handles both PDF and image files.
    """

    def __init__(self, tesseract_path: Optional[str] = None):
        """
        Initialize OCR extractor.

        Args:
            tesseract_path: Path to Tesseract executable (if not in PATH)
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        # OCR configuration for medical documents
        self.config = r'--oem 3 --psm 6 -l eng'

    def extract_from_pdf(self, pdf_path: str) -> Dict[str, str]:
        """
        Extract text from PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with page numbers as keys and extracted text as values
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Processing PDF: {pdf_path}")

        try:
            # Convert PDF to images
            pages = convert_from_path(pdf_path, dpi=300)
            extracted_text = {}

            for page_num, page_image in enumerate(pages, 1):
                logger.info(f"Processing page {page_num}/{len(pages)}")

                # Extract text from page
                text = self._extract_from_image(page_image)
                extracted_text[f"page_{page_num}"] = text

            logger.info(f"Successfully processed {len(pages)} pages")
            return extracted_text

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise

    def extract_from_image(self, image_path: str) -> str:
        """
        Extract text from image file.

        Args:
            image_path: Path to image file

        Returns:
            Extracted text as string
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        logger.info(f"Processing image: {image_path}")

        try:
            image = Image.open(image_path)
            return self._extract_from_image(image)
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            raise

    def _extract_from_image(self, image: Image.Image) -> str:
        """
        Internal method to extract text from PIL Image.

        Args:
            image: PIL Image object

        Returns:
            Extracted text as string
        """
        # Preprocess image for better OCR
        processed_image = self._preprocess_image(image)

        # Extract text using Tesseract
        text = pytesseract.image_to_string(processed_image, config=self.config)

        return self._clean_text(text)

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image to improve OCR accuracy.

        Args:
            image: Input PIL Image

        Returns:
            Preprocessed PIL Image
        """
        # Convert PIL to OpenCV format
        img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Apply noise reduction
        denoised = cv2.medianBlur(gray, 3)

        # Apply thresholding to get better contrast
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert back to PIL
        return Image.fromarray(thresh)

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Basic text cleaning
        text = text.strip()

        # Remove excessive whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        return '\n'.join(lines)

    def get_text_confidence(self, image: Image.Image) -> List[Dict]:
        """
        Get OCR confidence scores for debugging.

        Args:
            image: PIL Image object

        Returns:
            List of dictionaries with text and confidence scores
        """
        processed_image = self._preprocess_image(image)

        # Get detailed OCR data
        data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)

        results = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 30:  # Only include confident predictions
                results.append({
                    'text': data['text'][i],
                    'confidence': data['conf'][i],
                    'bbox': (data['left'][i], data['top'][i],
                             data['width'][i], data['height'][i])
                })

        return results


# Import numpy for image processing
import numpy as np
