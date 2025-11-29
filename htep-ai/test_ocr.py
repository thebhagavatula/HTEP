# test_ocr.py - Run this to test your OCR setup

import os
import sys
from pathlib import Path

# Add src to path so we can import our modules
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.ocr.extractor import OCRExtractor


def test_ocr_setup():
    """Test if OCR is properly configured."""
    print("Testing OCR Setup...")
    print("-" * 50)

    try:
        # Initialize OCR extractor
        extractor = OCRExtractor()
        print("OCR Extractor initialized successfully")

        # Test with a simple image (you'll need to add a test image)
        sample_dir = Path("data/sample")

        if not sample_dir.exists():
            print("Warning: data/sample directory doesn't exist")
            sample_dir.mkdir(parents=True, exist_ok=True)
            print("✓ Created data/sample directory")

        # Look for PDF files in sample directory
        pdf_files = list(sample_dir.glob("*.pdf"))
        image_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))

        if pdf_files:
            print(f"Found {len(pdf_files)} PDF file(s)")
            test_pdf = pdf_files[0]
            print(f"Testing with: {test_pdf}")

            try:
                result = extractor.extract_from_pdf(str(test_pdf))
                print(f"Successfully extracted text from {len(result)} page(s)")

                # Show preview of first page
                first_page = list(result.keys())[0]
                preview = result[first_page][:200] + "..." if len(result[first_page]) > 200 else result[first_page]
                print(f"\nPreview of {first_page}:")
                print("-" * 30)
                print(preview)

            except Exception as e:
                print(f"Error processing PDF: {e}")

        elif image_files:
            print(f"Found {len(image_files)} image file(s)")
            test_image = image_files[0]
            print(f"Testing with: {test_image}")

            try:
                result = extractor.extract_from_image(str(test_image))
                print("Successfully extracted text from image")

                preview = result[:200] + "..." if len(result) > 200 else result
                print(f"\nExtracted text preview:")
                print("-" * 30)
                print(preview)

            except Exception as e:
                print(f"✗ Error processing image: {e}")

        else:
            print("No PDF or image files found in data/sample/")
            print("Add some sample files to test OCR functionality")

        print("\n" + "=" * 50)
        print("OCR Test Complete!")

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you've installed all requirements:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")


def check_dependencies():
    """Check if all required packages are installed."""
    print("Checking Dependencies...")
    print("-" * 50)

    required_packages = [
        'cv2', 'pytesseract', 'pdf2image', 'PIL', 'numpy'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            elif package == 'pytesseract':
                import pytesseract
            elif package == 'pdf2image':
                import pdf2image
            elif package == 'numpy':
                import numpy

            print(f"{package}")

        except ImportError:
            print(f"{package} - NOT INSTALLED")
            missing_packages.append(package)

    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True


if __name__ == "__main__":
    print("Medical Document AI - OCR Test")
    print("=" * 50)

    # Check dependencies first
    if check_dependencies():
        print()
        test_ocr_setup()
    else:
        print("\nPlease install missing dependencies before testing OCR.")