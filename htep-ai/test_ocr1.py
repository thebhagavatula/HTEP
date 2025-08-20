# test_ocr.py - Focused OCR optimization with better diagnostics and performance

import os
import sys
import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional

# Add src to path so we can import our modules
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


class ColoredOutput:
    """Simple colored output for better readability."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

    @classmethod
    def success(cls, text: str) -> str:
        return f"{cls.GREEN}✓{cls.END} {text}"

    @classmethod
    def error(cls, text: str) -> str:
        return f"{cls.RED}✗{cls.END} {text}"

    @classmethod
    def warning(cls, text: str) -> str:
        return f"{cls.YELLOW}⚠{cls.END} {text}"

    @classmethod
    def info(cls, text: str) -> str:
        return f"{cls.BLUE}ℹ{cls.END} {text}"


def check_system_dependencies() -> bool:
    """Check system-level dependencies like Tesseract."""
    print("\nChecking System Dependencies...")
    print("-" * 50)

    try:
        import pytesseract

        # Test if Tesseract is actually available
        try:
            version = pytesseract.get_tesseract_version()
            print(ColoredOutput.success(f"Tesseract OCR {version}"))
            return True
        except Exception as e:
            print(ColoredOutput.error(f"Tesseract OCR - {str(e)}"))
            print(ColoredOutput.info("Install Tesseract OCR:"))
            print("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
            print("  macOS: brew install tesseract")
            print("  Ubuntu: sudo apt install tesseract-ocr")
            return False

    except ImportError:
        print(ColoredOutput.error("pytesseract not installed"))
        return False


def check_python_dependencies() -> Tuple[bool, List[str]]:
    """Enhanced dependency checking with version info."""
    print("\nChecking Python Dependencies...")
    print("-" * 50)

    required_packages = [
        ('cv2', 'opencv-python'),
        ('pytesseract', 'pytesseract'),
        ('pdf2image', 'pdf2image'),
        ('PIL', 'Pillow'),
        ('numpy', 'numpy')
    ]

    missing_packages = []
    all_good = True

    for module, package in required_packages:
        try:
            if module == 'PIL':
                import PIL
                version = PIL.__version__
            elif module == 'cv2':
                import cv2
                version = cv2.__version__
            elif module == 'pytesseract':
                import pytesseract
                version = getattr(pytesseract, '__version__', 'unknown')
            elif module == 'pdf2image':
                import pdf2image
                version = getattr(pdf2image, '__version__', 'unknown')
            elif module == 'numpy':
                import numpy
                version = numpy.__version__

            print(ColoredOutput.success(f"{module} v{version}"))

        except ImportError as e:
            print(ColoredOutput.error(f"{module} ({package}) - NOT INSTALLED"))
            print(f"  Error: {str(e)}")
            missing_packages.append(package)
            all_good = False
        except Exception as e:
            print(ColoredOutput.warning(f"{module} - Installed but issue: {str(e)}"))

    if missing_packages:
        print(f"\n{ColoredOutput.error('Missing packages:')} {', '.join(missing_packages)}")
        print(f"{ColoredOutput.info('Install with:')} pip install {' '.join(missing_packages)}")

    return all_good, missing_packages


def test_ocr_import() -> bool:
    """Test if our OCR extractor can be imported."""
    print("\nTesting OCR Extractor Import...")
    print("-" * 50)

    try:
        from src.ocr.extractor import OCRExtractor
        extractor = OCRExtractor()
        print(ColoredOutput.success("OCR Extractor imported and initialized"))
        return True
    except ImportError as e:
        print(ColoredOutput.error(f"Cannot import OCRExtractor: {str(e)}"))
        print(ColoredOutput.info("Check that src/ocr/extractor.py exists and is correct"))
        return False
    except Exception as e:
        print(ColoredOutput.error(f"Error initializing OCRExtractor: {str(e)}"))
        print(f"Full error:\n{traceback.format_exc()}")
        return False


def find_test_files(sample_dir: Path) -> Dict[str, List[Path]]:
    """Find and categorize test files."""
    if not sample_dir.exists():
        print(ColoredOutput.warning(f"Sample directory doesn't exist: {sample_dir}"))
        sample_dir.mkdir(parents=True, exist_ok=True)
        print(ColoredOutput.success(f"Created directory: {sample_dir}"))
        return {"pdf": [], "images": []}

    # Find files
    pdf_files = list(sample_dir.glob("*.pdf")) + list(sample_dir.glob("*.PDF"))
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.gif"]
    image_files = []

    for ext in image_extensions:
        image_files.extend(sample_dir.glob(ext))
        image_files.extend(sample_dir.glob(ext.upper()))

    return {"pdf": pdf_files, "images": image_files}


def process_single_file(file_path: Path, extractor) -> Dict:
    """Process a single file with detailed error reporting."""
    start_time = time.time()
    result = {
        "filename": file_path.name,
        "path": str(file_path),
        "success": False,
        "error": None,
        "processing_time": 0,
        "text_length": 0,
        "page_count": 0,
        "preview": ""
    }

    try:
        if file_path.suffix.lower() in ['.pdf']:
            print(f"  📄 Processing PDF: {file_path.name}")
            extracted_data = extractor.extract_from_pdf(str(file_path))

            if isinstance(extracted_data, dict):
                all_text = '\n'.join(extracted_data.values())
                result["page_count"] = len(extracted_data)
            else:
                all_text = str(extracted_data)
                result["page_count"] = 1

        else:
            print(f"  🖼️  Processing Image: {file_path.name}")
            all_text = extractor.extract_from_image(str(file_path))
            result["page_count"] = 1

        # Process results
        result["text_length"] = len(all_text)
        result["preview"] = all_text[:200] + ("..." if len(all_text) > 200 else "")
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        print(f"    {ColoredOutput.error('Failed')}: {str(e)}")

    result["processing_time"] = time.time() - start_time
    return result


def process_files_parallel(files: List[Path], extractor, max_workers: int = 3) -> List[Dict]:
    """Process multiple files in parallel."""
    if len(files) <= 1:
        return [process_single_file(files[0], extractor)] if files else []

    print(f"\n🚀 Processing {len(files)} files with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda f: process_single_file(f, extractor), files))

    return results


def generate_detailed_report(results: List[Dict]):
    """Generate a comprehensive report."""
    if not results:
        print(ColoredOutput.warning("No files were processed"))
        return

    print(f"\n{'=' * 70}")
    print(f"{ColoredOutput.info('OCR PROCESSING RESULTS')}")
    print(f"{'=' * 70}")

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    # Summary statistics
    total_time = sum(r["processing_time"] for r in results)
    total_text = sum(r["text_length"] for r in successful)
    total_pages = sum(r["page_count"] for r in successful)

    print(f"\n📊 SUMMARY:")
    print(f"  Total files: {len(results)}")
    print(f"  {ColoredOutput.success(f'Successful: {len(successful)}')} ")
    print(f"  {ColoredOutput.error(f'Failed: {len(failed)}')} ")
    print(f"  Total processing time: {total_time:.2f}s")
    print(f"  Average time per file: {total_time / len(results):.2f}s")
    print(f"  Total pages processed: {total_pages}")
    print(f"  Total text extracted: {total_text:,} characters")

    if successful:
        avg_chars = total_text / len(successful)
        print(f"  Average text per file: {avg_chars:.0f} characters")

    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    print("-" * 70)

    for i, result in enumerate(results, 1):
        status_icon = "✅" if result["success"] else "❌"
        print(f"\n{i}. {status_icon} {result['filename']}")
        print(f"   ⏱️  Processing time: {result['processing_time']:.2f}s")

        if result["success"]:
            print(f"   📄 Pages: {result['page_count']}")
            print(f"   📝 Text length: {result['text_length']:,} characters")

            if result["preview"]:
                print(f"   👀 Preview:")
                # Indent the preview text
                preview_lines = result["preview"].split('\n')
                for line in preview_lines[:3]:  # Show max 3 lines
                    if line.strip():
                        print(f"      {line.strip()}")
                if len(preview_lines) > 3:
                    print("      ...")
        else:
            print(f"   {ColoredOutput.error('Error')}: {result['error']}")

    # Performance insights
    if successful:
        fastest = min(successful, key=lambda x: x["processing_time"])
        slowest = max(successful, key=lambda x: x["processing_time"])

        print(f"\n🏆 PERFORMANCE INSIGHTS:")
        print(f"   Fastest: {fastest['filename']} ({fastest['processing_time']:.2f}s)")
        print(f"   Slowest: {slowest['filename']} ({slowest['processing_time']:.2f}s)")

        if total_pages > 0:
            print(f"   Pages per second: {total_pages / total_time:.1f}")


def main():
    """Main enhanced test function."""
    print(f"{ColoredOutput.info('Medical Document AI - Enhanced OCR Test')}")
    print("=" * 60)

    # Step 1: Check system dependencies
    system_ok = check_system_dependencies()

    # Step 2: Check Python dependencies
    python_ok, missing = check_python_dependencies()

    if not (system_ok and python_ok):
        print(f"\n{ColoredOutput.error('Dependencies not satisfied. Please install missing components.')}")
        return False

    # Step 3: Test OCR extractor import
    if not test_ocr_import():
        return False

    # Step 4: Find test files
    sample_dir = Path("data/sample")
    files = find_test_files(sample_dir)
    total_files = len(files["pdf"]) + len(files["images"])

    print(f"\n🔍 File Discovery:")
    print(f"   PDF files: {len(files['pdf'])}")
    print(f"   Image files: {len(files['images'])}")
    print(f"   Total: {total_files}")

    if total_files == 0:
        print(f"\n{ColoredOutput.warning('No test files found!')}")
        print(f"Add PDF or image files to: {sample_dir}")
        return False

    # Step 5: Process files
    try:
        from src.ocr.extractor import OCRExtractor
        extractor = OCRExtractor()

        all_files = files["pdf"] + files["images"]
        start_time = time.time()

        # Process files (use parallel processing if more than 2 files)
        if len(all_files) > 2:
            results = process_files_parallel(all_files, extractor, max_workers=3)
        else:
            results = [process_single_file(f, extractor) for f in all_files]

        total_time = time.time() - start_time

        # Step 6: Generate report
        generate_detailed_report(results)

        print(f"\n🏁 {ColoredOutput.success('Test completed!')} Total time: {total_time:.2f}s")
        return True

    except Exception as e:
        print(f"\n{ColoredOutput.error('Unexpected error during testing:')}")
        print(f"{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n{ColoredOutput.warning('Test interrupted by user')}")
    except Exception as e:
        print(f"\n{ColoredOutput.error('Critical error:')}: {str(e)}")
        sys.exit(1)