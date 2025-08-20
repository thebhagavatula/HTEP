# test_ocr.py - Enhanced OCR test with segmentation and classification

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add src to path so we can import our modules
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.ocr.extractor import OCRExtractor
from src.segmentation.segmenter import MedicalDocumentSegmenter, SimpleTextProcessor
from src.classification.classifier import MedicalDocumentClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MedicalDocumentProcessor:
    """Main processor for medical documents with OCR, segmentation, and classification."""

    def __init__(self):
        """Initialize all components."""
        try:
            self.ocr_extractor = OCRExtractor()
            self.segmenter = MedicalDocumentSegmenter()
            self.classifier = MedicalDocumentClassifier()
            self.text_processor = SimpleTextProcessor()
            print("✓ All components initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize components: {e}")
            raise

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a single PDF through the complete pipeline.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing all processing results
        """
        print(f"\n📄 Processing: {Path(pdf_path).name}")
        print("=" * 80)

        results = {
            'filename': Path(pdf_path).name,
            'file_path': pdf_path,
            'pages': {},
            'overall_classification': None,
            'processing_errors': []
        }

        try:
            # Step 1: Extract text from PDF
            extracted_text = self.ocr_extractor.extract_from_pdf(pdf_path)

            if not extracted_text:
                results['processing_errors'].append("No text extracted from PDF")
                print("❌ No text extracted from PDF")
                return results

            print(f"✓ Extracted text from {len(extracted_text)} page(s)")

            # Process each page
            all_segments = []
            page_results = {}

            for page_name, page_text in extracted_text.items():
                print(f"\n📑 {page_name.upper()}")
                print("-" * 80)

                # Show the extracted text first
                print("🔤 EXTRACTED TEXT:")
                print("-" * 40)
                print(page_text)
                print("-" * 40)

                # Step 2: Clean the text
                cleaned_text = self.text_processor.clean_medical_text(page_text)

                # Step 3: Segment the text
                segments = self.segmenter.segment_document(cleaned_text)
                print(f"\n🔍 SEGMENTATION ANALYSIS:")
                print(f"Found {len(segments)} segments:")

                # Show each segment
                for i, segment in enumerate(segments, 1):
                    print(f"\n  📝 Segment {i}: {segment.segment_type.replace('_', ' ').title()}")
                    print(f"     Lines {segment.start_line}-{segment.end_line}")
                    print(f"     Content: {segment.content[:150]}{'...' if len(segment.content) > 150 else ''}")

                # Step 4: Classify each segment and the whole page
                print(f"\n🏷️  CLASSIFICATION RESULTS:")

                segment_classifications = []
                for i, segment in enumerate(segments, 1):
                    classification = self.classifier.classify_document(segment.content)
                    print(
                        f"  Segment {i} ({segment.segment_type}): {classification.document_type} ({classification.confidence:.1%})")
                    if classification.keywords_found:
                        print(f"    Keywords: {', '.join(classification.keywords_found[:5])}")

                    segment_classifications.append({
                        'segment_type': segment.segment_type,
                        'content_preview': segment.content[:200] + "..." if len(
                            segment.content) > 200 else segment.content,
                        'full_content': segment.content,
                        'classification': {
                            'document_type': classification.document_type,
                            'confidence': classification.confidence,
                            'keywords_found': classification.keywords_found
                        },
                        'start_line': segment.start_line,
                        'end_line': segment.end_line
                    })

                # Step 5: Get overall page classification
                page_classification = self.classifier.classify_document(cleaned_text)
                print(
                    f"\n  📄 Overall Page Classification: {page_classification.document_type.replace('_', ' ').title()}")
                print(f"     Confidence: {page_classification.confidence:.1%}")
                if page_classification.secondary_types:
                    print(
                        f"     Secondary possibilities: {', '.join([f'{t[0]} ({t[1]:.1%})' for t in page_classification.secondary_types[:2]])}")

                # Step 6: Extract additional information
                print(f"\n🏥 MEDICAL INFORMATION EXTRACTION:")

                key_value_pairs = {}
                medical_entities = self.classifier.extract_medical_entities(cleaned_text)
                urgency_level, urgency_confidence = self.classifier.get_document_urgency(cleaned_text)

                # Extract key-value pairs from patient info segments
                for segment in segments:
                    if segment.segment_type == 'patient_info':
                        kvp = self.segmenter.extract_key_value_pairs(segment)
                        key_value_pairs.update(kvp)

                # Display extracted information
                if key_value_pairs:
                    print("  📋 Patient Information:")
                    for key, value in key_value_pairs.items():
                        print(f"     {key.title()}: {value}")

                # Show medical entities
                entity_count = sum(len(entities) for entities in medical_entities.values())
                if entity_count > 0:
                    print(f"  💊 Medical Entities Found ({entity_count} total):")
                    for entity_type, entities in medical_entities.items():
                        if entities:
                            print(f"     {entity_type.title()}: {', '.join(entities[:5])}")
                            if len(entities) > 5:
                                print(f"       ... and {len(entities) - 5} more")

                # Urgency assessment
                print(f"  ⚠️  Urgency Level: {urgency_level.title()} ({urgency_confidence:.1%})")

                # Extract dates and other info
                dates = self.text_processor.extract_dates(cleaned_text)
                if dates:
                    print(f"  📅 Dates Found: {', '.join(dates[:3])}")

                page_results[page_name] = {
                    'raw_text': page_text,
                    'cleaned_text': cleaned_text,
                    'text_length': len(cleaned_text),
                    'segments_count': len(segments),
                    'segments': segment_classifications,
                    'overall_classification': {
                        'document_type': page_classification.document_type,
                        'confidence': page_classification.confidence,
                        'secondary_types': page_classification.secondary_types
                    },
                    'medical_entities': medical_entities,
                    'urgency_assessment': {
                        'level': urgency_level,
                        'confidence': urgency_confidence
                    },
                    'key_value_pairs': key_value_pairs,
                    'dates_found': dates
                }

                all_segments.extend(segments)

            results['pages'] = page_results

            # Overall document classification (using all text)
            if extracted_text:
                combined_text = " ".join(extracted_text.values())
                overall_classification = self.classifier.classify_document(combined_text)
                results['overall_classification'] = {
                    'document_type': overall_classification.document_type,
                    'confidence': overall_classification.confidence,
                    'secondary_types': overall_classification.secondary_types
                }
                print(f"\n🎯 FINAL DOCUMENT CLASSIFICATION:")
                print(f"   Type: {overall_classification.document_type.replace('_', ' ').title()}")
                print(f"   Confidence: {overall_classification.confidence:.1%}")

            # Summary statistics
            segment_summary = self.segmenter.get_segment_summary(all_segments)
            results['segment_summary'] = segment_summary
            results['total_segments'] = len(all_segments)

            print(f"\n📊 PROCESSING SUMMARY:")
            print(f"   Total Segments: {len(all_segments)}")
            print(f"   Segment Types: {', '.join(segment_summary.keys())}")

        except Exception as e:
            error_msg = f"Error processing {Path(pdf_path).name}: {str(e)}"
            print(f"❌ {error_msg}")
            results['processing_errors'].append(error_msg)

        return results

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single image through the complete pipeline.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing all processing results
        """
        print(f"\n🖼️  Processing Image: {Path(image_path).name}")
        print("=" * 80)

        results = {
            'filename': Path(image_path).name,
            'file_path': image_path,
            'file_type': 'image',
            'pages': {},
            'overall_classification': None,
            'processing_errors': []
        }

        try:
            # Step 1: Extract text from image
            extracted_text = self.ocr_extractor.extract_from_image(image_path)

            if not extracted_text or not extracted_text.strip():
                results['processing_errors'].append("No text extracted from image")
                print("❌ No text extracted from image")
                return results

            print(f"✓ Text extraction successful from image")

            # Process the extracted text (treating image as single page)
            page_name = "image_content"

            print(f"\n📑 IMAGE CONTENT")
            print("-" * 80)

            # Show the extracted text first
            print("🔤 EXTRACTED TEXT:")
            print("-" * 40)
            print(extracted_text)
            print("-" * 40)

            # Step 2: Clean the text
            cleaned_text = self.text_processor.clean_medical_text(extracted_text)

            # Step 3: Segment the text
            segments = self.segmenter.segment_document(cleaned_text)
            print(f"\n🔍 SEGMENTATION ANALYSIS:")
            print(f"Found {len(segments)} segments:")

            # Show each segment
            for i, segment in enumerate(segments, 1):
                print(f"\n  📝 Segment {i}: {segment.segment_type.replace('_', ' ').title()}")
                print(f"     Lines {segment.start_line}-{segment.end_line}")
                print(f"     Content: {segment.content[:150]}{'...' if len(segment.content) > 150 else ''}")

            # Step 4: Classify each segment and the whole image
            print(f"\n🏷️  CLASSIFICATION RESULTS:")

            segment_classifications = []
            for i, segment in enumerate(segments, 1):
                classification = self.classifier.classify_document(segment.content)
                print(
                    f"  Segment {i} ({segment.segment_type}): {classification.document_type} ({classification.confidence:.1%})")
                if classification.keywords_found:
                    print(f"    Keywords: {', '.join(classification.keywords_found[:5])}")

                segment_classifications.append({
                    'segment_type': segment.segment_type,
                    'content_preview': segment.content[:200] + "..." if len(segment.content) > 200 else segment.content,
                    'full_content': segment.content,
                    'classification': {
                        'document_type': classification.document_type,
                        'confidence': classification.confidence,
                        'keywords_found': classification.keywords_found
                    },
                    'start_line': segment.start_line,
                    'end_line': segment.end_line
                })

            # Step 5: Get overall image classification
            overall_classification = self.classifier.classify_document(cleaned_text)
            print(
                f"\n  📄 Overall Image Classification: {overall_classification.document_type.replace('_', ' ').title()}")
            print(f"     Confidence: {overall_classification.confidence:.1%}")
            if overall_classification.secondary_types:
                print(
                    f"     Secondary possibilities: {', '.join([f'{t[0]} ({t[1]:.1%})' for t in overall_classification.secondary_types[:2]])}")

            # Step 6: Extract additional information
            print(f"\n🏥 MEDICAL INFORMATION EXTRACTION:")

            key_value_pairs = {}
            medical_entities = self.classifier.extract_medical_entities(cleaned_text)
            urgency_level, urgency_confidence = self.classifier.get_document_urgency(cleaned_text)

            # Extract key-value pairs from patient info segments
            for segment in segments:
                if segment.segment_type == 'patient_info':
                    kvp = self.segmenter.extract_key_value_pairs(segment)
                    key_value_pairs.update(kvp)

            # Display extracted information
            if key_value_pairs:
                print("  📋 Patient Information:")
                for key, value in key_value_pairs.items():
                    print(f"     {key.title()}: {value}")

            # Show medical entities
            entity_count = sum(len(entities) for entities in medical_entities.values())
            if entity_count > 0:
                print(f"  💊 Medical Entities Found ({entity_count} total):")
                for entity_type, entities in medical_entities.items():
                    if entities:
                        print(f"     {entity_type.title()}: {', '.join(entities[:5])}")
                        if len(entities) > 5:
                            print(f"       ... and {len(entities) - 5} more")

            # Urgency assessment
            print(f"  ⚠️  Urgency Level: {urgency_level.title()} ({urgency_confidence:.1%})")

            # Extract dates and other info
            dates = self.text_processor.extract_dates(cleaned_text)
            if dates:
                print(f"  📅 Dates Found: {', '.join(dates[:3])}")

            # Store results
            results['pages'][page_name] = {
                'raw_text': extracted_text,
                'cleaned_text': cleaned_text,
                'text_length': len(cleaned_text),
                'segments_count': len(segments),
                'segments': segment_classifications,
                'overall_classification': {
                    'document_type': overall_classification.document_type,
                    'confidence': overall_classification.confidence,
                    'secondary_types': overall_classification.secondary_types
                },
                'medical_entities': medical_entities,
                'urgency_assessment': {
                    'level': urgency_level,
                    'confidence': urgency_confidence
                },
                'key_value_pairs': key_value_pairs,
                'dates_found': dates
            }

            results['overall_classification'] = {
                'document_type': overall_classification.document_type,
                'confidence': overall_classification.confidence,
                'secondary_types': overall_classification.secondary_types
            }

            # Summary statistics
            segment_summary = self.segmenter.get_segment_summary(segments)
            results['segment_summary'] = segment_summary
            results['total_segments'] = len(segments)

            print(f"\n🎯 FINAL IMAGE CLASSIFICATION:")
            print(f"   Type: {overall_classification.document_type.replace('_', ' ').title()}")
            print(f"   Confidence: {overall_classification.confidence:.1%}")

            print(f"\n📊 PROCESSING SUMMARY:")
            print(f"   Total Segments: {len(segments)}")
            print(f"   Segment Types: {', '.join(segment_summary.keys())}")

        except Exception as e:
            error_msg = f"Error processing {Path(image_path).name}: {str(e)}"
            print(f"❌ {error_msg}")
            results['processing_errors'].append(error_msg)

        return results

    def process_all_files(self, sample_dir: Path):
        """
        Process all supported files (PDFs and images) in the sample directory.

        Args:
            sample_dir: Path to directory containing files

        Returns:
            List of processing results for each file
        """
        supported_exts = [".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        files = [f for f in sample_dir.iterdir() if f.suffix.lower() in supported_exts]

        if not files:
            print(f"❌ No supported files found in {sample_dir}")
            return []

        print(f"🔍 Found {len(files)} supported file(s) to process:")
        for file in files:
            print(f"   - {file.name}")

        all_results = []
        for i, file in enumerate(files, 1):
            print(f"\n{'=' * 80}")
            print(f"PROCESSING FILE {i}/{len(files)}")
            print(f"{'=' * 80}")

            if file.suffix.lower() == ".pdf":
                result = self.process_pdf(str(file))
            else:
                result = self.process_image(str(file))
            all_results.append(result)

        return all_results

    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save processing results to JSON file."""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Detailed results saved to {output_path}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")

    def print_final_summary(self, results: List[Dict[str, Any]]):
        """Print a final summary of all processing results."""
        print(f"\n{'=' * 80}")
        print("FINAL PROCESSING SUMMARY")
        print(f"{'=' * 80}")

        if not results:
            print("❌ No documents were processed successfully.")
            return

        successful_docs = [r for r in results if not r['processing_errors']]
        failed_docs = [r for r in results if r['processing_errors']]

        print(f"📊 Total Files: {len(results)}")
        print(f"✅ Successfully Processed: {len(successful_docs)}")
        print(f"❌ Failed: {len(failed_docs)}")

        if failed_docs:
            print(f"\n❌ FAILED FILES:")
            for doc in failed_docs:
                file_type = "PDF" if doc.get('file_type') != 'image' else "Image"
                print(f"   - {doc['filename']} ({file_type}): {', '.join(doc['processing_errors'])}")

        if successful_docs:
            print(f"\n✅ SUCCESSFULLY PROCESSED:")

            # Document type summary
            doc_types = {}
            total_segments = 0
            total_entities = 0

            for doc in successful_docs:
                if doc['overall_classification']:
                    doc_type = doc['overall_classification']['document_type']
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

                total_segments += doc.get('total_segments', 0)

                # Count medical entities
                for page_data in doc['pages'].values():
                    entities = page_data.get('medical_entities', {})
                    total_entities += sum(len(ent_list) for ent_list in entities.values())

            print(f"\n📋 Document Types Found:")
            for doc_type, count in doc_types.items():
                print(f"   - {doc_type.replace('_', ' ').title()}: {count}")

            print(f"\n📊 Processing Statistics:")
            print(f"   - Total Segments Identified: {total_segments}")
            print(f"   - Total Medical Entities Found: {total_entities}")

            # Show details for each successful document
            for doc in successful_docs:
                print(f"\n📄 {doc['filename']}:")
                if doc['overall_classification']:
                    cls = doc['overall_classification']
                    print(f"   Type: {cls['document_type'].replace('_', ' ').title()} ({cls['confidence']:.1%})")

                print(f"   Pages: {len(doc['pages'])}")
                print(f"   Segments: {doc.get('total_segments', 0)}")

                # Show segment breakdown
                if doc.get('segment_summary'):
                    segment_types = list(doc['segment_summary'].keys())[:3]  # Show top 3
                    if segment_types:
                        print(f"   Main Sections: {', '.join(s.replace('_', ' ').title() for s in segment_types)}")


def check_dependencies():
    """Check if all required packages are installed."""
    print("🔍 Checking Dependencies...")
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

            print(f"✓ {package}")

        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True


def main():
    """Main function to run the enhanced OCR pipeline."""
    print("🏥 Medical Document AI - Enhanced Processing Pipeline")
    print("=" * 80)

    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before running the pipeline.")
        return

    print()

    # Initialize processor
    try:
        processor = MedicalDocumentProcessor()
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        return

    # Setup directories
    sample_dir = Path("data/sample")
    if not sample_dir.exists():
        print("⚠️  Creating data/sample directory...")
        sample_dir.mkdir(parents=True, exist_ok=True)
        print("✅ Created data/sample directory")
        print("❌ No supported files found to process. Add files to data/sample/ and run again.")
        print("📁 Supported formats: PDF, JPG, JPEG, PNG, BMP, TIFF")
        return

    # Process all files (PDFs and Images)
    results = processor.process_all_files(sample_dir)

    if not results:
        print("❌ No supported files found to process.")
        print("📁 Add PDF, JPG, PNG, or other image files to data/sample/")
        return

    # Save detailed results
    output_file = "data/processing_results.json"
    processor.save_results(results, output_file)

    # Print final summary
    processor.print_final_summary(results)

    print(f"\n🎉 Processing complete! Check {output_file} for detailed results.")


if __name__ == "__main__":
    main()