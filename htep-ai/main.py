# main.py - Main pipeline for medical document processing

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ocr.extractor import OCRExtractor
from segmentation.segmenter import MedicalDocumentSegmenter
from classification.classifier import MedicalDocumentClassifier


class MedicalDocumentPipeline:
    """Main pipeline for processing medical documents."""

    def __init__(self):
        self.ocr = OCRExtractor()
        self.segmenter = MedicalDocumentSegmenter()
        self.classifier = MedicalDocumentClassifier()

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single medical document through the complete pipeline.

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary with complete processing results
        """
        start_time = time.time()

        print(f"Processing: {file_path}")
        print("-" * 50)

        results = {
            'file_path': file_path,
            'processing_time': 0,
            'ocr_results': {},
            'segments': [],
            'classification': {},
            'summary': {}
        }

        try:
            # Step 1: OCR Extraction
            print("1. Extracting text with OCR...")
            if file_path.lower().endswith('.pdf'):
                ocr_results = self.ocr.extract_from_pdf(file_path)
            else:
                # Assume image file
                text = self.ocr.extract_from_image(file_path)
                ocr_results = {'page_1': text}

            results['ocr_results'] = ocr_results

            # Combine all pages
            full_text = '\n\n'.join(ocr_results.values())
            print(f"   Extracted {len(full_text)} characters from {len(ocr_results)} page(s)")

            # Step 2: Document Segmentation
            print("2. Segmenting document...")
            segments = self.segmenter.segment_document(full_text)
            results['segments'] = [
                {
                    'type': seg.segment_type,
                    'content': seg.content,
                    'confidence': seg.confidence,
                    'start_line': seg.start_line,
                    'end_line': seg.end_line
                }
                for seg in segments
            ]

            segment_summary = self.segmenter.get_segment_summary(segments)
            print(f"   Found {len(segments)} segments: {segment_summary}")

            # Step 3: Document Classification
            print("3. Classifying document...")
            classification = self.classifier.classify_document(full_text)
            results['classification'] = {
                'document_type': classification.document_type,
                'confidence': classification.confidence,
                'secondary_types': classification.secondary_types,
                'keywords_found': classification.keywords_found
            }

            print(f"   Classified as: {classification.document_type} "
                  f"(confidence: {classification.confidence:.2f})")

            # Step 4: Extract additional insights
            print("4. Extracting medical entities...")
            entities = self.classifier.extract_medical_entities(full_text)
            urgency_level, urgency_confidence = self.classifier.get_document_urgency(full_text)

            results['summary'] = {
                'total_segments': len(segments),
                'segment_types': segment_summary,
                'document_type': classification.document_type,
                'classification_confidence': classification.confidence,
                'urgency_level': urgency_level,
                'urgency_confidence': urgency_confidence,
                'medical_entities': entities,
                'character_count': len(full_text),
                'word_count': len(full_text.split())
            }

            processing_time = time.time() - start_time
            results['processing_time'] = processing_time

            print(f"Processing completed in {processing_time:.2f} seconds")
            print("\nSUMMARY:")
            print(f"  Document Type: {classification.document_type}")
            print(f"  Segments: {len(segments)}")
            print(f"  Urgency: {urgency_level}")
            print(f"  Entities found: {sum(len(v) for v in entities.values())}")

        except Exception as e:
            print(f"✗ Error processing document: {str(e)}")
            results['error'] = str(e)

        return results

    def process_batch(self, input_dir: str, output_dir: str = None) -> List[Dict[str, Any]]:
        """
        Process multiple documents in a directory.

        Args:
            input_dir: Directory containing documents
            output_dir: Directory to save results (optional)

        Returns:
            List of processing results
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Find all PDF and image files
        file_patterns = ['*.pdf', '*.jpg', '*.jpeg', '*.png', '*.tiff']
        files = []
        for pattern in file_patterns:
            files.extend(input_path.glob(pattern))

        if not files:
            print(f"No files found in {input_dir}")
            return []

        print(f"Found {len(files)} files to process")
        print("=" * 60)

        results = []
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Processing: {file_path.name}")
            result = self.process_document(str(file_path))
            results.append(result)

        # Save results if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            results_file = output_path / 'processing_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            print(f"\nResults saved to: {results_file}")

        return results

    def generate_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate a summary report from processing results."""
        if not results:
            return "No results to report."

        total_docs = len(results)
        successful_docs = len([r for r in results if 'error' not in r])

        # Document type distribution
        doc_types = {}
        for result in results:
            if 'error' not in result:
                doc_type = result['summary']['document_type']
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

        # Average processing time
        processing_times = [r['processing_time'] for r in results if 'processing_time' in r]
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0

        report = f"""
MEDICAL DOCUMENT PROCESSING REPORT
{"=" * 50}

OVERVIEW:
  Total Documents: {total_docs}
  Successfully Processed: {successful_docs}
  Failed: {total_docs - successful_docs}
  Average Processing Time: {avg_time:.2f} seconds

DOCUMENT TYPES:
"""
        for doc_type, count in sorted(doc_types.items()):
            percentage = (count / successful_docs) * 100 if successful_docs > 0 else 0
            report += f"  {doc_type}: {count} ({percentage:.1f}%)\n"

        return report


def main():
    """Main function to run the pipeline."""
    print("Medical Document AI System")
    print("=" * 50)

    # Initialize pipeline
    pipeline = MedicalDocumentPipeline()

    # Check if sample data exists
    sample_dir = Path("data/sample")
    if not sample_dir.exists():
        print("Creating sample data directory...")
        sample_dir.mkdir(parents=True, exist_ok=True)
        print("Please add some PDF files to data/sample/ and run again.")
        return

    # Find files to process
    file_patterns = ['*.pdf', '*.jpg', '*.jpeg', '*.png']
    files = []
    for pattern in file_patterns:
        files.extend(sample_dir.glob(pattern))

    if not files:
        print("No files found in data/sample/")
        print("Please add some PDF or image files and try again.")
        return

    # Process files
    print(f"Found {len(files)} file(s) to process\n")

    # Process all files
    results = pipeline.process_batch(str(sample_dir), "data/processed")

    # Generate and display report
    report = pipeline.generate_report(results)
    print(report)

    # Save individual results
    print("\nDETAILED RESULTS:")
    print("-" * 50)
    for result in results:
        if 'error' not in result:
            filename = Path(result['file_path']).name
            print(f"\n{filename}:")
            print(f"  Type: {result['classification']['document_type']}")
            print(f"  Confidence: {result['classification']['confidence']:.2f}")
            print(f"  Segments: {result['summary']['total_segments']}")
            print(f"  Urgency: {result['summary']['urgency_level']}")

            # Show found entities
            entities = result['summary']['medical_entities']
            total_entities = sum(len(v) for v in entities.values())
            if total_entities > 0:
                print(f"  Entities: {total_entities} found")
                for entity_type, items in entities.items():
                    if items:
                        print(f"    {entity_type}: {', '.join(items[:3])}")
        else:
            filename = Path(result['file_path']).name
            print(f"\n{filename}: ERROR - {result['error']}")


def demo_single_file():
    """Demo function to process a single file interactively."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py <file_path>")
        print("Or just run: python main.py (to process all files in data/sample/)")
        return

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    pipeline = MedicalDocumentPipeline()
    result = pipeline.process_document(file_path)

    # Pretty print the results
    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)

    if 'error' in result:
        print(f"Error: {result['error']}")
        return

    print(f"File: {Path(result['file_path']).name}")
    print(f"Processing Time: {result['processing_time']:.2f} seconds")
    print(f"Document Type: {result['classification']['document_type']}")
    print(f"Confidence: {result['classification']['confidence']:.2f}")

    print(f"\nSegments Found ({result['summary']['total_segments']}):")
    for i, segment in enumerate(result['segments'][:5], 1):  # Show first 5
        content_preview = segment['content'][:100] + "..." if len(segment['content']) > 100 else segment['content']
        print(f"  {i}. {segment['type']}: {content_preview}")

    if len(result['segments']) > 5:
        print(f"  ... and {len(result['segments']) - 5} more segments")

    print(f"\nMedical Entities:")
    entities = result['summary']['medical_entities']
    for entity_type, items in entities.items():
        if items:
            print(f"  {entity_type.title()}: {', '.join(items)}")

    print(f"\nUrgency Level: {result['summary']['urgency_level']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Single file mode
        demo_single_file()
    else:
        # Batch processing mode
        main()