# src/segmentation/segmenter.py

import re
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DocumentSegment:
    """Represents a segment of a medical document."""
    segment_type: str
    content: str
    confidence: float
    start_line: int
    end_line: int


class MedicalDocumentSegmenter:
    """
    Segments medical documents into logical sections.
    Uses rule-based approach for quick prototype.
    """

    def __init__(self):
        # Define common medical document patterns
        self.section_patterns = {
            'patient_info': [
                r'patient\s+(?:information|details?|data)',
                r'(?:name|patient):\s*\w+',
                r'(?:dob|date\s+of\s+birth|birth\s+date)',
                r'(?:age|gender|sex):\s*\w+',
                r'(?:address|phone|contact)'
            ],
            'chief_complaint': [
                r'chief\s+complaint',
                r'presenting\s+complaint',
                r'reason\s+for\s+(?:visit|consultation)',
                r'cc:'
            ],
            'history': [
                r'history\s+of\s+present\s+illness',
                r'medical\s+history',
                r'past\s+(?:medical\s+)?history',
                r'hpi:',
                r'pmh:'
            ],
            'examination': [
                r'physical\s+examination',
                r'clinical\s+examination',
                r'(?:general\s+)?examination',
                r'vital\s+signs',
                r'pe:'
            ],
            'diagnosis': [
                r'(?:primary\s+)?diagnosis',
                r'(?:final\s+)?impression',
                r'assessment',
                r'dx:'
            ],
            'medications': [
                r'medications?',
                r'prescriptions?',
                r'drugs?',
                r'rx:',
                r'current\s+medications'
            ],
            'lab_results': [
                r'lab\s+(?:results?|findings?)',
                r'laboratory\s+(?:results?|findings?)',
                r'blood\s+work',
                r'test\s+results?'
            ],
            'treatment_plan': [
                r'treatment\s+plan',
                r'plan\s+of\s+care',
                r'recommendations?',
                r'follow[- ]?up'
            ]
        }

        # Compile regex patterns for efficiency
        self.compiled_patterns = {}
        for section_type, patterns in self.section_patterns.items():
            self.compiled_patterns[section_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def segment_document(self, text: str) -> List[DocumentSegment]:
        """
        Segment document text into logical sections.

        Args:
            text: Raw document text

        Returns:
            List of DocumentSegment objects
        """
        if not text or not text.strip():
            return []

        lines = text.split('\n')
        segments = []
        current_segment = None

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Check if this line starts a new section
            detected_section = self._detect_section_start(line)

            if detected_section:
                # Save previous segment if exists
                if current_segment:
                    current_segment.end_line = i - 1
                    segments.append(current_segment)

                # Start new segment
                current_segment = DocumentSegment(
                    segment_type=detected_section,
                    content=line,
                    confidence=0.8,  # Rule-based confidence
                    start_line=i,
                    end_line=i
                )
            elif current_segment:
                # Add to current segment
                current_segment.content += '\n' + line
                current_segment.end_line = i

        # Don't forget the last segment
        if current_segment:
            segments.append(current_segment)

        # If no segments detected, treat as general content
        if not segments and lines:
            segments.append(DocumentSegment(
                segment_type='general_content',
                content=text,
                confidence=0.5,
                start_line=0,
                end_line=len(lines) - 1
            ))

        logger.info(f"Detected {len(segments)} segments")
        return segments

    def _detect_section_start(self, line: str) -> str:
        """
        Detect if a line starts a new section.

        Args:
            line: Text line to analyze

        Returns:
            Section type if detected, None otherwise
        """
        line_lower = line.lower().strip()

        # Check each section type
        for section_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(line_lower):
                    return section_type

        return None

    def extract_key_value_pairs(self, segment: DocumentSegment) -> Dict[str, str]:
        """
        Extract key-value pairs from a segment.
        Useful for patient info, vital signs, etc.

        Args:
            segment: DocumentSegment to analyze

        Returns:
            Dictionary of key-value pairs
        """
        pairs = {}
        lines = segment.content.split('\n')

        for line in lines:
            # Pattern: "Key: Value"
            colon_match = re.search(r'([^:]+):\s*(.+)', line.strip())
            if colon_match:
                key = colon_match.group(1).strip().lower()
                value = colon_match.group(2).strip()
                pairs[key] = value
                continue

            # Pattern: "Key Value" (common in forms)
            # Look for specific patterns like "Age 45", "Weight 70kg"
            patterns = [
                (r'age\s+(\d+)', 'age'),
                (r'weight\s+(\d+(?:\.\d+)?)\s*(?:kg|lbs?)?', 'weight'),
                (r'height\s+(\d+(?:\.\d+)?)\s*(?:cm|ft|in)?', 'height'),
                (r'temperature\s+(\d+(?:\.\d+)?)\s*(?:°?[fc])?', 'temperature'),
                (r'bp\s+(\d+/\d+)', 'blood_pressure'),
                (r'pulse\s+(\d+)', 'pulse')
            ]

            for pattern, key in patterns:
                match = re.search(pattern, line.lower())
                if match:
                    pairs[key] = match.group(1)

        return pairs

    def get_segment_summary(self, segments: List[DocumentSegment]) -> Dict[str, int]:
        """
        Get summary of detected segments.

        Args:
            segments: List of DocumentSegment objects

        Returns:
            Dictionary with segment types and counts
        """
        summary = {}
        for segment in segments:
            if segment.segment_type in summary:
                summary[segment.segment_type] += 1
            else:
                summary[segment.segment_type] = 1

        return summary

    def filter_segments_by_type(self, segments: List[DocumentSegment],
                                segment_types: List[str]) -> List[DocumentSegment]:
        """
        Filter segments by type.

        Args:
            segments: List of all segments
            segment_types: Types to keep

        Returns:
            Filtered list of segments
        """
        return [seg for seg in segments if seg.segment_type in segment_types]


class SimpleTextProcessor:
    """Helper class for basic text processing tasks."""

    @staticmethod
    def clean_medical_text(text: str) -> str:
        """Clean medical text for better processing."""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Fix common OCR errors in medical text
        fixes = [
            (r'\bO\b', '0'),  # Letter O -> number 0
            (r'\bl\b', '1'),  # Letter l -> number 1
            (r'(\d)\s+(\d)', r'\1\2'),  # Fix split numbers
        ]

        for pattern, replacement in fixes:
            text = re.sub(pattern, replacement, text)

        return text.strip()

    @staticmethod
    def extract_dates(text: str) -> List[str]:
        """Extract dates from text."""
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY
            r'\d{1,2}-\d{1,2}-\d{2,4}',  # MM-DD-YYYY
            r'[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}',  # Month DD, YYYY
        ]

        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)

        return dates

    @staticmethod
    def extract_medications(text: str) -> List[str]:
        """Extract medication names (basic approach)."""
        # Look for common medication patterns
        med_patterns = [
            r'\b[A-Z][a-z]+(?:cillin|mycin|pril|sartan|olol|pine|zole)\b',
            r'\b(?:aspirin|ibuprofen|acetaminophen|paracetamol)\b',
        ]

        medications = []
        for pattern in med_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            medications.extend(matches)

        return list(set(medications))  # Remove duplicates