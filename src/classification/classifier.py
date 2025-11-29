# src/classification/classifier.py

import re
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of document classification."""
    document_type: str
    confidence: float
    secondary_types: List[Tuple[str, float]]
    keywords_found: List[str]


class MedicalDocumentClassifier:
    """
    Classifies medical documents into types.
    Rule-based approach for quick prototype.
    """

    def __init__(self):
        # Define document type keywords and patterns
        self.document_patterns = {
            'discharge_summary': {
                'keywords': [
                    'discharge', 'discharged', 'admission', 'admitted',
                    'hospital stay', 'length of stay', 'discharge date',
                    'admission date', 'discharge diagnosis', 'discharge instructions'
                ],
                'patterns': [
                    r'discharge\s+(?:summary|report|note)',
                    r'admission\s+(?:date|diagnosis)',
                    r'hospital\s+course',
                    r'condition\s+on\s+discharge'
                ]
            },
            'lab_report': {
                'keywords': [
                    'laboratory', 'lab results', 'blood test', 'urine test',
                    'hemoglobin', 'glucose', 'cholesterol', 'creatinine',
                    'normal range', 'abnormal', 'reference range'
                ],
                'patterns': [
                    r'lab\s+(?:results?|report|findings)',
                    r'laboratory\s+(?:results?|report)',
                    r'reference\s+range',
                    r'\d+\.\d+\s*(?:mg/dl|mmol/l|g/dl)'
                ]
            },
            'consultation_note': {
                'keywords': [
                    'consultation', 'referred by', 'consulting physician',
                    'opinion', 'recommendations', 'specialist',
                    'follow up', 'follow-up'
                ],
                'patterns': [
                    r'consultation\s+(?:note|report)',
                    r'referred\s+(?:by|to)',
                    r'consulting\s+physician',
                    r'specialist\s+opinion'
                ]
            },
            'prescription': {
                'keywords': [
                    'prescription', 'medication', 'dosage', 'frequency',
                    'tablets', 'capsules', 'mg', 'ml', 'twice daily',
                    'once daily', 'pharmacy', 'refill'
                ],
                'patterns': [
                    r'(?:rx|prescription)[:.]',
                    r'\d+\s*mg\s+(?:once|twice|three times)\s+daily',
                    r'take\s+\d+\s+(?:tablet|capsule)',
                    r'sig[:.]'
                ]
            },
            'radiology_report': {
                'keywords': [
                    'radiology', 'x-ray', 'ct scan', 'mri', 'ultrasound',
                    'impression', 'findings', 'radiologist',
                    'contrast', 'image', 'scan'
                ],
                'patterns': [
                    r'(?:ct|mri|x-ray|ultrasound)\s+(?:scan|report)',
                    r'radiological\s+(?:findings|impression)',
                    r'contrast\s+(?:agent|medium)',
                    r'image\s+quality'
                ]
            },
            'progress_note': {
                'keywords': [
                    'progress note', 'daily note', 'soap note',
                    'assessment', 'plan', 'subjective', 'objective',
                    'today', 'patient reports', 'continues'
                ],
                'patterns': [
                    r'progress\s+note',
                    r'soap\s+note',
                    r'(?:subjective|objective|assessment|plan)[:.]',
                    r'patient\s+(?:reports|states|complains)'
                ]
            },
            'operative_report': {
                'keywords': [
                    'operative report', 'surgery', 'procedure', 'operation',
                    'anesthesia', 'incision', 'suture', 'surgeon',
                    'postoperative', 'intraoperative'
                ],
                'patterns': [
                    r'operative\s+(?:report|note)',
                    r'surgical\s+procedure',
                    r'(?:pre|post|intra)operative',
                    r'anesthesia\s+type'
                ]
            }
        }

        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for doc_type, data in self.document_patterns.items():
            self.compiled_patterns[doc_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in data['patterns']
            ]

    def classify_document(self, text: str) -> ClassificationResult:
        """
        Classify a medical document.

        Args:
            text: Document text to classify

        Returns:
            ClassificationResult with classification details
        """
        if not text or not text.strip():
            return ClassificationResult(
                document_type='unknown',
                confidence=0.0,
                secondary_types=[],
                keywords_found=[]
            )

        text_lower = text.lower()
        scores = {}
        all_keywords_found = []

        # Score each document type
        for doc_type, data in self.document_patterns.items():
            score = 0
            keywords_found = []

            # Check keywords
            for keyword in data['keywords']:
                if keyword.lower() in text_lower:
                    score += 1
                    keywords_found.append(keyword)

            # Check patterns (give higher weight)
            for pattern in self.compiled_patterns[doc_type]:
                matches = pattern.findall(text)
                score += len(matches) * 2  # Patterns worth more than keywords
                if matches:
                    keywords_found.extend(matches)

            scores[doc_type] = score
            all_keywords_found.extend(keywords_found)

        # Find best match
        if not any(scores.values()):
            return ClassificationResult(
                document_type='general_medical',
                confidence=0.3,
                secondary_types=[],
                keywords_found=[]
            )

        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_type, best_score = sorted_scores[0]

        # Calculate confidence
        total_score = sum(scores.values())
        confidence = best_score / max(total_score, 1) if total_score > 0 else 0

        # Get secondary types
        secondary_types = [(doc_type, score / max(total_score, 1))
                           for doc_type, score in sorted_scores[1:3]
                           if score > 0]

        return ClassificationResult(
            document_type=best_type,
            confidence=min(confidence, 0.95),  # Cap at 95% for rule-based
            secondary_types=secondary_types,
            keywords_found=list(set(all_keywords_found))
        )

    def classify_segments(self, segments: List) -> Dict[str, ClassificationResult]:
        """
        Classify individual document segments.

        Args:
            segments: List of DocumentSegment objects

        Returns:
            Dictionary mapping segment types to classifications
        """
        results = {}

        for segment in segments:
            # Focus classification on segment content
            result = self.classify_document(segment.content)
            results[f"{segment.segment_type}_{segment.start_line}"] = result

        return results

    def get_document_urgency(self, text: str) -> Tuple[str, float]:
        """
        Assess document urgency level.

        Args:
            text: Document text

        Returns:
            Tuple of (urgency_level, confidence)
        """
        urgent_keywords = [
            'emergency', 'urgent', 'stat', 'immediate', 'critical',
            'severe', 'acute', 'emergency room', 'er visit',
            'ambulance', 'code blue', 'trauma'
        ]

        high_keywords = [
            'abnormal', 'concerning', 'suspicious', 'significant',
            'requires attention', 'follow up immediately',
            'contact physician'
        ]

        text_lower = text.lower()

        urgent_count = sum(1 for keyword in urgent_keywords if keyword in text_lower)
        high_count = sum(1 for keyword in high_keywords if keyword in text_lower)

        if urgent_count >= 2:
            return ('urgent', 0.9)
        elif urgent_count >= 1:
            return ('urgent', 0.7)
        elif high_count >= 2:
            return ('high', 0.8)
        elif high_count >= 1:
            return ('high', 0.6)
        else:
            return ('routine', 0.5)

    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract basic medical entities from text.

        Args:
            text: Document text

        Returns:
            Dictionary of entity types and found entities
        """
        entities = {
            'medications': [],
            'conditions': [],
            'procedures': [],
            'measurements': []
        }

        # Medication patterns
        med_patterns = [
            r'\b[A-Z][a-z]+(?:cillin|mycin|pril|sartan|olol|pine|zole|mine)\b',
            r'\b(?:aspirin|ibuprofen|acetaminophen|paracetamol|insulin)\b'
        ]

        for pattern in med_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['medications'].extend(matches)

        # Common conditions
        condition_patterns = [
            r'\b(?:diabetes|hypertension|pneumonia|bronchitis|asthma)\b',
            r'\b(?:fracture|infection|inflammation|tumor|cancer)\b'
        ]

        for pattern in condition_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['conditions'].extend(matches)

        # Procedures
        procedure_patterns = [
            r'\b(?:surgery|biopsy|x-ray|ct scan|mri|ultrasound)\b',
            r'\b(?:blood test|urine test|ecg|ekg|colonoscopy)\b'
        ]

        for pattern in procedure_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['procedures'].extend(matches)

        # Measurements
        measurement_patterns = [
            r'\d+(?:\.\d+)?\s*(?:mg/dl|mmol/l|g/dl|kg|lbs|cm|ft|in)',
            r'\d+/\d+\s*mmHg',  # Blood pressure
            r'\d+(?:\.\d+)?\s*°[FC]'  # Temperature
        ]

        for pattern in measurement_patterns:
            matches = re.findall(pattern, text)
            entities['measurements'].extend(matches)

        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))

        return entities