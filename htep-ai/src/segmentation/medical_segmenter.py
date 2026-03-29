from dataclasses import dataclass
from typing import List


@dataclass
class DocumentSegment:
	"""Structured segment output for downstream NLP/classification."""

	segment_type: str
	content: str
	confidence: float
	start_line: int
	end_line: int


class MedicalDocumentSegmenter:
	"""
	Lightweight medical text segmenter.

	Current strategy:
	- split by non-empty lines
	- classify each line into a coarse segment type
	"""

	SECTION_KEYWORDS = {
		"medication": ["rx", "prescription", "dosage", "tablet", "capsule"],
		"diagnosis": ["diagnosis", "impression", "assessment"],
		"plan": ["plan", "follow up", "follow-up", "recommendation"],
		"vitals": ["bp", "pulse", "temperature", "mmhg"],
	}

	def _infer_segment_type(self, line: str) -> str:
		text = line.lower()

		for segment_type, keywords in self.SECTION_KEYWORDS.items():
			if any(keyword in text for keyword in keywords):
				return segment_type

		return "general"

	def segment_document(self, text: str) -> List[DocumentSegment]:
		if not text or not text.strip():
			return []

		segments: List[DocumentSegment] = []
		lines = [line.strip() for line in text.splitlines() if line.strip()]

		for idx, line in enumerate(lines):
			segment_type = self._infer_segment_type(line)
			segments.append(
				DocumentSegment(
					segment_type=segment_type,
					content=line,
					confidence=0.8,
					start_line=idx,
					end_line=idx,
				)
			)

		return segments
