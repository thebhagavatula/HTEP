# src/nlp/ocr_postprocessor.py
"""
OCR Post-Processor: Drug/Disease dictionary matching with RapidFuzz.

Pipeline:  raw OCR text  →  drug fuzzy match  →  disease fuzzy match
Each stage is independently fallback-safe.
PaddleOCR's native corrections are preserved as-is.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing rapidfuzz — everything degrades gracefully without it
# ---------------------------------------------------------------------------
try:
    from rapidfuzz import fuzz, process as rf_process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    logger.warning("rapidfuzz not installed — fuzzy matching disabled")


# ---------------------------------------------------------------------------
# Fallback seed lists (used only when dictionary files are missing)
# ---------------------------------------------------------------------------

_FALLBACK_DRUGS = [
    "acetaminophen", "amoxicillin", "aspirin", "atorvastatin", "azithromycin",
    "ciprofloxacin", "clopidogrel", "doxycycline", "furosemide", "gabapentin",
    "hydrochlorothiazide", "ibuprofen", "insulin", "levothyroxine", "lisinopril",
    "losartan", "metformin", "metoprolol", "omeprazole", "pantoprazole",
    "paracetamol", "prednisone", "rosuvastatin", "sertraline", "simvastatin",
    "tamsulosin", "tramadol", "trazodone", "venlafaxine", "warfarin",
]

_FALLBACK_DISEASES = [
    "anemia", "anxiety", "arthritis", "asthma", "bronchitis",
    "cancer", "cholesterol", "copd", "depression", "diabetes",
    "epilepsy", "fever", "gastritis", "gerd", "gout",
    "heart failure", "hepatitis", "hypertension", "hypothyroidism", "infection",
    "influenza", "insomnia", "migraine", "obesity", "osteoporosis",
    "pneumonia", "sepsis", "sinusitis", "stroke", "tuberculosis",
]


# ---------------------------------------------------------------------------
# Helper: load terms from a text file (one per line)
# ---------------------------------------------------------------------------

def _load_terms(path: Path, fallback: List[str]) -> List[str]:
    """Load terms from a text file; return fallback on any failure."""
    if not path.exists():
        logger.warning("Dictionary file not found: %s — using fallback list", path)
        return list(fallback)
    try:
        terms: List[str] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                term = line.strip().lower()
                if term and len(term) >= 2:
                    terms.append(term)
        if terms:
            logger.info("Loaded %d terms from %s", len(terms), path)
            return terms
        logger.warning("File %s was empty — using fallback list", path)
        return list(fallback)
    except Exception:
        logger.exception("Failed to load %s — using fallback list", path)
        return list(fallback)


# ---------------------------------------------------------------------------
# OCRPostProcessor
# ---------------------------------------------------------------------------

class OCRPostProcessor:
    """
    Two-stage OCR post-processing pipeline:
      1. Drug name fuzzy matching (RapidFuzz)
      2. Disease name fuzzy matching (RapidFuzz)

    PaddleOCR's native output is kept as-is — no character-level corrections.
    Every stage is independently fallback-safe.
    """

    def __init__(
        self,
        drug_dict_path: Optional[Path] = None,
        disease_dict_path: Optional[Path] = None,
        drug_threshold: int = 85,
        disease_threshold: int = 85,
    ):
        # Import config defaults lazily so the module can be tested standalone
        try:
            from src.config import (
                DRUG_DICT_PATH, DISEASE_DICT_PATH,
                DRUG_FUZZY_THRESHOLD, DISEASE_FUZZY_THRESHOLD,
            )
        except ImportError:
            DRUG_DICT_PATH = Path("data/dictionaries/drugs.txt")
            DISEASE_DICT_PATH = Path("data/dictionaries/diseases.txt")
            DRUG_FUZZY_THRESHOLD = 85
            DISEASE_FUZZY_THRESHOLD = 85

        self.drug_threshold = drug_threshold or DRUG_FUZZY_THRESHOLD
        self.disease_threshold = disease_threshold or DISEASE_FUZZY_THRESHOLD

        # Load dictionaries
        self._drugs = _load_terms(
            drug_dict_path or DRUG_DICT_PATH, _FALLBACK_DRUGS
        )
        self._diseases = _load_terms(
            disease_dict_path or DISEASE_DICT_PATH, _FALLBACK_DISEASES
        )

        # Build lookup sets for exact matching (fast path)
        self._drug_set = set(d.lower() for d in self._drugs)
        self._disease_set = set(d.lower() for d in self._diseases)

        # Separate single-word and multi-word entries for n-gram matching
        self._drugs_single = [d for d in self._drugs if " " not in d]
        self._drugs_multi = [d for d in self._drugs if " " in d]
        self._diseases_single = [d for d in self._diseases if " " not in d]
        self._diseases_multi = [d for d in self._diseases if " " in d]

        logger.info(
            "OCRPostProcessor ready: %d drugs, %d diseases, rapidfuzz=%s",
            len(self._drugs), len(self._diseases), RAPIDFUZZ_AVAILABLE,
        )

    @property
    def drugs(self) -> List[str]:
        return self._drugs

    @property
    def diseases(self) -> List[str]:
        return self._diseases

    # ------------------------------------------------------------------
    # Fuzzy Dictionary Matching
    # ------------------------------------------------------------------

    def _fuzzy_match_single(
        self, word: str, dictionary: List[str], threshold: int
    ) -> Optional[Tuple[str, int]]:
        """Try to fuzzy-match a single word against a dictionary."""
        if not RAPIDFUZZ_AVAILABLE or not dictionary:
            return None
        if len(word) < 3:
            return None

        result = rf_process.extractOne(
            word.lower(), dictionary, scorer=fuzz.ratio, score_cutoff=threshold
        )
        if result:
            match, score, _ = result
            return match, int(score)
        return None

    def _find_multi_word_matches(
        self, text: str, multi_dict: List[str], threshold: int
    ) -> List[Tuple[str, str, int]]:
        """Find multi-word dictionary entries using n-gram sliding window."""
        if not RAPIDFUZZ_AVAILABLE or not multi_dict:
            return []

        matches: List[Tuple[str, str, int]] = []
        words = text.lower().split()

        for entry in multi_dict:
            n = len(entry.split())
            if n > len(words):
                continue
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i + n])
                score = fuzz.ratio(ngram, entry)
                if score >= threshold:
                    matches.append((ngram, entry, int(score)))
                    break  # one match per entry is enough

        return matches

    def _correct_with_dictionary(
        self,
        text: str,
        single_dict: List[str],
        multi_dict: List[str],
        exact_set: set,
        threshold: int,
        label: str,
    ) -> Tuple[str, List[Dict], List[str]]:
        """
        Run fuzzy matching against a dictionary.
        Returns (corrected_text, corrections, matched_terms).
        """
        corrections: List[Dict] = []
        matched: List[str] = []

        if not RAPIDFUZZ_AVAILABLE:
            # Without rapidfuzz, do exact-match detection only
            text_lower = text.lower()
            for term in sorted(exact_set):
                # Match whole word for single-word terms
                if " " not in term:
                    if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                        matched.append(term)
                else:
                    if term in text_lower:
                        matched.append(term)
            return text, corrections, matched

        try:
            # --- Single-word fuzzy correction ---
            words = text.split()
            new_words = []
            for word in words:
                clean = re.sub(r'[^A-Za-z]', '', word).lower()
                if len(clean) < 3:
                    new_words.append(word)
                    continue

                # Exact match first (fast path)
                if clean in exact_set:
                    matched.append(clean)
                    new_words.append(word)
                    continue

                # Fuzzy match
                result = self._fuzzy_match_single(clean, single_dict, threshold)
                if result:
                    match_term, score = result
                    if match_term != clean:
                        # Preserve punctuation around the word
                        prefix = ""
                        suffix = ""
                        stripped = word
                        while stripped and not stripped[0].isalpha():
                            prefix += stripped[0]
                            stripped = stripped[1:]
                        while stripped and not stripped[-1].isalpha():
                            suffix = stripped[-1] + suffix
                            stripped = stripped[:-1]

                        # Preserve case
                        if stripped.isupper():
                            corrected = match_term.upper()
                        elif stripped and stripped[0].isupper():
                            corrected = match_term.capitalize()
                        else:
                            corrected = match_term

                        full_corrected = prefix + corrected + suffix
                        corrections.append({
                            "from": word, "to": full_corrected,
                            "type": label, "score": score
                        })
                        new_words.append(full_corrected)
                        matched.append(match_term)
                    else:
                        matched.append(match_term)
                        new_words.append(word)
                else:
                    new_words.append(word)

            text = " ".join(new_words)

            # --- Multi-word fuzzy detection ---
            multi_matches = self._find_multi_word_matches(
                text, multi_dict, threshold
            )
            for _ngram, entry, _score in multi_matches:
                if entry not in matched:
                    matched.append(entry)

        except Exception:
            logger.exception("%s fuzzy matching failed — passing text through", label)

        return text, corrections, sorted(set(matched))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, text: str) -> Dict:
        """
        Run the full post-processing pipeline.

        Returns:
            {
                'corrected_text': str,
                'corrections': [{'from': str, 'to': str, 'type': str, ...}],
                'matched_drugs': [str],
                'matched_diseases': [str],
            }
        """
        if not text or not text.strip():
            return {
                "corrected_text": "",
                "corrections": [],
                "matched_drugs": [],
                "matched_diseases": [],
            }

        all_corrections: List[Dict] = []

        # Stage 1: Drug dictionary
        matched_drugs: List[str] = []
        try:
            text, drug_corrections, matched_drugs = self._correct_with_dictionary(
                text,
                self._drugs_single, self._drugs_multi,
                self._drug_set, self.drug_threshold, "drug"
            )
            all_corrections.extend(drug_corrections)
        except Exception:
            logger.exception("Stage 1 (drug matching) failed — continuing")

        # Stage 2: Disease dictionary
        matched_diseases: List[str] = []
        try:
            text, disease_corrections, matched_diseases = self._correct_with_dictionary(
                text,
                self._diseases_single, self._diseases_multi,
                self._disease_set, self.disease_threshold, "disease"
            )
            all_corrections.extend(disease_corrections)
        except Exception:
            logger.exception("Stage 2 (disease matching) failed — continuing")

        return {
            "corrected_text": text,
            "corrections": all_corrections,
            "matched_drugs": matched_drugs,
            "matched_diseases": matched_diseases,
        }
