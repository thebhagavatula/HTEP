import difflib
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import spacy
    from spacy.matcher import PhraseMatcher
except ImportError:  # pragma: no cover - handled by fallback path
    spacy = None
    PhraseMatcher = None


logger = logging.getLogger(__name__)


DEFAULT_MEDICAL_DICTIONARY = [
    "acetaminophen",
    "admission",
    "allergy",
    "amoxicillin",
    "antibiotic",
    "aspirin",
    "assessment",
    "asthma",
    "blood",
    "bronchitis",
    "capsule",
    "cholesterol",
    "clinic",
    "condition",
    "consultation",
    "creatinine",
    "diagnosis",
    "diabetes",
    "discharge",
    "dosage",
    "emergency",
    "fever",
    "follow",
    "frequency",
    "glucose",
    "hemoglobin",
    "hospital",
    "hypertension",
    "ibuprofen",
    "impression",
    "infection",
    "insulin",
    "instructions",
    "laboratory",
    "medication",
    "metformin",
    "milligram",
    "note",
    "objective",
    "paracetamol",
    "patient",
    "pharmacy",
    "physician",
    "plan",
    "pneumonia",
    "prescription",
    "procedure",
    "progress",
    "radiology",
    "refill",
    "report",
    "results",
    "routine",
    "severe",
    "soap",
    "specialist",
    "subjective",
    "summary",
    "surgery",
    "symptoms",
    "tablet",
    "treatment",
    "ultrasound",
    "urgent",
    "urine",
    "visit",
    "xray",
]


DEFAULT_COMMON_ENGLISH_DICTIONARY = [
    "a", "about", "after", "all", "also", "and", "any", "are", "as", "at",
    "be", "because", "but", "by", "can", "come", "day", "did", "do", "for",
    "from", "good", "has", "have", "he", "hello", "her", "him", "his", "how",
    "i", "if", "in", "is", "it", "its", "just", "know", "like", "man",
    "me", "more", "my", "name", "new", "no", "not", "now", "of", "on",
    "one", "or", "our", "out", "people", "say", "she", "so", "some", "test",
    "that", "the", "their", "them", "there", "they", "this", "time", "to", "up",
    "use", "was", "we", "well", "what", "when", "which", "who", "will", "with",
    "word", "work", "world", "would", "you", "your",
]


DEFAULT_ENGLISH_WORDS_PATH = Path("data/dictionaries/english_words_alpha.txt")


class BlockTextParser:
    """
    Dictionary-assisted parser for block handwritten text.
    Uses SpaCy tokenization and optional sciSpaCy NER when available.
    """

    def __init__(
        self,
        dictionary_terms: Optional[List[str]] = None,
        english_terms: Optional[List[str]] = None,
        english_words_path: Optional[str] = None,
        enable_english_layer: bool = True,
        similarity_cutoff: float = 0.84,
        english_similarity_cutoff: float = 0.93,
    ):
        self.similarity_cutoff = similarity_cutoff
        self.english_similarity_cutoff = english_similarity_cutoff
        self.enable_english_layer = enable_english_layer

        self._medical_dictionary = sorted(set(dictionary_terms or DEFAULT_MEDICAL_DICTIONARY))
        self._medical_lookup = {
            term.lower(): term for term in self._medical_dictionary
        }
        self._medical_lower = sorted(self._medical_lookup.keys())

        self._english_dictionary = self._load_english_dictionary(
            english_terms=english_terms,
            english_words_path=english_words_path,
            enable_english_layer=enable_english_layer,
        )
        self._english_lookup = {
            term.lower(): term for term in self._english_dictionary
        }
        self._english_lower = sorted(self._english_lookup.keys())

        all_terms = sorted(set(self._medical_dictionary) | set(self._english_dictionary))
        self._dictionary = all_terms
        self._dictionary_lookup = {
            term.lower(): term for term in self._dictionary
        }
        self._dictionary_lower = sorted(self._dictionary_lookup.keys())

        self.backend = "regex"
        self.nlp = None
        self.sci_nlp = None
        self.matcher = None

        self._init_models()

    @staticmethod
    def _load_terms_from_file(file_path: Path) -> List[str]:
        terms: List[str] = []

        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                word = line.strip().lower()
                if len(word) < 2:
                    continue
                if not word.isalpha():
                    continue
                terms.append(word)

        return terms

    def _load_english_dictionary(
        self,
        english_terms: Optional[List[str]],
        english_words_path: Optional[str],
        enable_english_layer: bool,
    ) -> List[str]:
        if not enable_english_layer:
            return []

        if english_terms:
            return sorted(set(term.lower() for term in english_terms if term and term.strip()))

        path = Path(english_words_path) if english_words_path else DEFAULT_ENGLISH_WORDS_PATH

        if path.exists():
            try:
                loaded = self._load_terms_from_file(path)
                if loaded:
                    logger.info("Loaded %s English terms from %s", len(loaded), path)
                    return sorted(set(loaded))
            except Exception:
                logger.exception("Failed to load English word list from %s", path)

        logger.warning(
            "English word file not found at %s; using fallback common-word list",
            path,
        )
        return sorted(set(DEFAULT_COMMON_ENGLISH_DICTIONARY))

    def _init_models(self) -> None:
        if spacy is None:
            logger.warning("SpaCy not installed; using regex fallback parser")
            return

        # Generic SpaCy model for tokenization + sentence boundaries.
        self.nlp = self._try_load_model(["en_core_web_sm", "en_core_web_md"])
        if self.nlp is None:
            self.nlp = spacy.blank("en")
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")

        self.backend = "spacy"

        # Optional sciSpaCy model for clinical/scientific entities.
        self.sci_nlp = self._try_load_model(["en_core_sci_sm", "en_ner_bc5cdr_md"])
        if self.sci_nlp is not None:
            self.backend = "spacy+scispacy"

        self._init_phrase_matcher()

    @staticmethod
    def _try_load_model(model_names: List[str]):
        for model_name in model_names:
            try:
                return spacy.load(model_name)
            except Exception:
                continue
        return None

    def _init_phrase_matcher(self) -> None:
        if PhraseMatcher is None or self.nlp is None:
            return

        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        medical_patterns = [self.nlp.make_doc(term) for term in self._medical_dictionary]
        english_patterns = [self.nlp.make_doc(term) for term in self._english_dictionary]

        if medical_patterns:
            self.matcher.add("MEDICAL_DICTIONARY", medical_patterns)
        if english_patterns:
            self.matcher.add("ENGLISH_DICTIONARY", english_patterns)

    @staticmethod
    def _normalize_ocr_confusions(token: str) -> str:
        table = str.maketrans({
            "0": "o",
            "1": "l",
            "3": "e",
            "5": "s",
            "6": "g",
            "8": "b",
        })
        return token.translate(table)

    @staticmethod
    def _apply_case(source: str, target: str) -> str:
        if source.isupper():
            return target.upper()
        if source[:1].isupper() and source[1:].islower():
            return target.capitalize()
        if source.islower():
            return target.lower()
        return target

    @staticmethod
    def _is_word_token(token: str) -> bool:
        return len(token) >= 3 and bool(re.search(r"[A-Za-z]", token))

    @staticmethod
    def _looks_mergeable(left: str, right: str) -> bool:
        return (
            bool(left)
            and bool(right)
            and bool(re.search(r"\d", left))
            and len(right) == 1
            and right.isalpha()
        )

    def _correct_token(self, token: str) -> Tuple[str, Optional[str]]:
        if not self._is_word_token(token):
            return token, None

        original = token
        normalized = self._normalize_ocr_confusions(token)

        cleaned = re.sub(r"[^A-Za-z]", "", normalized).lower()
        if not cleaned:
            return original, None

        exact_medical = self._medical_lookup.get(cleaned)
        if exact_medical:
            return self._apply_case(original, exact_medical), "medical"

        exact_english = self._english_lookup.get(cleaned)
        if exact_english:
            return self._apply_case(original, exact_english), "english"

        close_medical = difflib.get_close_matches(
            cleaned,
            self._medical_lower,
            n=1,
            cutoff=self.similarity_cutoff,
        )
        if close_medical:
            corrected = self._medical_lookup[close_medical[0]]
            return self._apply_case(original, corrected), "medical"

        close_english = difflib.get_close_matches(
            cleaned,
            self._english_lower,
            n=1,
            cutoff=self.english_similarity_cutoff,
        )
        if close_english:
            corrected = self._english_lookup[close_english[0]]
            return self._apply_case(original, corrected), "english"

        return original, None

    def _correct_with_spacy(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        doc = self.nlp(text)
        chunks: List[str] = []
        corrections: List[Dict[str, str]] = []

        idx = 0
        while idx < len(doc):
            token = doc[idx]

            # Handle split OCR tokens like "te5" + "t" => "te5t".
            if idx + 1 < len(doc):
                next_token = doc[idx + 1]
                if token.whitespace_ == "" and self._looks_mergeable(token.text, next_token.text):
                    merged_text = token.text + next_token.text
                    corrected, source = self._correct_token(merged_text)
                    if corrected != merged_text:
                        correction = {"from": merged_text, "to": corrected}
                        if source:
                            correction["source"] = source
                        corrections.append(correction)
                    chunks.append(corrected + next_token.whitespace_)
                    idx += 2
                    continue

            corrected, source = self._correct_token(token.text)
            if corrected != token.text:
                correction = {"from": token.text, "to": corrected}
                if source:
                    correction["source"] = source
                corrections.append(correction)
            chunks.append(corrected + token.whitespace_)
            idx += 1

        return "".join(chunks), corrections

    def _correct_with_regex(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        parts = re.findall(r"\s+|[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?|[^\w\s]", text)
        corrected_parts: List[str] = []
        corrections: List[Dict[str, str]] = []

        for part in parts:
            corrected, source = self._correct_token(part)
            if corrected != part:
                correction = {"from": part, "to": corrected}
                if source:
                    correction["source"] = source
                corrections.append(correction)
            corrected_parts.append(corrected)

        return "".join(corrected_parts), corrections

    def _extract_scispacy_entities(self, text: str) -> List[str]:
        if self.sci_nlp is None:
            return []

        try:
            doc = self.sci_nlp(text)
            return sorted(set(ent.text for ent in doc.ents if ent.text.strip()))
        except Exception:
            logger.exception("sciSpaCy entity extraction failed")
            return []

    def _find_dictionary_matches(self, text: str) -> List[str]:
        if not text.strip():
            return []

        if self.matcher is not None and self.nlp is not None:
            doc = self.nlp(text)
            matches = self.matcher(doc)
            found = sorted(set(doc[start:end].text for _, start, end in matches))
            if found:
                return found

        # Fallback term match if matcher is unavailable.
        text_lower = text.lower()
        return sorted(set(term for term in self._dictionary if term.lower() in text_lower))

    def _find_layered_matches(self, text: str) -> Dict[str, List[str]]:
        if not text.strip():
            return {"medical": [], "english": []}

        if self.matcher is not None and self.nlp is not None:
            doc = self.nlp(text)
            medical: List[str] = []
            english: List[str] = []

            for match_id, start, end in self.matcher(doc):
                label = self.nlp.vocab.strings[match_id]
                span_text = doc[start:end].text

                if label == "MEDICAL_DICTIONARY":
                    medical.append(span_text)
                elif label == "ENGLISH_DICTIONARY":
                    english.append(span_text)

            return {
                "medical": sorted(set(medical)),
                "english": sorted(set(english)),
            }

        text_lower = text.lower()

        token_set = set(re.findall(r"[A-Za-z]+", text_lower))

        medical = sorted(
            set(term for term in self._medical_dictionary if term.lower() in token_set)
        )
        english = sorted(
            set(term for term in self._english_dictionary if term.lower() in token_set)
        )

        return {"medical": medical, "english": english}

    def parse(self, text: str) -> Dict:
        """
        Parse and normalize block text.

        Returns:
            {
              corrected_text: str,
              corrections: [{from, to}],
              dictionary_matches: [str],
                            dictionary_layers: {medical: [str], english: [str]},
              entities: [str],
              backend: str
            }
        """
        if not text or not text.strip():
            return {
                "corrected_text": "",
                "corrections": [],
                "dictionary_matches": [],
                "dictionary_layers": {"medical": [], "english": []},
                "entities": [],
                "backend": self.backend,
            }

        if self.nlp is not None:
            corrected_text, corrections = self._correct_with_spacy(text)
        else:
            corrected_text, corrections = self._correct_with_regex(text)

        return {
            "corrected_text": corrected_text,
            "corrections": corrections,
            "dictionary_matches": self._find_dictionary_matches(corrected_text),
            "dictionary_layers": self._find_layered_matches(corrected_text),
            "entities": self._extract_scispacy_entities(corrected_text),
            "backend": self.backend,
        }
