import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


class LexiconBeamDecoder:
    """
    Beam-search decoder with lexicon-aware reranking.

    Input per character position:
      [{"character": "A", "confidence": 0.82}, ...]
    """

    def __init__(
        self,
        lexicon_terms: Iterable[str],
        primary_terms: Optional[Iterable[str]] = None,
        max_edit_distance: int = 2,
        replacement_confidence_threshold: float = 0.80,
        replacement_min_char_confidence_threshold: float = 0.60,
        non_primary_replacement_min_char_confidence: float = 0.45,
    ):
        self.max_edit_distance = max_edit_distance
        self.replacement_confidence_threshold = replacement_confidence_threshold
        self.replacement_min_char_confidence_threshold = (
            replacement_min_char_confidence_threshold
        )
        self.non_primary_replacement_min_char_confidence = (
            non_primary_replacement_min_char_confidence
        )
        self.lexicon = set()
        self.primary_lexicon = set()
        self.by_signature = defaultdict(list)
        self.primary_by_signature = defaultdict(list)

        for term in lexicon_terms:
            normalized = self._normalize_word(term)
            if len(normalized) < 2:
                continue
            self.lexicon.add(normalized)

        if primary_terms:
            for term in primary_terms:
                normalized = self._normalize_word(term)
                if len(normalized) < 2:
                    continue
                self.primary_lexicon.add(normalized)

        # Primary terms are always part of the usable lexicon.
        self.lexicon.update(self.primary_lexicon)

        for word in self.lexicon:
            self.by_signature[(len(word), word[0])].append(word)

        for word in self.primary_lexicon:
            self.primary_by_signature[(len(word), word[0])].append(word)

    @staticmethod
    def _normalize_word(word: str) -> str:
        return re.sub(r"[^A-Za-z]", "", word or "").upper()

    @staticmethod
    def _edit_distance(a: str, b: str, max_distance: int) -> int:
        if abs(len(a) - len(b)) > max_distance:
            return max_distance + 1

        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            curr = [i]
            min_row = i
            for j, cb in enumerate(b, start=1):
                cost = 0 if ca == cb else 1
                curr_val = min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
                curr.append(curr_val)
                min_row = min(min_row, curr_val)

            if min_row > max_distance:
                return max_distance + 1
            prev = curr

        return prev[-1]

    def _nearest_word_from_pool(
        self,
        word: str,
        pool: List[str],
    ) -> Tuple[Optional[str], Optional[int]]:
        normalized = self._normalize_word(word)
        if len(normalized) < 2:
            return None, None

        if not pool:
            return None, None

        best_word = None
        best_dist = self.max_edit_distance + 1

        for candidate in pool:
            dist = self._edit_distance(normalized, candidate, self.max_edit_distance)
            if dist < best_dist:
                best_dist = dist
                best_word = candidate
                if dist == 1:
                    break

        if best_word is None or best_dist > self.max_edit_distance:
            return None, None

        return best_word, best_dist

    def _nearest_lexicon_word(self, word: str) -> Tuple[Optional[str], Optional[int]]:
        normalized = self._normalize_word(word)
        if len(normalized) < 2:
            return None, None

        if normalized in self.lexicon:
            return normalized, 0

        pool = []
        for length in range(len(normalized) - 1, len(normalized) + 2):
            if length < 2:
                continue
            pool.extend(self.by_signature.get((length, normalized[0]), []))

        return self._nearest_word_from_pool(word, pool)

    def _nearest_primary_word(self, word: str) -> Tuple[Optional[str], Optional[int]]:
        normalized = self._normalize_word(word)
        if len(normalized) < 2 or not self.primary_lexicon:
            return None, None

        if normalized in self.primary_lexicon:
            return normalized, 0

        pool = []
        for length in range(len(normalized) - 1, len(normalized) + 2):
            if length < 2:
                continue
            pool.extend(self.primary_by_signature.get((length, normalized[0]), []))

        return self._nearest_word_from_pool(word, pool)

    def decode_word(
        self,
        char_candidates: List[List[Dict[str, float]]],
        beam_width: int = 20,
    ) -> Dict:
        if not char_candidates:
            return {
                "raw_word": "",
                "decoded_word": "",
                "lexicon_word": None,
                "distance": None,
                "score": float("-inf"),
                "raw_confidence": 0.0,
                "replacement_applied": False,
                "replacement_reason": None,
            }

        beams = [("", 0.0)]
        top1_confidences: List[float] = []

        for position_candidates in char_candidates:
            if not position_candidates:
                continue

            best_conf = max(float(cand.get("confidence", 0.0)) for cand in position_candidates)
            top1_confidences.append(best_conf)

            expanded = []
            for prefix, score in beams:
                for cand in position_candidates:
                    ch = str(cand.get("character", ""))
                    confidence = max(float(cand.get("confidence", 0.0)), 1e-8)
                    expanded.append((prefix + ch, score + math.log(confidence)))

            expanded.sort(key=lambda item: item[1], reverse=True)
            beams = expanded[:beam_width]

        raw_word, raw_score = beams[0]
        normalized_raw = self._normalize_word(raw_word)
        length = max(1, len(normalized_raw))
        raw_confidence = float(math.exp(raw_score / length))

        if top1_confidences:
            top1_mean_confidence = sum(top1_confidences) / len(top1_confidences)
            min_top1_confidence = min(top1_confidences)
        else:
            top1_mean_confidence = 0.0
            min_top1_confidence = 0.0

        lexicon_word = None
        distance = None
        replacement_applied = False
        replacement_reason = None

        if normalized_raw in self.lexicon:
            lexicon_word = normalized_raw
            distance = 0

        decoded_word = normalized_raw or raw_word

        if lexicon_word is not None:
            decoded_word = lexicon_word
        else:
            near_word, near_dist = self._nearest_lexicon_word(raw_word)
            if near_word is not None and near_dist is not None:
                lexicon_word = near_word
                distance = near_dist

                # Confidence-gated replacement: only override when confidence is low.
                is_primary_candidate = near_word in self.primary_lexicon

                if is_primary_candidate:
                    allow_replacement = (
                        top1_mean_confidence <= self.replacement_confidence_threshold
                        or min_top1_confidence <= self.replacement_min_char_confidence_threshold
                    )
                else:
                    # Broad-lexicon fallback is intentionally conservative.
                    allow_replacement = (
                        near_dist <= 1
                        and min_top1_confidence <= self.non_primary_replacement_min_char_confidence
                    )

                if allow_replacement:
                    decoded_word = near_word
                    replacement_applied = True
                    replacement_reason = "low-confidence-lexicon-fallback"
                else:
                    decoded_word = normalized_raw or raw_word
                    replacement_reason = "high-confidence-kept-raw"

        # If raw is a non-primary exact lexicon hit, prefer close primary terms on low confidence.
        if (
            normalized_raw in self.lexicon
            and normalized_raw not in self.primary_lexicon
            and self.primary_lexicon
        ):
            primary_word, primary_dist = self._nearest_primary_word(raw_word)
            if (
                primary_word is not None
                and primary_dist is not None
                and primary_dist <= 1
                and (
                    top1_mean_confidence <= self.replacement_confidence_threshold
                    or min_top1_confidence <= self.replacement_min_char_confidence_threshold
                )
            ):
                decoded_word = primary_word
                lexicon_word = primary_word
                distance = primary_dist
                replacement_applied = True
                replacement_reason = "low-confidence-primary-fallback"

        return {
            "raw_word": normalized_raw or raw_word,
            "decoded_word": decoded_word,
            "lexicon_word": lexicon_word,
            "distance": distance,
            "score": raw_score,
            "raw_confidence": raw_confidence,
            "top1_mean_confidence": top1_mean_confidence,
            "min_top1_confidence": min_top1_confidence,
            "replacement_applied": replacement_applied,
            "replacement_reason": replacement_reason,
        }


def load_english_words(
    path: Path,
    max_words: int = 80000,
    min_len: int = 2,
    max_len: int = 14,
) -> List[str]:
    if not path.exists():
        return []

    words = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            token = line.strip()
            if token and token.isalpha() and min_len <= len(token) <= max_len:
                words.append(token)
                if max_words and len(words) >= max_words:
                    break

    return words
