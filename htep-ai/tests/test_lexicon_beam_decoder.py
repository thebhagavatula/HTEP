from src.nlp.lexicon_beam_decoder import LexiconBeamDecoder


def test_low_confidence_allows_lexicon_fallback():
    decoder = LexiconBeamDecoder(
        lexicon_terms=["hello", "name", "nilesh"],
        replacement_confidence_threshold=0.8,
        non_primary_replacement_min_char_confidence=0.55,
    )

    candidates = [
        [{"character": "H", "confidence": 0.95}],
        [{"character": "E", "confidence": 0.94}],
        [{"character": "L", "confidence": 0.93}],
        [{"character": "L", "confidence": 0.92}],
        [
            {"character": "D", "confidence": 0.52},
            {"character": "O", "confidence": 0.48},
        ],
    ]

    decoded = decoder.decode_word(candidates)

    assert decoded["raw_word"] == "HELLD"
    assert decoded["decoded_word"] == "HELLO"
    assert decoded["replacement_applied"] is True
    assert decoded["replacement_reason"] == "low-confidence-lexicon-fallback"


def test_high_confidence_keeps_raw_word():
    decoder = LexiconBeamDecoder(
        lexicon_terms=["hello", "name", "nilesh"],
        replacement_confidence_threshold=0.8,
    )

    candidates = [
        [{"character": "H", "confidence": 0.99}],
        [{"character": "E", "confidence": 0.99}],
        [{"character": "L", "confidence": 0.99}],
        [{"character": "L", "confidence": 0.99}],
        [
            {"character": "D", "confidence": 0.99},
            {"character": "O", "confidence": 0.01},
        ],
    ]

    decoded = decoder.decode_word(candidates)

    assert decoded["raw_word"] == "HELLD"
    assert decoded["decoded_word"] == "HELLD"
    assert decoded["replacement_applied"] is False
    assert decoded["replacement_reason"] == "high-confidence-kept-raw"
