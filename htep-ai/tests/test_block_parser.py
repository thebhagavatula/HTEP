from src.nlp.block_parser import BlockTextParser


def test_block_parser_corrects_ocr_confusions():
    parser = BlockTextParser(
        dictionary_terms=["aspirin", "diabetes", "metformin", "hypertension"]
    )

    result = parser.parse("Patient has diabete5 and takes asp1rin daily.")

    assert "diabetes" in result["corrected_text"].lower()
    assert "aspirin" in result["corrected_text"].lower()
    assert any(item["from"] == "diabete5" for item in result["corrections"])
    assert any(item["from"] == "asp1rin" for item in result["corrections"])


def test_block_parser_returns_dictionary_matches_and_backend():
    parser = BlockTextParser(dictionary_terms=["hypertension", "metformin"])
    result = parser.parse("Hypertens1on treated with metformin")

    lowered_matches = {match.lower() for match in result["dictionary_matches"]}

    assert "hypertension" in lowered_matches
    assert "metformin" in lowered_matches
    assert result["backend"] in {"regex", "spacy", "spacy+scispacy"}


def test_block_parser_english_layer_corrects_common_words():
    parser = BlockTextParser(
        dictionary_terms=["aspirin"],
        english_terms=["hello", "world", "test"],
        english_similarity_cutoff=0.8,
    )
    result = parser.parse("he11o wor1d te5t")

    assert result["corrected_text"].lower() == "hello world test"
    assert all(item.get("source") == "english" for item in result["corrections"])


def test_block_parser_reports_layered_dictionary_matches():
    parser = BlockTextParser(
        dictionary_terms=["diabetes"],
        english_terms=["patient", "daily"],
    )
    result = parser.parse("Patient has diabetes daily")

    medical_matches = {word.lower() for word in result["dictionary_layers"]["medical"]}
    english_matches = {word.lower() for word in result["dictionary_layers"]["english"]}

    assert "diabetes" in medical_matches
    assert "patient" in english_matches
    assert "daily" in english_matches
