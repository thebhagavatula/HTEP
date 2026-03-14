def main(document):
    # Step 1: Load and preprocess document
    pages = load_document(document)

    final_text = ""

    for page in pages:
        # Step 2: OCR for printed text
        printed_text = ocr_module(page)
        final_text += printed_text

        # Step 3: Detect handwritten regions
        block_regions, cursive_regions = detect_handwriting(page)

        # Step 4: Block ICR (A–Z, 0–9)
        if block_regions:
            block_text = block_icr(block_regions)
            final_text += block_text

        # Step 5: Cursive ICR (A–Z)
        if cursive_regions:
            cursive_text = cursive_icr(cursive_regions)
            final_text += cursive_text

    # Step 6: Return consolidated output
    return final_text
