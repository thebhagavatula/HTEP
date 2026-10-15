# main.py
# Command-line runner: digitize one prescription and print the record as JSON.
#   python main.py path/to/prescription.jpg

import json
import sys

from src.pipeline.digitizer import Digitizer

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python main.py <prescription.pdf|png|jpg>")
    result = Digitizer().process(sys.argv[1])
    print(json.dumps(
        {k: result[k] for k in ("extracted_data", "field_sources", "review_required", "ocr_confidence", "vlm", "timings")},
        indent=2, ensure_ascii=False,
    ))
