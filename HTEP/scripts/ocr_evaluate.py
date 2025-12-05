"""
OCR evaluation harness: runs the improved processing across a grid of preprocessing parameters
and scores results using simple metrics.

Run:
    python scripts/ocr_evaluate.py

Results saved to `data/processed/ocr_evaluation_results.json`
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from itertools import product

# Ensure workspace root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ocr.extractor import OCRExtractor
from scripts.process_raw_data_improved import process_file_improved


def score_result(result):
    """Compute raw metrics for an OCR result dict (pages dict)"""
    pages = result.get("pages", {})
    total_pages = len(pages)
    nonempty_pages = 0
    total_chars = 0
    total_words = 0
    distinct_words = set()

    for p, text in pages.items():
        if text and text.strip():
            nonempty_pages += 1
            cleaned = text.strip()
            total_chars += len(cleaned)
            tokens = [t.strip("\"'.,:;()[]{}<>!?") for t in cleaned.split() if t.strip()]
            words = [t.lower() for t in tokens if any(c.isalpha() for c in t)]
            total_words += len(words)
            distinct_words.update(words)

    nonempty_ratio = (nonempty_pages / total_pages) if total_pages else 0
    return {
        "total_pages": total_pages,
        "nonempty_pages": nonempty_pages,
        "nonempty_ratio": nonempty_ratio,
        "total_chars": total_chars,
        "total_words": total_words,
        "distinct_words": len(distinct_words),
    }


def normalize_scores(metrics_list, key):
    vals = [m[key] for m in metrics_list]
    if not vals:
        return [0 for _ in vals]
    mn = min(vals)
    mx = max(vals)
    if mx == mn:
        return [1.0 for _ in vals]
    norms = [(v - mn) / (mx - mn) for v in vals]
    return norms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpis", default="300,400,500", help="Comma-separated DPI values")
    parser.add_argument("--upscales", default="1,2", help="Comma-separated upscales")
    parser.add_argument("--denoise-options", default="True,False", help="Comma-separated denoise bools")
    parser.add_argument("--deskew-options", default="True,False", help="Comma-separated deskew bools")
    parser.add_argument("--save-images", action="store_true", help="Save debug images")
    parser.add_argument("--files", default=None, help="Optional comma-separated raw files to limit to those in data/raw")
    parser.add_argument("--top", type=int, default=5, help="Number of top results to show per file")
    args = parser.parse_args()

    dpis = [int(x) for x in args.dpis.split(",") if x]
    upscales = [int(x) for x in args.upscales.split(",") if x]
    denoise_opts = [x.strip().lower() == "true" for x in args.denoise_options.split(",") if x]
    deskew_opts = [x.strip().lower() == "true" for x in args.deskew_options.split(",") if x]

    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = list(raw_dir.glob("**/*.*"))
    if args.files:
        wanted = [s.strip() for s in args.files.split(",")]
        files = [f for f in files if f.name in wanted]

    if not files:
        print("No input files found in data/raw/")
        raise SystemExit(1)

    extractor = OCRExtractor()

    grid = list(product(dpis, upscales, denoise_opts, deskew_opts))
    print(f"Running grid with {len(grid)} combos on {len(files)} file(s)\n")

    results = {}

    for file_path in files:
        file_results = []
        print(f"Evaluating: {file_path.name}")
        for dpi, upscale, denoise, deskew in grid:
            # Build an args namespace to pass to process_file_improved
            _args = SimpleNamespace(dpi=dpi, upscale=upscale, denoise=denoise, deskew=deskew, save_images=args.save_images)
            print(f"  Combo: dpi={dpi}, upscale={upscale}, denoise={denoise}, deskew={deskew}")
            out = process_file_improved(extractor, file_path, _args)
            metrics = score_result(out)
            entry = {
                "combo": {"dpi": dpi, "upscale": upscale, "denoise": denoise, "deskew": deskew},
                "metrics": metrics,
                "result": out,
            }
            file_results.append(entry)

        # Normalize metrics across combos for this file and compute a final score
        total_chars_norm = normalize_scores([e['metrics'] for e in file_results], 'total_chars')
        total_words_norm = normalize_scores([e['metrics'] for e in file_results], 'total_words')
        distinct_norm = normalize_scores([e['metrics'] for e in file_results], 'distinct_words')
        nonempty_norm = normalize_scores([e['metrics'] for e in file_results], 'nonempty_ratio')

        # scoring weights
        weights = {'distinct_words': 0.5, 'total_words': 0.3, 'total_chars': 0.1, 'nonempty_ratio': 0.1}

        # Attach normalized values and compute score
        for idx, ent in enumerate(file_results):
            sc = (weights['distinct_words'] * distinct_norm[idx]
                  + weights['total_words'] * total_words_norm[idx]
                  + weights['total_chars'] * total_chars_norm[idx]
                  + weights['nonempty_ratio'] * nonempty_norm[idx])
            ent['score'] = sc
            ent['norm'] = {'total_chars': total_chars_norm[idx], 'total_words': total_words_norm[idx], 'distinct_words': distinct_norm[idx], 'nonempty_ratio': nonempty_norm[idx]}

        # sort by score desc
        file_results_sorted = sorted(file_results, key=lambda x: x['score'], reverse=True)
        results[file_path.name] = file_results_sorted

        # Save per-file results
        with open(processed_dir / f"ocr_eval_{file_path.stem}.json", 'w', encoding='utf-8') as f:
            json.dump({file_path.name: file_results_sorted}, f, indent=2, ensure_ascii=False)

        # Print top combos
        print(f"Top {args.top} combos for {file_path.name}:")
        for r in file_results_sorted[:args.top]:
            combo = r['combo']
            m = r['metrics']
            print(f"  score={r['score']:.4f} combo={combo} distinct={m['distinct_words']} words={m['total_words']} chars={m['total_chars']} non_empty_pages={m['nonempty_pages']}/{m['total_pages']}")

    # Save overall evaluation
    outpath = processed_dir / 'ocr_evaluation_results.json'
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results written to {outpath}")
