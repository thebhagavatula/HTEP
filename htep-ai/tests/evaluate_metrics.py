import json
import os
import sys
import time
from rapidfuzz.distance import Levenshtein

# Add the project root to sys.path so we can import 'app' and 'src' modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.api import ocr_engine, medical_extractor

def compute_cer(truth, pred):
    dist = Levenshtein.distance(truth.strip(), pred.strip())
    return dist / max(len(truth.strip()), 1)

def flatten_extracted(data):
    fields = []
    for k, v in data.items():
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    for sub_k, sub_v in item.items():
                        if sub_v:
                            fields.append(f"{k}.{sub_k}:{sub_v}")
                else:
                    if item:
                        fields.append(f"{k}:{item}")
        elif isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if sub_v:
                    fields.append(f"{k}.{sub_k}:{sub_v}")
        else:
            if v:
                fields.append(f"{k}:{v}")
    return set(fields)

def main():
    gt_file = os.path.join(os.path.dirname(__file__), "ground_truth.json")
    with open(gt_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metrics = {
        "printed_cer": [],
        "handwritten_cer": [],
        "entity_accuracy": []
    }

    for item in data:
        img_path = item["image_path"]
        abs_img_path = os.path.join(project_root, img_path)
        print(f"Processing {abs_img_path}...")
        
        # Run OCR
        pred_text = ocr_engine.extract_from_image(abs_img_path)
        
        # Calculate CER
        cer = compute_cer(item["raw_text"], pred_text)
        if item["type"] == "printed":
            metrics["printed_cer"].append(cer)
        else:
            metrics["handwritten_cer"].append(cer)
            
        # Run Extractor
        pred_ext = medical_extractor.extract(pred_text)
        
        # Compare entities
        gt_fields = flatten_extracted(item["extracted_data"])
        pred_fields = flatten_extracted(pred_ext)
        
        intersection = gt_fields.intersection(pred_fields)
        accuracy = len(intersection) / max(len(gt_fields), 1)
        metrics["entity_accuracy"].append(accuracy)

    # Summarize
    avg_printed_acc = (1 - sum(metrics["printed_cer"])/len(metrics["printed_cer"])) * 100 if metrics["printed_cer"] else 0
    avg_hw_acc = (1 - sum(metrics["handwritten_cer"])/len(metrics["handwritten_cer"])) * 100 if metrics["handwritten_cer"] else 0
    avg_entity_acc = (sum(metrics["entity_accuracy"])/len(metrics["entity_accuracy"])) * 100 if metrics["entity_accuracy"] else 0

    print("\n--- RESULTS ---")
    print(f"Printed OCR Accuracy: {avg_printed_acc:.2f}%")
    if metrics["handwritten_cer"]:
        print(f"Handwritten ICR Accuracy: {avg_hw_acc:.2f}%")
    else:
        print("Handwritten ICR Accuracy: N/A (No handwritten test data)")
    print(f"Entity Extraction Accuracy: {avg_entity_acc:.2f}%")

if __name__ == "__main__":
    main()
