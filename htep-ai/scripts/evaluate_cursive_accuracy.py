import cv2
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
MODEL_PATH = "models/icr_cursive/icr_cursive_infer.h5"
CHAR_MAP_PATH = "models/icr_cursive/char_map.json"
TEST_DIR = Path("data/icr_training/cursive/test")

IMG_HEIGHT = 32
MAX_WIDTH = 128

# ---------------- LOAD MAP ----------------
with open(CHAR_MAP_PATH) as f:
    char_to_idx = json.load(f)

idx_to_char = {v: k for k, v in char_to_idx.items()}
BLANK_IDX = len(char_to_idx)

# ---------------- PREPROCESS ----------------
def preprocess(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape
    scale = IMG_HEIGHT / h
    new_w = min(int(w * scale), MAX_WIDTH)

    img = cv2.resize(img, (new_w, IMG_HEIGHT))

    canvas = np.ones((IMG_HEIGHT, MAX_WIDTH), dtype=np.uint8) * 255
    canvas[:, :new_w] = img

    img = canvas.astype("float32") / 255.0
    return img[..., np.newaxis]

# ---------------- LOAD MODEL ----------------
print("🔄 Loading inference model...")
model = load_model(MODEL_PATH, compile=False)

# ---------------- EVALUATE ----------------
total = 0
correct = 0
errors = []

print("🧪 Evaluating...")

for word_dir in tqdm(sorted(TEST_DIR.iterdir())):
    if not word_dir.is_dir():
        continue

    gt = word_dir.name

    for img_path in word_dir.glob("*.png"):
        img = preprocess(img_path)
        if img is None:
            continue

        img = img[np.newaxis, ...]
        preds = model.predict(img, verbose=0)

        decoded, _ = tf.keras.backend.ctc_decode(
            preds,
            input_length=[preds.shape[1]],
            greedy=True
        )

        pred_idxs = decoded[0][0].numpy()
        pred = "".join(idx_to_char[i] for i in pred_idxs if i != -1)

        total += 1
        if pred == gt:
            correct += 1
        elif len(errors) < 10:
            errors.append((gt, pred))

# ---------------- RESULTS ----------------
acc = 100 * correct / total
print("\n✅ Evaluation complete")
print(f"📊 Total samples     : {total}")
print(f"🎯 Correct predictions: {correct}")
print(f"🏆 Word Accuracy     : {acc:.2f}%")

if errors:
    print("\n🔎 Sample errors:")
    for g, p in errors:
        print(f"GT: {g} | Pred: {p}")
