import cv2
import json
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import tqdm

# ---------------- CONFIG ----------------
TEST_DIR = Path("data/icr_training/cursive/test")
IMG_HEIGHT = 32
MAX_WIDTH = 128
CHARS = "abcdefghijklmnopqrstuvwxyz"

MODEL_PATH = "models/icr_cursive/icr_cursive_model.h5"
CHAR_MAP_PATH = "models/icr_cursive/char_map.json"

# ---------------- LOAD CHAR MAP ----------------
with open(CHAR_MAP_PATH, "r") as f:
    char_to_idx = json.load(f)

idx_to_char = {int(v): k for k, v in char_to_idx.items()}
NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank

# ---------------- IMAGE PREPROCESS ----------------
def preprocess(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None  # unreadable image

    h, w = img.shape

    scale = IMG_HEIGHT / h
    new_w = int(w * scale)
    img = cv2.resize(img, (new_w, IMG_HEIGHT))

    canvas = np.ones((IMG_HEIGHT, MAX_WIDTH), dtype=np.uint8) * 255
    canvas[:, :min(new_w, MAX_WIDTH)] = img[:, :min(new_w, MAX_WIDTH)]

    img = canvas.astype("float32") / 255.0
    return img[np.newaxis, ..., np.newaxis]

# ---------------- BUILD PREDICTION MODEL ----------------
inputs = layers.Input(shape=(IMG_HEIGHT, MAX_WIDTH, 1))

x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D(2, 2)(x)

x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D(2, 2)(x)

x = layers.Reshape((MAX_WIDTH // 4, -1))(x)

x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.load_weights(MODEL_PATH)

# ---------------- CTC GREEDY DECODER ----------------
def decode_beam(pred, beam_width=10):
    pred = tf.convert_to_tensor(pred)
    input_len = tf.ones(pred.shape[0]) * pred.shape[1]

    decoded, _ = tf.keras.backend.ctc_decode(
        pred,
        input_length=input_len,
        greedy=False,
        beam_width=beam_width
    )

    result = decoded[0][0].numpy()
    return "".join(idx_to_char[c] for c in result if c != -1)


# ---------------- EVALUATION ----------------
total = 0
correct = 0
errors = []

label_dirs = [d for d in TEST_DIR.iterdir() if d.is_dir()]

for label_dir in tqdm(label_dirs, desc="Evaluating"):
    gt_label = label_dir.name

    for img_path in label_dir.glob("*.png"):
        img = preprocess(img_path)

        if img is None:
            continue  # skip unreadable image

        pred = model.predict(img, verbose=0)
        pred_text = decode_beam(pred)

        total += 1
        if pred_text == gt_label:
            correct += 1
        else:
            errors.append((gt_label, pred_text))

accuracy = (correct / total) * 100 if total > 0 else 0.0

print("\n✅ Evaluation complete")
print(f"📊 Total samples     : {total}")
print(f"🎯 Correct predictions: {correct}")
print(f"🏆 Word Accuracy     : {accuracy:.2f}%")

print("\n🔎 Sample errors (first 10):")
for gt, pred in errors[:10]:
    print(f"GT: {gt} | Pred: {pred}")
