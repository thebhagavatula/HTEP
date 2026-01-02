import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------- CONFIG ----------------
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
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
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

prediction_model = models.Model(inputs, outputs)

# ---------------- LOAD TRAINED WEIGHTS ----------------
prediction_model.load_weights(MODEL_PATH)

# ---------------- GREEDY CTC DECODE ----------------
def decode(pred):
    pred = np.argmax(pred, axis=-1)[0]

    last = -1
    result = []

    for p in pred:
        if p != last and p in idx_to_char:
            result.append(idx_to_char[p])
        last = p

    return "".join(result)

# ---------------- TEST ----------------
img_path = r"tests\shrey.jpeg" # CHANGE THIS
img = preprocess(img_path)

pred = prediction_model.predict(img)
text = decode(pred)

print("📝 Predicted text:", text)
