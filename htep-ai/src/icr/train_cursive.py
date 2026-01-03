import os
import cv2
import json
import numpy as np
import random
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------- CONFIG ----------------
DATA_DIR = Path("data/icr_training/cursive/train")
IMG_HEIGHT = 32
MAX_WIDTH = 128
CHARS = "abcdefghijklmnopqrstuvwxyz"
BATCH_SIZE = 16
EPOCHS = 50

# ---------------- CHAR MAP ----------------
char_to_idx = {c: i for i, c in enumerate(CHARS)}
idx_to_char = {i: c for i, c in enumerate(CHARS)}
BLANK_IDX = len(CHARS)
NUM_CLASSES = len(CHARS) + 1

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(img_path, augment=False):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape
    scale = IMG_HEIGHT / h
    new_w = min(int(w * scale), MAX_WIDTH)

    img = cv2.resize(img, (new_w, IMG_HEIGHT))

    if augment:
        if random.random() < 0.3:
            angle = random.uniform(-2, 2)
            M = cv2.getRotationMatrix2D((new_w//2, IMG_HEIGHT//2), angle, 1)
            img = cv2.warpAffine(img, M, (new_w, IMG_HEIGHT), borderValue=255)

    canvas = np.ones((IMG_HEIGHT, MAX_WIDTH), dtype=np.uint8) * 255
    canvas[:, :new_w] = img

    img = canvas.astype("float32") / 255.0
    return img[..., np.newaxis]

# ---------------- LOAD DATA ----------------
images, labels = [], []

for word_dir in DATA_DIR.iterdir():
    if not word_dir.is_dir():
        continue

    word = word_dir.name
    encoded = [char_to_idx[c] for c in word]

    for img_path in word_dir.glob("*.png"):
        img = preprocess_image(img_path, augment=True)
        if img is None:
            continue
        images.append(img)
        labels.append(encoded)

X = np.array(images)
y = labels

# ---------------- PAD LABELS ----------------
max_label_len = max(len(l) for l in y)
y_padded = np.ones((len(y), max_label_len), dtype=np.int32) * -1
label_lengths = []

for i, l in enumerate(y):
    y_padded[i, :len(l)] = l
    label_lengths.append(len(l))

label_lengths = np.array(label_lengths).reshape(-1, 1)

# ---------------- MODEL ----------------
inputs = layers.Input(shape=(IMG_HEIGHT, MAX_WIDTH, 1))

x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D((2, 2))(x)

# TIME DIMENSION = WIDTH // 4
x = layers.Reshape((MAX_WIDTH // 4, -1))(x)

x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

# ---------------- CTC LOSS ----------------
labels_in = layers.Input(shape=(max_label_len,), dtype="int32")
input_len = layers.Input(shape=(1,), dtype="int32")
label_len = layers.Input(shape=(1,), dtype="int32")

loss = layers.Lambda(
    lambda args: tf.keras.backend.ctc_batch_cost(*args),
    name="ctc_loss"
)([labels_in, outputs, input_len, label_len])

train_model = models.Model(
    inputs=[inputs, labels_in, input_len, label_len],
    outputs=loss
)

train_model.compile(
    optimizer="adam",
    loss=lambda y_true, y_pred: y_pred
)

input_lengths = np.ones((len(X), 1), dtype=np.int32) * (MAX_WIDTH // 4)

# ---------------- TRAIN ----------------
train_model.fit(
    [X, y_padded, input_lengths, label_lengths],
    np.zeros(len(X)),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

# ---------------- SAVE ----------------
os.makedirs("models/icr_cursive", exist_ok=True)

# SAVE INFERENCE MODEL (IMPORTANT)
infer_model = models.Model(inputs, outputs)
infer_model.save("models/icr_cursive/icr_cursive_infer.h5")

with open("models/icr_cursive/char_map.json", "w") as f:
    json.dump(char_to_idx, f)

print("✅ Cursive ICR training complete (CTC-safe)")
