# =========================================================
# FINAL FIXED CRNN + CTC TRAINING SCRIPT (Cursive OCR)
# =========================================================

import os
import cv2
import json
import random
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

# ---------------- CONFIG ----------------
DATA_DIR = Path("data/icr_training/cursive/train")
IMG_HEIGHT = 32
MAX_WIDTH = 128
BATCH_SIZE = 64
EPOCHS = 80

CHARS = "abcdefghijklmnopqrstuvwxyz"
NUM_CLASSES = len(CHARS) + 1  # + CTC blank

char_to_idx = {c: i for i, c in enumerate(CHARS)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(img_path, augment=False):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None

    h, w = img.shape
    scale = IMG_HEIGHT / h
    new_w = min(int(w * scale), MAX_WIDTH)

    img = cv2.resize(img, (new_w, IMG_HEIGHT))

    if augment:
        if random.random() < 0.3:
            angle = random.uniform(-4, 4)
            M = cv2.getRotationMatrix2D((new_w // 2, IMG_HEIGHT // 2), angle, 1)
            img = cv2.warpAffine(img, M, (new_w, IMG_HEIGHT), borderValue=255)

        if random.random() < 0.2:
            noise = np.random.normal(0, 4, img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)

    canvas = np.ones((IMG_HEIGHT, MAX_WIDTH), dtype=np.uint8) * 255
    canvas[:, :new_w] = img

    img = canvas.astype("float32") / 255.0
    img = img[..., np.newaxis]

    # IMPORTANT: time steps = actual width (no division)
    time_steps = new_w

    return img, time_steps

# ---------------- LOAD DATA ----------------
images, labels, input_lengths = [], [], []

for word_dir in DATA_DIR.iterdir():
    if not word_dir.is_dir():
        continue

    word = word_dir.name.lower()
    encoded = [char_to_idx[c] for c in word if c in char_to_idx]

    if not encoded:
        continue

    for img_path in word_dir.glob("*.png"):
        img, t = preprocess_image(img_path, augment=True)
        if img is None:
            continue

        images.append(img)
        labels.append(encoded)
        input_lengths.append(t)

X = np.array(images, dtype=np.float32)
input_lengths = np.array(input_lengths).reshape(-1, 1)

print(f"Loaded {len(X)} images")

# ---------------- PAD LABELS ----------------
max_label_len = max(len(l) for l in labels)

y_padded = np.ones((len(labels), max_label_len), dtype=np.int32) * -1
label_lengths = np.zeros((len(labels), 1), dtype=np.int32)

for i, l in enumerate(labels):
    y_padded[i, :len(l)] = l
    label_lengths[i] = len(l)

# ---------------- MODEL (CRNN) ----------------
inputs = layers.Input(shape=(IMG_HEIGHT, MAX_WIDTH, 1), name="image")

# CNN — width preserved
x = layers.Conv2D(64, 3, padding="same")(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D((2, 1))(x)

x = layers.Conv2D(128, 3, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D((2, 1))(x)

x = layers.Conv2D(256, 3, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

# ✅ FIXED RESHAPE (NO HARDCODE)
shape = x.shape  # (None, H, W, C)
x = layers.Reshape(
    target_shape=(shape[2], shape[1] * shape[3])
)(x)

# RNN
x = layers.Bidirectional(
    layers.LSTM(256, return_sequences=True)
)(x)
x = layers.Dropout(0.3)(x)

x = layers.Bidirectional(
    layers.LSTM(256, return_sequences=True)
)(x)
x = layers.Dropout(0.3)(x)

# Output
outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="softmax")(x)

# ---------------- CTC LOSS ----------------
labels_in = layers.Input(shape=(max_label_len,), dtype="int32", name="labels")
input_len = layers.Input(shape=(1,), dtype="int32", name="input_length")
label_len = layers.Input(shape=(1,), dtype="int32", name="label_length")

ctc_loss = layers.Lambda(
    lambda args: K.ctc_batch_cost(*args),
    name="ctc_loss"
)([labels_in, outputs, input_len, label_len])

train_model = models.Model(
    inputs=[inputs, labels_in, input_len, label_len],
    outputs=ctc_loss
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.99)

train_model.compile(
    optimizer=optimizer,
    loss=lambda y_true, y_pred: y_pred
)

train_model.summary()

# ---------------- CALLBACKS ----------------
os.makedirs("models/icr_cursive", exist_ok=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=15,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss",
        factor=0.5,
        patience=7,
        min_lr=1e-6
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "models/icr_cursive/best_model.h5",
        monitor="loss",
        save_best_only=True
    )
]

# ---------------- TRAIN ----------------
print("\n===== TRAINING START =====\n")

history = train_model.fit(
    [X, y_padded, input_lengths, label_lengths],
    np.zeros(len(X)),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# ---------------- SAVE INFERENCE MODEL ----------------
infer_model = models.Model(inputs, outputs)
infer_model.save("models/icr_cursive/icr_cursive_infer.h5")

with open("models/icr_cursive/char_map.json", "w") as f:
    json.dump(char_to_idx, f)

print("\n✅ Training complete")
print(f"Final loss: {history.history['loss'][-1]:.4f}")
