# src/icr/train_block.py
# CNN-based Block ICR Training (UNCHANGED LOGIC, UPDATED PATHS ONLY)

import os
import json
import pickle
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# TensorFlow / Keras
from tensorflow import keras
from tensorflow.keras import layers, models


class ICRTrainer:
    """
    CNN trainer for BLOCK character recognition.
    """

    def __init__(
        self,
        dataset_dir="data/icr_training/block",
        model_dir="models/icr_block"
    ):
        self.dataset_dir = Path(dataset_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.img_width = 28
        self.img_height = 28

        self.label_encoder = LabelEncoder()
        self.model = None

    # --------------------------------------------------
    # DATA LOADING (UNCHANGED)
    # --------------------------------------------------

    def _load_data_from_dir(self, directory: Path):
        images = []
        labels = []

        for char_dir in directory.iterdir():
            if not char_dir.is_dir():
                continue

            label = char_dir.name

            for img_path in char_dir.iterdir():
                if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    continue

                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                img = cv2.resize(img, (self.img_width, self.img_height))
                images.append(img)
                labels.append(label)

        return np.array(images), labels

    def load_dataset(self):
        print("📂 Loading block ICR dataset...")

        X_train, y_train = self._load_data_from_dir(self.dataset_dir / "train")
        X_test, y_test = self._load_data_from_dir(self.dataset_dir / "test")

        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_test_enc = self.label_encoder.transform(y_test)

        X_train = X_train.astype("float32") / 255.0
        X_test = X_test.astype("float32") / 255.0

        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)

        print(f"✔ Train samples: {len(X_train)}")
        print(f"✔ Test samples: {len(X_test)}")
        print(f"✔ Classes: {list(self.label_encoder.classes_)}")

        return X_train, y_train_enc, X_test, y_test_enc

    # --------------------------------------------------
    # CNN MODEL (UNCHANGED)
    # --------------------------------------------------

    def build_cnn(self):
        num_classes = len(self.label_encoder.classes_)

        model = models.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation="relu"),

            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax")
        ])

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        model.summary()
        return model

    # --------------------------------------------------
    # TRAINING (UNCHANGED)
    # --------------------------------------------------

    def train(self, epochs=20, batch_size=32):
        X_train, y_train, X_test, y_test = self.load_dataset()

        self.model = self.build_cnn()

        print("🚀 Training CNN block ICR...")
        self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )

        print("\n📊 Evaluation:")
        preds = np.argmax(self.model.predict(X_test), axis=1)
        print(classification_report(
            y_test,
            preds,
            target_names=self.label_encoder.classes_
        ))

        self.save_model()

    # --------------------------------------------------
    # SAVE MODEL (PATHS UPDATED ONLY)
    # --------------------------------------------------

    def save_model(self):
        model_path = self.model_dir / "block_icr_cnn.h5"
        labels_path = self.model_dir / "labels.pkl"
        metadata_path = self.model_dir / "metadata.json"

        self.model.save(model_path)

        with open(labels_path, "wb") as f:
            pickle.dump(self.label_encoder, f)

        metadata = {
            "type": "block_icr",
            "model": "cnn",
            "img_width": 28,
            "img_height": 28,
            "classes": list(self.label_encoder.classes_)
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n💾 Model saved to: {self.model_dir}")


# --------------------------------------------------
# SAME quick_train() YOU USED BEFORE
# --------------------------------------------------

def quick_train(model_type="cnn", epochs=20):
    if model_type != "cnn":
        raise ValueError("Only CNN is supported for block ICR")

    trainer = ICRTrainer()
    trainer.train(epochs=epochs)


if __name__ == "__main__":
    quick_train("cnn", epochs=20)
