# src/icr/train_model.py

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Try to import deep learning libraries
try:
    from tensorflow import keras
    from tensorflow.keras import layers, models

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("⚠ TensorFlow not installed. Using scikit-learn fallback.")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class ICRTrainer:
    """
    Train an ICR model for block letter recognition.
    Supports both deep learning (CNN) and traditional ML approaches.
    """

    def __init__(self, dataset_dir: str = "data/icr_training",
                 model_type: str = "cnn"):
        """
        Initialize ICR trainer.

        Args:
            dataset_dir: Directory containing training data
            model_type: 'cnn' for deep learning, 'rf' for Random Forest, 'svm' for SVM
        """
        self.dataset_dir = Path(dataset_dir)
        self.model_type = model_type
        self.model = None
        self.label_encoder = LabelEncoder()

        # Load metadata
        metadata_file = self.dataset_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            self.characters = self.metadata['characters']
            self.img_width = self.metadata['image_width']
            self.img_height = self.metadata['image_height']
        else:
            print("⚠ Metadata not found. Using defaults.")
            self.characters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            self.img_width = 28
            self.img_height = 28

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load training and test datasets.

        Returns:
            X_train, y_train, X_test, y_test
        """
        print("Loading dataset...")

        # Load training data
        X_train, y_train = self._load_data_from_dir(self.dataset_dir / "train")
        print(f"✓ Loaded {len(X_train)} training samples")

        # Load test data
        X_test, y_test = self._load_data_from_dir(self.dataset_dir / "test")
        print(f"✓ Loaded {len(X_test)} test samples")

        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

        # Normalize images
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0

        return X_train, y_train_encoded, X_test, y_test_encoded

    def _load_data_from_dir(self, directory: Path) -> Tuple[np.ndarray, List[str]]:
        """Load all images and labels from directory."""
        images = []
        labels = []

        for char_dir in directory.iterdir():
            if not char_dir.is_dir():
                continue

            char = char_dir.name

            # Load all images for this character
            for img_file in char_dir.glob("*.png"):
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)

                if img is not None:
                    images.append(img)
                    labels.append(char)

        return np.array(images), labels

    def build_cnn_model(self) -> models.Sequential:
        """
        Build a Convolutional Neural Network for character recognition.

        Returns:
            Keras Sequential model
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required for CNN model")

        num_classes = len(self.characters)

        model = models.Sequential([
            # First Convolutional Block
            layers.Input(shape=(self.img_height, self.img_width, 1)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            # Third Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),

            # Fully Connected Layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("\nCNN Model Architecture:")
        model.summary()

        return model

    def train_cnn(self, X_train, y_train, X_test, y_test,
                  epochs: int = 20, batch_size: int = 32):
        """
        Train CNN model.

        Args:
            X_train, y_train, X_test, y_test: Training and test data
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        print(f"\nTraining CNN model...")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")

        # Reshape data for CNN (add channel dimension)
        X_train = X_train.reshape(-1, self.img_height, self.img_width, 1)
        X_test = X_test.reshape(-1, self.img_height, self.img_width, 1)

        # Build model
        self.model = self.build_cnn_model()

        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )

        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\n✓ Training complete!")
        print(f"Test accuracy: {test_accuracy * 100:.2f}%")

        return history

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest classifier."""
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for Random Forest")

        print(f"\nTraining Random Forest model...")

        # Flatten images
        X_train_flat = X_train.reshape(len(X_train), -1)
        X_test_flat = X_test.reshape(len(X_test), -1)

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        self.model.fit(X_train_flat, y_train)

        # Evaluate
        train_accuracy = self.model.score(X_train_flat, y_train)
        test_accuracy = self.model.score(X_test_flat, y_test)

        print(f"\n✓ Training complete!")
        print(f"Train accuracy: {train_accuracy * 100:.2f}%")
        print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    def evaluate_model(self, X_test, y_test):
        """Evaluate model and print detailed metrics."""
        print("\nEvaluating model...")

        if self.model_type == 'cnn':
            X_test = X_test.reshape(-1, self.img_height, self.img_width, 1)
            y_pred = np.argmax(self.model.predict(X_test, verbose=0), axis=1)
        else:
            X_test_flat = X_test.reshape(len(X_test), -1)
            y_pred = self.model.predict(X_test_flat)

        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_
        ))

        # Per-character accuracy
        print("\nPer-Character Accuracy:")
        for i, char in enumerate(self.label_encoder.classes_):
            char_mask = y_test == i
            if char_mask.sum() > 0:
                char_accuracy = (y_pred[char_mask] == y_test[char_mask]).mean()
                print(f"  {char}: {char_accuracy * 100:.2f}%")

    def save_model(self, model_path: str = "models/icr_model"):
        """Save trained model."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        if self.model_type == 'cnn':
            # Save Keras model
            self.model.save(str(model_path) + '.h5')
            print(f"✓ CNN model saved to: {model_path}.h5")
        else:
            # Save sklearn model
            with open(str(model_path) + '.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            print(f"✓ Model saved to: {model_path}.pkl")

        # Save label encoder
        with open(str(model_path) + '_labels.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"✓ Label encoder saved")

        # Save metadata
        model_metadata = {
            'model_type': self.model_type,
            'characters': self.characters,
            'img_width': self.img_width,
            'img_height': self.img_height,
            'num_classes': len(self.characters)
        }

        with open(str(model_path) + '_metadata.json', 'w') as f:
            json.dump(model_metadata, f, indent=2)
        print(f"✓ Metadata saved")

    def train(self, epochs: int = 20, batch_size: int = 32):
        """
        Complete training pipeline.

        Args:
            epochs: Number of epochs (for CNN)
            batch_size: Batch size (for CNN)
        """
        # Load data
        X_train, y_train, X_test, y_test = self.load_dataset()

        # Train model based on type
        if self.model_type == 'cnn' and HAS_TENSORFLOW:
            history = self.train_cnn(X_train, y_train, X_test, y_test, epochs, batch_size)
        elif self.model_type == 'rf':
            self.train_random_forest(X_train, y_train, X_test, y_test)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Evaluate
        self.evaluate_model(X_test, y_test)

        # Save model
        self.save_model()

        print("\n" + "=" * 60)
        print("ICR Training Complete!")
        print("=" * 60)


def quick_train(model_type: str = 'rf', epochs: int = 10):
    """
    Quick training function.

    Args:
        model_type: 'cnn', 'rf', or 'svm'
        epochs: Number of epochs for CNN
    """
    trainer = ICRTrainer(model_type=model_type)
    trainer.train(epochs=epochs)


if __name__ == "__main__":
    # Quick training with Random Forest (faster, no GPU needed)
    print("Starting ICR Training...")
    quick_train(model_type='rf')