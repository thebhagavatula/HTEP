# src/icr/inference.py

import cv2
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
import re

try:
    from tensorflow import keras

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


class ICRPredictor:
    """
    Use trained ICR model to predict characters from images.
    """

    def __init__(self, model_path: str = "models/icr_model"):
        """
        Load trained ICR model.

        Args:
            model_path: Path to saved model (without extension)
        """
        self.model_path = Path(model_path)

        # Load metadata
        metadata_file = str(self.model_path) + '_metadata.json'
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        self.model_type = self.metadata['model_type']
        self.img_width = self.metadata['img_width']
        self.img_height = self.metadata['img_height']
        self.characters = self.metadata['characters']

        # Load label encoder
        with open(str(self.model_path) + '_labels.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)

        # Load model
        if self.model_type == 'cnn':
            if not HAS_TENSORFLOW:
                raise ImportError("TensorFlow required to load CNN model")
            self.model = keras.models.load_model(str(self.model_path) + '.h5')
        else:
            with open(str(self.model_path) + '.pkl', 'rb') as f:
                self.model = pickle.load(f)

        print(f"✓ ICR model loaded: {self.model_type}")
        print(f"✓ Supports {len(self.characters)} characters")

    def predict_image(self, image_path: str) -> Dict:
        """
        Predict character from image file.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = img.astype('float32') / 255.0

        # Predict
        if self.model_type == 'cnn':
            img = img.reshape(1, self.img_height, self.img_width, 1)
            predictions = self.model.predict(img, verbose=0)[0]
            predicted_idx = np.argmax(predictions)
            confidence = predictions[predicted_idx]
        else:
            img_flat = img.reshape(1, -1)
            predicted_idx = self.model.predict(img_flat)[0]

            # Get confidence from probability estimates
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(img_flat)[0]
                confidence = probabilities[predicted_idx]
            else:
                confidence = 1.0  # No confidence available

        predicted_char = self.label_encoder.inverse_transform([predicted_idx])[0]

        return {
            'character': predicted_char,
            'confidence': float(confidence),
            'all_probabilities': predictions.tolist() if self.model_type == 'cnn' else None
        }

    def predict_array(self, img_array: np.ndarray) -> Dict:
        """
        Predict character from numpy array.

        Args:
            img_array: Grayscale image as numpy array

        Returns:
            Dictionary with prediction results
        """
        # Ensure correct size
        if img_array.shape != (self.img_height, self.img_width):
            img_array = cv2.resize(img_array, (self.img_width, self.img_height))

        # Normalize
        img_array = img_array.astype('float32') / 255.0

        # Predict
        if self.model_type == 'cnn':
            img_array = img_array.reshape(1, self.img_height, self.img_width, 1)
            predictions = self.model.predict(img_array, verbose=0)[0]
            predicted_idx = np.argmax(predictions)
            confidence = predictions[predicted_idx]
        else:
            img_flat = img_array.reshape(1, -1)
            predicted_idx = self.model.predict(img_flat)[0]

            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(img_flat)[0]
                confidence = probabilities[predicted_idx]
            else:
                confidence = 1.0

        predicted_char = self.label_encoder.inverse_transform([predicted_idx])[0]

        return {
            'character': predicted_char,
            'confidence': float(confidence)
        }

    def predict_batch(self, image_dir: str) -> List[Dict]:
        """
        Predict characters for all images in directory.

        Args:
            image_dir: Directory containing images

        Returns:
            List of prediction dictionaries
        """
        image_path = Path(image_dir)
        image_files = list(image_path.glob('*.png')) + list(image_path.glob('*.jpg'))

        results = []
        for img_file in image_files:
            result = self.predict_image(str(img_file))
            result['file'] = img_file.name
            results.append(result)

        return results

    def correct_text(self, text: str, confidence_threshold: float = 0.7) -> str:
        """
        Correct text using ICR for low-confidence OCR results.
        This is a placeholder - real implementation would need character segmentation.

        Args:
            text: Text to correct
            confidence_threshold: Minimum confidence to trust OCR

        Returns:
            Corrected text
        """
        # Common OCR errors that ICR can fix
        corrections = {
            'O': ['0', 'Q'],
            '0': ['O', 'D'],
            'l': ['1', 'I'],
            '1': ['l', 'I'],
            'S': ['5'],
            '5': ['S'],
            'Z': ['2'],
            '2': ['Z']
        }

        # This is a simple rule-based correction
        # Real ICR would segment individual characters and predict them
        corrected_text = text

        # Apply common corrections based on context
        # In medical context: numbers in values, letters in words
        words = text.split()
        corrected_words = []

        for word in words:
            # Check if word looks like a number
            if re.match(r'^\d+\.?\d*$', word):
                # Apply number corrections
                corrected = word.replace('O', '0').replace('l', '1')
            else:
                # Apply letter corrections
                corrected = word.replace('0', 'O').replace('1', 'l')

            corrected_words.append(corrected)

        return ' '.join(corrected_words)


def test_model(model_path: str = "models/icr_model",
               test_dir: str = "data/icr_training/test"):
    """
    Test ICR model on test dataset.

    Args:
        model_path: Path to trained model
        test_dir: Path to test data directory
    """
    print("Testing ICR Model...")
    print("=" * 60)

    predictor = ICRPredictor(model_path)
    test_path = Path(test_dir)

    total_correct = 0
    total_tested = 0
    per_char_results = {}

    # Test each character
    for char_dir in test_path.iterdir():
        if not char_dir.is_dir():
            continue

        char = char_dir.name
        images = list(char_dir.glob('*.png'))

        if not images:
            continue

        char_correct = 0
        char_total = len(images)

        for img_file in images:
            result = predictor.predict_image(str(img_file))

            if result['character'] == char:
                char_correct += 1
                total_correct += 1

            total_tested += 1

        char_accuracy = (char_correct / char_total) * 100
        per_char_results[char] = char_accuracy

        print(f"{char}: {char_correct}/{char_total} ({char_accuracy:.1f}%)")

    overall_accuracy = (total_correct / total_tested) * 100

    print("\n" + "=" * 60)
    print(f"Overall Accuracy: {total_correct}/{total_tested} ({overall_accuracy:.2f}%)")
    print("=" * 60)

    # Show best and worst performers
    sorted_results = sorted(per_char_results.items(), key=lambda x: x[1])

    print(f"\nLowest Accuracy (need more training):")
    for char, acc in sorted_results[:5]:
        print(f"  {char}: {acc:.1f}%")

    print(f"\nHighest Accuracy (well learned):")
    for char, acc in sorted_results[-5:]:
        print(f"  {char}: {acc:.1f}%")


if __name__ == "__main__":
    # Test the trained model
    test_model()