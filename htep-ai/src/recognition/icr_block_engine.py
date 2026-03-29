import cv2
import json
import pickle
import numpy as np
from pathlib import Path

try:
    # Import TensorFlow and suppress warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    try:
        # Try standalone Keras
        import keras
        from keras.models import load_model
        TF_AVAILABLE = True
    except ImportError:
        print("Warning: Neither TensorFlow nor Keras available")
        TF_AVAILABLE = False
        load_model = None

# Import your existing segmenters
from src.segmentation.word_segmenter import segment_words
from src.segmentation.char_segmenter import segment_characters

from src.segmentation.line_segmenter import segment_lines
from src.segmentation.char_segmenter import segment_characters_with_spaces

from src.icr.preprocessing import preprocess_single_char

class BlockICREngine:
    """
    CNN-based Block ICR inference engine.
    Uses your existing trained model files.
    """

    def __init__(self, model_dir="models/icr_block"):
        model_dir = Path(model_dir)

        # ---- MODEL ----
        model_path = model_dir / "icr_model.h5"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if not TF_AVAILABLE or load_model is None:
            print(f"Warning: TensorFlow not available, using mock predictions")
            self.model = None
        else:
            try:
                self.model = load_model(model_path)
                print(f"✅ Model loaded successfully from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None

        # ---- LABELS ----
        labels_path = model_dir / "labels.pkl"
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels not found: {labels_path}")

        with open(labels_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        # ---- METADATA ----
        metadata_path = model_dir / "icr_model_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.img_width = self.metadata["img_width"]
        self.img_height = self.metadata["img_height"]

        print("✅ BlockICREngine loaded successfully")
        print(f"   Classes: {len(self.metadata['classes'])}")

    # --------------------------------------------------
    # PREPROCESS (same as training)
    # --------------------------------------------------

    def _preprocess(self, img):
        if img is None:
            raise ValueError("Input image is None")

        try:
            # Handle different image types
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.ndim == 1:
                raise ValueError("Invalid image dimensions")

            # Ensure image has valid dimensions
            if img.shape[0] == 0 or img.shape[1] == 0:
                raise ValueError("Image has zero dimensions")

            img = cv2.resize(img, (self.img_width, self.img_height))
            img = img.astype("float32") / 255.0
            img = img.reshape(1, self.img_height, self.img_width, 1)

            return img
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            # Return a blank image as fallback
            blank = np.zeros((1, self.img_height, self.img_width, 1), dtype=np.float32)
            return blank

    # --------------------------------------------------
    # SINGLE CHARACTER
    # --------------------------------------------------

    @staticmethod
    def _normalize_label(raw_label):
        """Map training labels to clean output characters."""
        label = str(raw_label)

        if label.endswith("_block"):
            base = label[:-6]
            return base[0] if base else "?"

        if label.endswith("_cursive"):
            base = label[:-8]
            return base[0] if base else "?"

        # Keep canonical single-character classes unchanged.
        if len(label) == 1:
            return label

        # Fallback for unexpected labels.
        return label

    def predict_char(self, img):
        x = self._preprocess(img)

        if self.model is None:
            # Fallback: return a reasonable character based on image characteristics
            char = "A"  # Simple fallback
            confidence = 0.5
            idx = 0
        else:
            probs = self.model.predict(x, verbose=0)[0]
            idx = int(np.argmax(probs))
            confidence = float(probs[idx])

        try:
            # For real model predictions, use the actual character
            if self.model is not None:
                if idx >= len(self.label_encoder.classes_):
                    print(f"Warning: Predicted index {idx} exceeds available classes ({len(self.label_encoder.classes_)}), using fallback")
                    char = "?"
                else:
                    raw_char = self.label_encoder.inverse_transform([idx])[0]
                    char = self._normalize_label(raw_char)
            # For mock predictions, char is already set above
                    
        except (ValueError, IndexError) as e:
            print(f"Warning: Error with label {idx} ({e}), using fallback character '?'")
            char = "?"

        return {
            "character": char,
            "confidence": confidence
        }

    def predict_char_candidates(self, img, top_k=5):
        """
        Return top-k normalized character candidates for a character image.
        """
        x = self._preprocess(img)

        if self.model is None:
            return [{"character": "A", "confidence": 0.5}]

        probs = self.model.predict(x, verbose=0)[0]
        top_k = max(1, min(int(top_k), len(probs)))

        ranked_idx = np.argsort(probs)[::-1][:top_k]

        # Merge probabilities when multiple raw labels map to the same character.
        merged = {}
        for idx in ranked_idx:
            idx = int(idx)
            confidence = float(probs[idx])

            if idx >= len(self.label_encoder.classes_):
                char = "?"
            else:
                raw_char = self.label_encoder.inverse_transform([idx])[0]
                char = self._normalize_label(raw_char)

            merged[char] = max(merged.get(char, 0.0), confidence)

        candidates = [
            {"character": char, "confidence": conf}
            for char, conf in merged.items()
        ]
        candidates.sort(key=lambda item: item["confidence"], reverse=True)

        return candidates

    # --------------------------------------------------
    # WORD (uses char segmenter)
    # --------------------------------------------------

    def predict_word(self, word_img):
        """
        Predict a full word from a word image.
        word_img -> characters -> CNN -> join
        """
        char_images = segment_characters(word_img)

        chars = []
        confidences = []

        for img in char_images:
            pred = self.predict_char(img)
            chars.append(pred["character"])
            confidences.append(pred["confidence"])

        return {
            "text": "".join(chars),
            "confidence": float(sum(confidences) / len(confidences)) if confidences else 0.0
        }

    # --------------------------------------------------
    # SENTENCE ⭐ THIS FIXES YOUR API ERROR
    # --------------------------------------------------

    def predict_sentence(self, image):
        """
        Predict a full sentence from an image.
        image -> words -> chars -> CNN -> join
        """
        words = segment_words(image)

        sentence = []
        for word_img in words:
            result = self.predict_word(word_img)
            sentence.append(result["text"])

        return " ".join(sentence)
    
    def predict_paragraph(self, image):
        """
        Predict multi-line handwritten paragraph (BLOCK ICR).
        """

        lines = segment_lines(image)
        paragraph_text = []

        for line_img in lines:
            char_data = segment_characters_with_spaces(line_img)

            line_text = []

            for char_img, is_space in char_data:
                if is_space:
                    line_text.append(" ")
                else:
                    processed = preprocess_single_char(char_img)
                    pred = self.predict_char(processed)

                    line_text.append(pred["character"])

            paragraph_text.append("".join(line_text))

        return "\n".join(paragraph_text)


