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

from src.config import MODELS_DIR

class BlockICREngine:
    """
    CNN-based Block ICR inference engine.
    Uses your existing trained model files.
    """

    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = MODELS_DIR / "icr_block"
        else:
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
                print(f"[OK] Model loaded successfully from {model_path}")
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

        print("[OK] BlockICREngine loaded successfully")
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

        # Keep canonical single-character classes unchanged.
        if len(label) == 1:
            return label

        # Fallback for unexpected labels.
        return label

    def predict_char(self, img):
        x = self._preprocess(img)

        if self.model is None:
            return {"character": "A", "confidence": 0.5}

        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

        try:
            if idx >= len(self.label_encoder.classes_):
                print(f"Warning: Predicted index {idx} exceeds available classes ({len(self.label_encoder.classes_)}), using fallback")
                char = "?"
            else:
                raw_char = self.label_encoder.inverse_transform([idx])[0]
                char = self._normalize_label(raw_char)
        except (ValueError, IndexError) as e:
            print(f"Warning: Error with label {idx} ({e}), using fallback character '?'")
            char = "?"

        return {"character": char, "confidence": confidence}

    def _batch_predict_chars(self, char_images: list) -> list:
        """
        Predict a list of character images in a SINGLE model.predict() call.
        Returns a list of {"character": str, "confidence": float} dicts.

        Calling model.predict() once per character (the old behaviour) adds
        significant TensorFlow graph-dispatch overhead for every character in
        a paragraph.  Batching reduces that to a single kernel launch.
        """
        if not char_images:
            return []

        if self.model is None:
            return [{"character": "A", "confidence": 0.5}] * len(char_images)

        # Build batch tensor  (N, H, W, 1)
        batch = np.concatenate([self._preprocess(img) for img in char_images], axis=0)
        all_probs = self.model.predict(batch, verbose=0)  # single call

        results = []
        for probs in all_probs:
            idx = int(np.argmax(probs))
            confidence = float(probs[idx])
            try:
                if idx >= len(self.label_encoder.classes_):
                    char = "?"
                else:
                    raw_char = self.label_encoder.inverse_transform([idx])[0]
                    char = self._normalize_label(raw_char)
            except (ValueError, IndexError):
                char = "?"
            results.append({"character": char, "confidence": confidence})

        return results

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
        image -> words -> chars -> CNN (batched) -> join
        """
        words = segment_words(image)

        # Collect all character images across all words in one pass so we
        # can batch them into a single model.predict() call.
        word_char_counts = []
        all_char_imgs = []
        for word_img in words:
            char_imgs = segment_characters(word_img)
            word_char_counts.append(len(char_imgs))
            all_char_imgs.extend(char_imgs)

        if not all_char_imgs:
            return ""

        all_preds = self._batch_predict_chars(all_char_imgs)

        sentence = []
        offset = 0
        for count in word_char_counts:
            chars = [p["character"] for p in all_preds[offset:offset + count]]
            sentence.append("".join(chars))
            offset += count

        return " ".join(sentence)

    def predict_paragraph(self, image):
        """
        Predict multi-line handwritten paragraph (BLOCK ICR).
        All non-space characters across ALL lines are batched into a single
        model.predict() call to eliminate per-character TF overhead.
        """
        lines = segment_lines(image)
        paragraph_text = []

        for line_img in lines:
            char_data = segment_characters_with_spaces(line_img)

            # Separate real characters from space placeholders.
            char_imgs = []
            slot_is_space = []
            for char_img, is_space in char_data:
                if is_space:
                    slot_is_space.append(True)
                    char_imgs.append(None)  # placeholder
                else:
                    slot_is_space.append(False)
                    char_imgs.append(preprocess_single_char(char_img))

            # Batch-predict only the real characters.
            real_imgs = [img for img in char_imgs if img is not None]
            preds = self._batch_predict_chars(real_imgs)

            # Reconstruct the line, inserting spaces in the right slots.
            line_text = []
            pred_iter = iter(preds)
            for is_space in slot_is_space:
                if is_space:
                    line_text.append(" ")
                else:
                    pred = next(pred_iter, {"character": "?"})
                    line_text.append(pred["character"])

            paragraph_text.append("".join(line_text))

        return "\n".join(paragraph_text)


