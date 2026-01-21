import cv2
import json
import pickle
import numpy as np
from pathlib import Path
from tensorflow import keras

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

        self.model = keras.models.load_model(model_path)

        # ---- LABELS ----
        labels_path = model_dir / "icr_model_labels.pkl"
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
        print(f"   Classes: {len(self.metadata['characters'])}")

    # --------------------------------------------------
    # PREPROCESS (same as training)
    # --------------------------------------------------

    def _preprocess(self, img):
        if img is None:
            raise ValueError("Input image is None")

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, (self.img_width, self.img_height))
        img = img.astype("float32") / 255.0
        img = img.reshape(1, self.img_height, self.img_width, 1)

        return img

    # --------------------------------------------------
    # SINGLE CHARACTER
    # --------------------------------------------------

    def predict_char(self, img):
        x = self._preprocess(img)

        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))

        char = self.label_encoder.inverse_transform([idx])[0]
        confidence = float(probs[idx])

        return {
            "character": char,
            "confidence": confidence
        }

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


