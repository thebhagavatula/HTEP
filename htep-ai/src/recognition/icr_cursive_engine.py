import cv2
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import load_model

from src.segmentation.line_segmenter import segment_lines
from src.segmentation.cursive_word_segmenter import segment_cursive_words



class CursiveICREngine:
    """
    CRNN + CTC based cursive handwriting recognizer.
    Word-level inference.
    """

    def __init__(self, model_dir="models/icr_cursive"):
        model_dir = Path(model_dir)

        # ---- LOAD MODEL ----
        model_path = model_dir / "icr_cursive_infer.h5"
        if not model_path.exists():
            raise FileNotFoundError(f"Cursive model not found: {model_path}")

        self.model = load_model(model_path, compile=False)

        # ---- LOAD CHAR MAP ----
        char_map_path = model_dir / "char_map.json"
        if not char_map_path.exists():
            raise FileNotFoundError(f"Char map not found: {char_map_path}")

        with open(char_map_path) as f:
            self.char_to_idx = json.load(f)

        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.blank_idx = len(self.char_to_idx)

        # ---- CONFIG ----
        self.img_height = 32
        self.max_width = 128

        print("✅ CursiveICREngine loaded (experimental)")

    # --------------------------------------------------
    # PREPROCESS (same as training / testing)
    # --------------------------------------------------

    def _preprocess(self, img):
        """
        img: BGR or grayscale image
        returns: (1, H, W, 1)
        """
        if img is None:
            return None

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = img.shape
        scale = self.img_height / h
        new_w = min(int(w * scale), self.max_width)

        img = cv2.resize(img, (new_w, self.img_height))

        canvas = np.ones((self.img_height, self.max_width), dtype=np.uint8) * 255
        canvas[:, :new_w] = img

        img = canvas.astype("float32") / 255.0
        return img[np.newaxis, ..., np.newaxis]

    # --------------------------------------------------
    # CTC DECODE (greedy, same as eval script)
    # --------------------------------------------------

    def _decode(self, preds):
        decoded, _ = tf.keras.backend.ctc_decode(
            preds,
            input_length=[preds.shape[1]],
            greedy=True
        )

        idxs = decoded[0][0].numpy()
        chars = [self.idx_to_char[i] for i in idxs if i != -1]

        return "".join(chars)

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def predict_word(self, image):
        """
        Predict a single cursive word image.
        """
        x = self._preprocess(image)
        if x is None:
            return {"text": "", "confidence": 0.0}

        preds = self.model.predict(x, verbose=0)
        text = self._decode(preds)

        # Confidence is unreliable for CTC now → keep low
        return {
            "text": text,
            "confidence": 0.4
        }

    def predict_paragraph(self, image):
        """
        Predict cursive handwriting from a paragraph image.
        Uses line → word segmentation + CRNN word inference.
        """

        lines = segment_lines(image)
        all_lines = []
    
        for line_img in lines:
            words = segment_cursive_words(line_img)
    
            line_words = []
            for word_img in words:
                result = self.predict_word(word_img)
                line_words.append(result["text"])
    
            all_lines.append(" ".join(line_words))
    
        final_text = "\n".join(all_lines)
    
        return {
            "text": final_text,
           "confidence": 0.4  # conservative, experimental
        }

