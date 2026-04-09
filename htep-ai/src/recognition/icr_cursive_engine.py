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

        self.model = self._load_model(model_path)

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

    def _load_model(self, model_path: Path):
        try:
            return load_model(model_path, compile=False)
        except TypeError as e:
            # Compatibility path for models saved with a Dense config that includes
            # quantization_config, which older keras/tf builds don't accept.
            if "quantization_config" not in str(e):
                raise

            class DenseCompat(tf.keras.layers.Dense):
                def __init__(self, *args, **kwargs):
                    kwargs.pop("quantization_config", None)
                    super().__init__(*args, **kwargs)

            print("⚠️ Loading cursive model with Dense compatibility fallback")
            return load_model(
                model_path,
                compile=False,
                custom_objects={"Dense": DenseCompat}
            )

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

        # Remove empty borders so the network sees tighter text strokes.
        ink = np.where(img < 245)
        if ink[0].size and ink[1].size:
            y0, y1 = int(np.min(ink[0])), int(np.max(ink[0])) + 1
            x0, x1 = int(np.min(ink[1])), int(np.max(ink[1])) + 1
            img = img[y0:y1, x0:x1]

        # Slight denoising + contrast normalization improves cursive legibility.
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        h, w = img.shape
        if h == 0 or w == 0:
            return None

        scale = self.img_height / h
        new_w = max(1, min(int(w * scale), self.max_width))

        img = cv2.resize(img, (new_w, self.img_height))

        canvas = np.ones((self.img_height, self.max_width), dtype=np.uint8) * 255
        canvas[:, :new_w] = img

        img = canvas.astype("float32") / 255.0
        return img[np.newaxis, ..., np.newaxis]

    # --------------------------------------------------
    # CTC DECODE (greedy, same as eval script)
    # --------------------------------------------------

    def _decode(self, preds):
        try:
            decoded, _ = tf.keras.backend.ctc_decode(
                preds,
                input_length=[preds.shape[1]],
                greedy=False,
                beam_width=25,
                top_paths=1
            )
        except Exception:
            decoded, _ = tf.keras.backend.ctc_decode(
                preds,
                input_length=[preds.shape[1]],
                greedy=True
            )

        idxs = decoded[0][0].numpy()
        chars = [self.idx_to_char[i] for i in idxs if i != -1 and i in self.idx_to_char]

        return "".join(chars)

    def _predict_once(self, image):
        x = self._preprocess(image)
        if x is None:
            return {"text": "", "confidence": 0.0}

        preds = self.model.predict(x, verbose=0)
        text = self._decode(preds)

        # Approximate confidence from timestep max probabilities.
        conf = float(np.mean(np.max(preds[0], axis=1))) if preds.size else 0.0
        return {"text": text, "confidence": conf}

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def predict_word(self, image):
        """
        Predict a single cursive word image.
        """
        if image is None:
            return {"text": "", "confidence": 0.0}

        primary = self._predict_once(image)

        # Fallback for dark-on-light inversion mismatch.
        inverted = cv2.bitwise_not(image)
        alt = self._predict_once(inverted)

        best = primary
        if (len(alt["text"]) > len(primary["text"])) or (
            len(alt["text"]) == len(primary["text"]) and alt["confidence"] > primary["confidence"]
        ):
            best = alt

        return best

    def predict_paragraph(self, image):
        """
        Predict cursive handwriting from a paragraph image.
        Uses line → word segmentation + CRNN word inference.
        """

        lines = segment_lines(image)
        all_lines = []
        confs = []

        for line_img in lines:
            words = segment_cursive_words(line_img)

            if not words:
                words = [line_img]

            line_words = []
            for word_img in words:
                result = self.predict_word(word_img)
                if result["text"].strip():
                    line_words.append(result["text"].strip())
                confs.append(result["confidence"])

            all_lines.append(" ".join(line_words))

        final_text = "\n".join(line for line in all_lines if line.strip())
        mean_conf = float(np.mean(confs)) if confs else 0.0

        return {
            "text": final_text,
            "confidence": mean_conf
        }

