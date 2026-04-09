import cv2
import tempfile
import pathlib
import ollama

class LlavaICREngine:
    """
    LLaVa-based ICR engine using the local Ollama API.
    Used for inferring text directly from image snippets via vision-language models.
    """

    def __init__(self, model_name="llava"):
        """
        Initializes the LLaVa engine.
        :param model_name: The tag used in Ollama (default is 'llava', could be 'llava:13b')
        """
        self.model_name = model_name
        self._system_prompt = (
            "You are an OCR transcriber for handwritten medical notes. "
            "Transcribe text exactly as written in the image. "
            "Do not summarize, explain, or correct grammar/spelling. "
            "Keep original line breaks when visible. "
            "Return only plain transcribed text, no labels and no markdown."
        )
        print(f"✅ LlavaICREngine initialized (Using Ollama tag: {self.model_name})")

    def _preprocess(self, img):
        """
        Build OCR-friendly image variants to improve transcription reliability.
        """
        if img is None:
            return []

        # Resize if huge to prevent OOM on the LLM
        h, w = img.shape[:2]
        max_dim = 1400
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))

        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Trim empty borders.
        ink = cv2.findNonZero((255 - gray))
        if ink is not None:
            x, y, w_box, h_box = cv2.boundingRect(ink)
            gray = gray[y:y + h_box, x:x + w_box]

        # Upscale small text for better VLM readability.
        if max(gray.shape[:2]) < 900:
            gray = cv2.resize(gray, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)

        # Variant 1: normalized grayscale
        gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Variant 2: adaptive threshold to emphasize pen strokes
        gray_blur = cv2.GaussianBlur(gray_norm, (3, 3), 0)
        bw = cv2.adaptiveThreshold(
            gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
        )

        return [gray_norm, bw]

    def _extract_response_text(self, response) -> str:
        # ollama-python can return dict-like or typed objects depending on version.
        message = None
        if isinstance(response, dict):
            message = response.get("message", {})
            content = message.get("content", "")
        else:
            message = getattr(response, "message", None)
            content = getattr(message, "content", "") if message is not None else ""

        if isinstance(content, list):
            content = "".join(str(part) for part in content)

        return str(content).strip()

    def _clean_text(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned.strip("`").strip()
        cleaned = cleaned.replace("Transcription:", "").replace("OCR:", "").strip()
        return cleaned

    def _run_ocr(self, image_path: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": self._system_prompt
                },
                {
                    "role": "user",
                    "content": (
                        "Read this image and return exact visible text only. "
                        "Preserve line breaks. If unreadable, return best effort characters."
                    ),
                    "images": [image_path]
                }
            ],
            options={
                "temperature": 0,
                "top_p": 0.1,
                "repeat_penalty": 1.05,
                "seed": 7
            }
        )
        return self._clean_text(self._extract_response_text(response))

    def predict_paragraph(self, img) -> dict:
        """
        Predict handwritten text from a paragraph image directly.
        Returns a dict: {"text": "extracted text", "confidence": <mock_val>}
        """
        variants = self._preprocess(img)
        if not variants:
            return {"text": "", "confidence": 0.0}

        try:
            candidates = []
            for variant in variants:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    tmp_path = tmp_file.name

                try:
                    cv2.imwrite(tmp_path, variant)
                    candidate = self._run_ocr(tmp_path)
                    candidates.append(candidate)
                finally:
                    pathlib.Path(tmp_path).unlink(missing_ok=True)

            predicted_text = max(candidates, key=lambda t: len(t.strip())) if candidates else ""
            if not predicted_text:
                print("⚠️ LLaVa returned an empty response")

            return {
                "text": predicted_text,
                "confidence": 0.55  # heuristic
            }

        except ollama.ResponseError as e:
            print(f"⚠️ Ollama model error (Did you pull the '{self.model_name}' model?):", e)
            return {"text": "", "confidence": 0.0}
        except Exception as e:
            print(f"⚠️ LLaVa ICR crashed:", e)
            return {"text": "", "confidence": 0.0}
