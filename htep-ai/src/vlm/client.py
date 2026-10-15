# src/vlm/client.py
"""
Vision-language model (VLM) client for reading prescriptions.

The VLM runs as a separate service so the web backend stays light: point
VLM_URL at an Ollama server (default API) or any OpenAI-compatible server
such as vLLM, ideally on a machine with a GPU. With VLM_URL unset, the
pipeline uses OCR and rules only.

The model is asked for the prescription fields as JSON that matches
PRESCRIPTION_SCHEMA. Its answers are never trusted on their own: the
digitizer checks each value against the OCR text (src/pipeline/grounding.py)
and flags the ones it cannot find for human review.
"""

import base64
import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

import cv2
import numpy as np

from src.config import VLM_API, VLM_MAX_IMAGE_SIDE, VLM_MODEL, VLM_TIMEOUT_S, VLM_URL

logger = logging.getLogger(__name__)

_NULLABLE_STRING = {"type": ["string", "null"]}

PRESCRIPTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "patient_name": _NULLABLE_STRING,
        "age": _NULLABLE_STRING,
        "gender": _NULLABLE_STRING,
        "date": _NULLABLE_STRING,
        "doctor_name": _NULLABLE_STRING,
        "hospital": _NULLABLE_STRING,
        "diagnosis": {"type": "array", "items": {"type": "string"}},
        "medicines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "dosage": _NULLABLE_STRING,
                    "frequency": _NULLABLE_STRING,
                    "duration": _NULLABLE_STRING,
                },
                "required": ["name"],
            },
        },
        "instructions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["patient_name", "medicines"],
}

SYSTEM_PROMPT = (
    "You read medical prescriptions. Copy every value exactly as it is printed "
    "or handwritten, keeping the original spelling and units. Use null for a "
    "field that is not visible. Never guess, never correct, and never add "
    "medicines or diagnoses that are not written in the image."
)

USER_PROMPT = "Extract the prescription fields from this image."


class VLMClient:
    """Calls a remote VLM. Every method returns None instead of raising."""

    def __init__(
        self,
        url: str = VLM_URL,
        model: str = VLM_MODEL,
        api: str = VLM_API,
        timeout_s: float = VLM_TIMEOUT_S,
        max_image_side: int = VLM_MAX_IMAGE_SIDE,
    ):
        self.url = (url or "").rstrip("/")
        self.model = model
        self.api = api
        self.timeout_s = timeout_s
        self.max_image_side = max_image_side

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def is_reachable(self) -> bool:
        if not self.enabled:
            return False
        path = "/api/tags" if self.api == "ollama" else "/v1/models"
        try:
            with urllib.request.urlopen(self.url + path, timeout=3):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def extract_prescription(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Read the prescription fields from a BGR image. None on any failure."""
        if not self.enabled or image is None:
            return None

        image_b64 = self._encode_image(image)
        t0 = time.time()
        try:
            if self.api == "openai":
                content = self._chat_openai(image_b64)
            else:
                content = self._chat_ollama(image_b64)
            data = json.loads(content)
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            logger.warning("VLM request failed after %.1fs: %s", time.time() - t0, e)
            return None

        logger.info("VLM extraction took %.1fs", time.time() - t0)
        return data if isinstance(data, dict) else None

    # ------------------------------------------------------------------

    def _encode_image(self, image: np.ndarray) -> str:
        # Image tokens grow with resolution; a capped side keeps the request
        # fast and within the model's context without losing printed text.
        h, w = image.shape[:2]
        scale = self.max_image_side / max(h, w)
        if scale < 1:
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("Could not encode image for the VLM")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.load(response)

    def _chat_ollama(self, image_b64: str) -> str:
        body = {
            "model": self.model,
            "stream": False,
            "format": PRESCRIPTION_SCHEMA,
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 1024},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT, "images": [image_b64]},
            ],
        }
        return self._post("/api/chat", body)["message"]["content"]

    def _chat_openai(self, image_b64: str) -> str:
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 1024,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "prescription", "schema": PRESCRIPTION_SCHEMA},
            },
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                },
            ],
        }
        return self._post("/v1/chat/completions", body)["choices"][0]["message"]["content"]
