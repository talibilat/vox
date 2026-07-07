"""API STT backend: an OpenAI-compatible /audio/transcriptions endpoint.

Same interface as the local faster-whisper backend, so callers cannot tell
the difference. The API key is read from the environment variable named in
config (never a literal key in YAML). With fallback_to_local enabled, a
request failure transparently degrades to the local backend (created
lazily, so the whisper model only loads if it is ever needed).
"""

from __future__ import annotations

import io
import json
import logging
import os
import urllib.error
import urllib.request
import uuid
import wave

import numpy as np

from earshot.stt.base import SttBackend

logger = logging.getLogger("earshot.stt.api")

REQUEST_TIMEOUT = 60.0


class BackendUnavailable(Exception):
    """The API backend cannot serve the request; message is actionable."""


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(audio.tobytes())
    return buffer.getvalue()


def _multipart(fields: dict[str, str], file_field: str, filename: str, payload: bytes):
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        header = f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
        parts.append(f"{header}\r\n\r\n{value}\r\n".encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
        f'filename="{filename}"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
    )
    parts.append(payload)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def read_api_key(api_key_env: str) -> str:
    key = os.environ.get(api_key_env, "")
    if not key:
        raise BackendUnavailable(
            f"the API key environment variable {api_key_env} is not set; "
            "export it or switch the backend to 'local'"
        )
    return key


class ApiSttBackend(SttBackend):
    def __init__(
        self,
        base_url: str,
        api_key_env: str,
        model: str,
        fallback: SttBackend | None = None,
        fallback_factory=None,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self._model = model
        self._fallback = fallback
        self._fallback_factory = fallback_factory

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        try:
            return self._transcribe_remote(audio, sample_rate)
        except BackendUnavailable as error:
            fallback = self._get_fallback()
            if fallback is None:
                raise
            logger.warning("API STT failed (%s); falling back to local", error)
            return fallback.transcribe(audio, sample_rate)

    def _transcribe_remote(self, audio: np.ndarray, sample_rate: int) -> str:
        body, content_type = _multipart(
            {"model": self._model, "response_format": "json"},
            "file",
            "speech.wav",
            _wav_bytes(audio, sample_rate),
        )
        request = urllib.request.Request(
            f"{self._base_url}/audio/transcriptions",
            data=body,
            headers={
                "Content-Type": content_type,
                "Authorization": f"Bearer {read_api_key(self._api_key_env)}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read()[:200].decode("utf-8", "replace")
            raise BackendUnavailable(
                f"STT API returned HTTP {error.code}: {detail or error.reason}"
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise BackendUnavailable(f"STT API is unreachable: {error}") from error
        except json.JSONDecodeError as error:
            raise BackendUnavailable("STT API returned malformed JSON") from error
        text = payload.get("text")
        if not isinstance(text, str):
            raise BackendUnavailable("STT API response had no 'text' field")
        return text.strip()

    def _get_fallback(self) -> SttBackend | None:
        if self._fallback is None and self._fallback_factory is not None:
            try:
                self._fallback = self._fallback_factory()
            except Exception:
                logger.exception("could not create the local STT fallback")
                self._fallback_factory = None
        return self._fallback
