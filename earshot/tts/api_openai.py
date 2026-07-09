"""API TTS backend: an OpenAI-compatible /audio/speech endpoint.

Same interface as the local Piper backend. Audio is requested as raw PCM
(`response_format: "pcm"`, 24kHz mono 16-bit per the OpenAI contract) and
streamed chunk-by-chunk into the existing playback path, so first audio
does not wait for the full synthesis. With fallback_to_local enabled, a
request failure transparently degrades to the local engine.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Iterator

import numpy as np

from earshot.api_fallback import get_or_create_fallback
from earshot.stt.api_openai import BackendUnavailable, read_api_key
from earshot.tts.base import TtsBackend

logger = logging.getLogger("earshot.tts.api")

REQUEST_TIMEOUT = 60.0
PCM_SAMPLE_RATE = 24000  # the OpenAI /audio/speech pcm format
_CHUNK_BYTES = 4096


class ApiTtsBackend(TtsBackend):
    def __init__(
        self,
        base_url: str,
        api_key_env: str,
        model: str,
        voice: str,
        fallback: TtsBackend | None = None,
        fallback_factory=None,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self._model = model
        self._voice = voice
        self._fallback = fallback
        self._fallback_factory = fallback_factory

    @property
    def sample_rate(self) -> int:
        # The playback stream is opened for this rate, so the fallback's
        # audio is resampled to match (see _resampled).
        return PCM_SAMPLE_RATE

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        try:
            yield from self._synthesize_remote(text)
        except BackendUnavailable as error:
            fallback = self._get_fallback()
            if fallback is None:
                raise
            logger.warning("API TTS failed (%s); falling back to local", error)
            yield from self._resampled(fallback, text)

    def _synthesize_remote(self, text: str) -> Iterator[np.ndarray]:
        request = urllib.request.Request(
            f"{self._base_url}/audio/speech",
            data=json.dumps(
                {
                    "model": self._model,
                    "voice": self._voice,
                    "input": text,
                    "response_format": "pcm",
                }
            ).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {read_api_key(self._api_key_env)}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                carry = b""
                while True:
                    chunk = response.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    data = carry + chunk
                    usable = len(data) - (len(data) % 2)  # int16 alignment
                    carry = data[usable:]
                    if usable:
                        yield np.frombuffer(data[:usable], dtype=np.int16)
        except urllib.error.HTTPError as error:
            detail = error.read()[:200].decode("utf-8", "replace")
            raise BackendUnavailable(
                f"TTS API returned HTTP {error.code}: {detail or error.reason}"
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise BackendUnavailable(f"TTS API is unreachable: {error}") from error

    def _resampled(self, fallback: TtsBackend, text: str) -> Iterator[np.ndarray]:
        """Fallback audio, resampled to this backend's advertised rate."""
        from scipy.signal import resample_poly

        source_rate = fallback.sample_rate
        for chunk in fallback.synthesize(text):
            if source_rate == PCM_SAMPLE_RATE:
                yield chunk
            else:
                resampled = resample_poly(chunk.astype(np.float32), PCM_SAMPLE_RATE, source_rate)
                yield resampled.clip(-32768, 32767).astype(np.int16)

    def _get_fallback(self) -> TtsBackend | None:
        self._fallback, self._fallback_factory = get_or_create_fallback(
            self._fallback,
            self._fallback_factory,
            logger,
            "TTS",
        )
        return self._fallback
