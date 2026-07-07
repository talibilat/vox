"""Local faster-whisper STT backend (the offline default)."""

from __future__ import annotations

import numpy as np

from earshot.stt.base import SttBackend


class LocalWhisperBackend(SttBackend):
    def __init__(self, model: str = "base.en", device: str = "cpu", compute_type: str = "int8"):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != 16000:
            raise ValueError(f"faster-whisper expects 16kHz audio, got {sample_rate}")
        float_audio = audio.astype(np.float32) / 32768.0
        segments, _info = self._model.transcribe(float_audio, beam_size=1, language="en")
        return " ".join(segment.text.strip() for segment in segments).strip()
