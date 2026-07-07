"""End-of-speech detection for the record-after-wake flow.

Uses openWakeWord's bundled Silero VAD (ONNX, no torch dependency): while
recording, the utterance ends once VAD has reported silence for
`silence_ms` continuously, after speech was heard at least once.
"""

from __future__ import annotations

import numpy as np

from earshot.audio import FRAME_SAMPLES, SAMPLE_RATE
from earshot.openwakeword_resources import (
    download_openwakeword_resources,
    is_missing_openwakeword_resource,
)

FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE


class EndOfSpeechDetector:
    def __init__(
        self,
        threshold: float = 0.5,
        silence_ms: int = 800,
        max_utterance_ms: int = 30_000,
    ):
        from openwakeword.vad import VAD

        try:
            self._vad = VAD()
        except Exception as exc:
            if not is_missing_openwakeword_resource(exc):
                raise
            download_openwakeword_resources()
            self._vad = VAD()
        self._threshold = threshold
        self._silence_frames_needed = max(1, silence_ms // FRAME_MS)
        self._max_frames = max(1, max_utterance_ms // FRAME_MS)
        self.reset()

    def reset(self) -> None:
        self._vad.reset_states()
        self._heard_speech = False
        self._silent_frames = 0
        self._total_frames = 0

    def finished(self, frame: np.ndarray) -> bool:
        """Feed one frame; True once the utterance is over."""
        self._total_frames += 1
        # 640-sample sub-frames divide the 1280-sample frame exactly.
        score = self._vad.predict(frame, frame_size=640)
        if score >= self._threshold:
            self._heard_speech = True
            self._silent_frames = 0
        else:
            self._silent_frames += 1
        if self._total_frames >= self._max_frames:
            return True
        return self._heard_speech and self._silent_frames >= self._silence_frames_needed
