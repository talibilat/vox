"""Speech-onset detection for barge-in.

Runs Silero VAD (openWakeWord's bundled ONNX build, same engine as
end-of-speech detection) over mic frames while the agent is speaking, and
fires the moment sustained user speech appears.

Latency budget (the 200ms target, from the P0-03 numbers):
- mic frames arrive every 80ms (1280 samples at 16kHz);
- each frame is scored as two 640-sample sub-chunks, and the onset fires on
  ONSET_PATIENCE consecutive high sub-chunks, so a decision can complete
  within a single frame when speech spans it;
- worst case is speech starting right after a frame boundary and split
  across two frames: ~160ms of audio time, leaving ~40ms for the playback
  stop, which small-slice playback writes satisfy.

Echo discrimination, MVP assumptions (documented per the issue): this
detector does no acoustic echo cancellation. It assumes the user wears
headphones (the plan's stated assumption) or keeps speaker volume modest,
plus the sub-chunk patience requirement so brief playback bleed does not
fire. Real-noise tuning and speaker-mode hardening are #15.
"""

from __future__ import annotations

import collections

import numpy as np

from earshot.openwakeword_resources import (
    download_openwakeword_resources,
    is_missing_openwakeword_resource,
)

SUB_CHUNK = 640  # 40ms at 16kHz; divides the 1280-sample mic frame exactly
ONSET_PATIENCE = 2  # consecutive high sub-chunks (80ms of speech evidence)
PRE_ROLL_FRAMES = 8  # ~640ms of audio kept so the utterance start is not lost


class SpeechOnsetDetector:
    def __init__(self, threshold: float = 0.5):
        from openwakeword.vad import VAD

        try:
            self._vad = VAD()
        except Exception as exc:
            if not is_missing_openwakeword_resource(exc):
                raise
            download_openwakeword_resources()
            self._vad = VAD()
        self._threshold = threshold
        self._streak = 0
        self._pre_roll: collections.deque[np.ndarray] = collections.deque(maxlen=PRE_ROLL_FRAMES)

    def reset(self) -> None:
        self._vad.reset_states()
        self._streak = 0
        self._pre_roll.clear()

    def onset(self, frame: np.ndarray) -> bool:
        """Feed one mic frame; True when sustained speech has just started."""
        self._pre_roll.append(frame)
        for start in range(0, len(frame), SUB_CHUNK):
            score = self._vad.predict(frame[start : start + SUB_CHUNK], frame_size=SUB_CHUNK)
            if score >= self._threshold:
                self._streak += 1
                if self._streak >= ONSET_PATIENCE:
                    return True
            else:
                self._streak = 0
        return False

    def pre_roll(self) -> list[np.ndarray]:
        """The most recent frames, so recording can include the onset audio."""
        return list(self._pre_roll)
