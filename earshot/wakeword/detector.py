"""openWakeWord wrapper with the patience-based trigger rule.

The Phase 0 spike (docs/latency-spike.md) showed that firing on a single
high-scoring window false-triggers heavily, while requiring `patience`
consecutive windows above the threshold cuts false positives by an order of
magnitude. That rule lives here so every caller gets it.
"""

from __future__ import annotations

import numpy as np


class WakeWordDetector:
    def __init__(self, model_path: str, sensitivity: float = 0.95, patience: int = 4):
        from openwakeword.model import Model

        self._model = Model(wakeword_models=[model_path], inference_framework="onnx")
        self._name = next(iter(self._model.models))
        self._sensitivity = sensitivity
        self._patience = max(1, patience)
        self._streak = 0

    def reset(self) -> None:
        self._model.reset()
        self._streak = 0

    def detected(self, frame: np.ndarray) -> bool:
        """Feed one 16kHz int16 frame; True when the wake word fires."""
        score = self._model.predict(frame)[self._name]
        if score >= self._sensitivity:
            self._streak += 1
        else:
            self._streak = 0
        if self._streak >= self._patience:
            self._streak = 0
            return True
        return False
