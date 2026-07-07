"""The STT backend interface every implementation satisfies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SttBackend(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe int16 mono audio to text. Returns '' for silence."""
