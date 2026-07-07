"""The TTS backend interface every implementation satisfies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np


class TtsBackend(ABC):
    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""

    @abstractmethod
    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        """Yield int16 mono audio chunks for the text as they are ready."""
