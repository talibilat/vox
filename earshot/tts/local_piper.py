"""Local Piper TTS backend (the offline default).

Piper won the Phase 0 latency spike by an order of magnitude (46ms first
audio; see docs/latency-spike.md), so it is the local engine. Kokoro remains
a possible opt-in quality voice later; the interface does not care.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from earshot.tts.base import TtsBackend

logger = logging.getLogger("earshot.tts")

VOICES_DIR = Path("~/.local/share/earshot/voices").expanduser()


def _ensure_voice(voice: str, voices_dir: Path) -> Path:
    """Return the voice model path, downloading it on first use."""
    model_path = voices_dir / f"{voice}.onnx"
    if model_path.exists():
        return model_path
    voices_dir.mkdir(parents=True, exist_ok=True)
    logger.info("downloading piper voice %s to %s", voice, voices_dir)
    from piper.download_voices import download_voice

    download_voice(voice, voices_dir)
    return model_path


class PiperBackend(TtsBackend):
    def __init__(self, voice: str = "en_US-lessac-medium", speed: float = 1.0, voices_dir=None):
        from piper import PiperVoice, SynthesisConfig

        model_path = _ensure_voice(voice, Path(voices_dir) if voices_dir else VOICES_DIR)
        self._voice = PiperVoice.load(str(model_path))
        # Piper's length_scale is inverse speed: 2.0 is half speed.
        self._config = SynthesisConfig(length_scale=1.0 / speed)

    @property
    def sample_rate(self) -> int:
        return self._voice.config.sample_rate

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        for chunk in self._voice.synthesize(text, syn_config=self._config):
            yield np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
