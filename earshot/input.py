"""The audio input pipeline: wake word -> record -> STT -> on_transcript.

Consumes frames from any AudioSource, waits for the wake word, records until
end-of-speech, transcribes with the configured backend, and delivers plain
text to the `on_transcript` hook. The agent loop (#8) and name parsing (#12)
build on that hook without knowing anything about audio.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator

import numpy as np

from earshot.audio import SAMPLE_RATE
from earshot.audio.endpointing import EndOfSpeechDetector
from earshot.config import Config
from earshot.stt import SttBackend, create_backend
from earshot.wakeword import WakeWordDetector

logger = logging.getLogger("earshot.input")


class InputPipeline:
    def __init__(
        self,
        config: Config,
        on_transcript: Callable[[str], None],
        source=None,
        stt: SttBackend | None = None,
    ):
        if not config.wake_word.model_path:
            raise ValueError(
                "wake_word.model_path is not set; the input pipeline needs a trained model"
            )
        if source is None:
            from earshot.audio.capture import MicSource

            source = MicSource()
        self._source = source
        self._on_transcript = on_transcript
        self._detector = WakeWordDetector(
            model_path=config.wake_word.model_path,
            sensitivity=config.wake_word.sensitivity,
            patience=config.wake_word.patience,
        )
        self._end_of_speech = EndOfSpeechDetector(threshold=config.barge_in.vad_threshold)
        self._stt = stt if stt is not None else create_backend(config)
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        """Blocking loop; call stop() (from another thread) to exit."""
        frames = self._source.frames()
        while not self._stop.is_set():
            if not self._listen_for_wake(frames):
                return
            logger.info("wake word detected, recording")
            utterance = self._record_utterance(frames)
            if utterance is None:
                return
            text = self._stt.transcribe(utterance, SAMPLE_RATE)
            if text:
                logger.info("transcript: %s", text)
                self._on_transcript(text)
            else:
                logger.info("no speech transcribed after wake")

    def _listen_for_wake(self, frames: Iterator[np.ndarray]) -> bool:
        """True when the wake word fires; False when the source is exhausted."""
        self._detector.reset()
        for frame in frames:
            if self._stop.is_set():
                return False
            if self._detector.detected(frame):
                return True
        return False

    def _record_utterance(self, frames: Iterator[np.ndarray]) -> np.ndarray | None:
        self._end_of_speech.reset()
        recorded: list[np.ndarray] = []
        for frame in frames:
            if self._stop.is_set():
                return None
            recorded.append(frame)
            if self._end_of_speech.finished(frame):
                return np.concatenate(recorded)
        # Source exhausted (file/test input): treat what we have as the clip.
        return np.concatenate(recorded) if recorded else None
