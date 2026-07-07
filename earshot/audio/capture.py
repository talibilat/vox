"""Microphone capture behind a swappable frame-source interface.

Everything downstream consumes an iterator of fixed-size int16 mono frames at
16kHz, so tests (and future file/replay inputs) can substitute `ArraySource`
for a real microphone.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

import numpy as np

from earshot.audio import FRAME_SAMPLES, SAMPLE_RATE


class ArraySource:
    """A frame source over an in-memory clip; the test double for MicSource."""

    def __init__(self, audio: np.ndarray):
        if audio.dtype != np.int16:
            raise ValueError(f"expected int16 audio, got {audio.dtype}")
        self._audio = audio

    def frames(self) -> Iterator[np.ndarray]:
        total = len(self._audio) - len(self._audio) % FRAME_SAMPLES
        for start in range(0, total, FRAME_SAMPLES):
            yield self._audio[start : start + FRAME_SAMPLES]


class MicSource:
    """Live microphone frames via sounddevice (PortAudio).

    sounddevice is imported lazily so that machines without PortAudio can
    still import the package and run non-audio code paths.
    """

    def __init__(self, device: int | str | None = None):
        self._device = device
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._stream = None
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def _enqueue_frame(self, indata) -> None:
        frame = np.frombuffer(bytes(indata), dtype=np.int16).copy()
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass

    def frames(self) -> Iterator[np.ndarray]:
        import sounddevice

        self._stop.clear()

        def _on_audio(indata, _frames, _time, status):
            if status:
                pass  # over/underruns are survivable; never raise in the callback
            self._enqueue_frame(indata)

        self._stream = sounddevice.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            channels=1,
            dtype="int16",
            device=self._device,
            callback=_on_audio,
        )
        with self._stream:
            while not self._stop.is_set():
                try:
                    yield self._queue.get(timeout=0.1)
                except queue.Empty:
                    pass
