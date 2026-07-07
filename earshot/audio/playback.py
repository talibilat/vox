"""Audio playback with an interruption-first design.

`Player` runs synthesis-to-speaker on a worker thread and exposes
`stop_and_flush()`: the hook the barge-in subsystem (#7) calls. Stopping
must be near-instant, so audio is written to the sink in small slices and
the stop flag is checked between slices; the queue and any pending
synthesis are dropped on the floor.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

import numpy as np

# Slice size for sink writes, in samples. At 22050Hz, 1024 samples is ~46ms,
# which bounds how long a stop can lag behind the flag check.
_SLICE = 1024


class SounddeviceSink:
    """Speaker output via sounddevice (lazy import, like MicSource)."""

    def __init__(self, sample_rate: int):
        import sounddevice

        self._stream = sounddevice.OutputStream(samplerate=sample_rate, channels=1, dtype="int16")
        self._stream.start()

    def write(self, samples: np.ndarray) -> None:
        self._stream.write(samples.reshape(-1, 1))

    def abort(self) -> None:
        self._stream.abort()  # drop buffered audio immediately
        self._stream.start()

    def close(self) -> None:
        self._stream.abort()
        self._stream.close()


class Player:
    """Plays queued audio chunks; interruptible between small slices."""

    def __init__(self, sink):
        self._sink = sink
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._interrupt = threading.Event()
        self._lock = threading.Lock()
        self._generation = 0
        self._idle = threading.Event()
        self._idle.set()
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="playback")
        self._thread.start()

    def enqueue(self, chunks: Iterator[np.ndarray] | np.ndarray) -> None:
        """Queue audio for playback. Accepts one array or an iterator."""
        with self._lock:
            if self._interrupt.is_set():
                return
            generation = self._generation
        if isinstance(chunks, np.ndarray):
            chunks = iter([chunks])
        for chunk in chunks:
            with self._lock:
                stale = generation != self._generation or self._interrupt.is_set()
            if stale:
                return  # synthesis backlog is dropped during an interrupt
            self._idle.clear()
            self._queue.put(chunk)

    def stop_and_flush(self) -> None:
        """Halt current audio and drop everything queued. The barge-in hook."""
        with self._lock:
            self._generation += 1
            self._interrupt.set()
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        self._sink.abort()
        # Wait for the worker to acknowledge before accepting new audio, so
        # a chunk played right after an interrupt cannot be a stale one.
        self._idle.wait(timeout=2)
        with self._lock:
            self._interrupt.clear()

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        """Block until everything queued has been played."""
        return self._idle.wait(timeout=timeout)

    def close(self) -> None:
        self._closed = True
        self._interrupt.set()
        self._queue.put(None)
        self._thread.join(timeout=5)
        self._sink.close()

    def _run(self) -> None:
        while True:
            chunk = self._queue.get()
            if chunk is None:
                return
            if self._interrupt.is_set():
                if self._queue.empty():
                    self._idle.set()
                continue
            for start in range(0, len(chunk), _SLICE):
                if self._interrupt.is_set():
                    break
                self._sink.write(chunk[start : start + _SLICE])
            if self._queue.empty():
                self._idle.set()
