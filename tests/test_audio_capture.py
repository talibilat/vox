import queue
import sys
import threading
from types import SimpleNamespace

import numpy as np

from earshot.audio import FRAME_SAMPLES
from earshot.audio.capture import MicSource


def test_mic_source_drops_oldest_frame_when_queue_is_full():
    source = MicSource()
    source._queue = queue.Queue(maxsize=2)

    oldest = np.full(FRAME_SAMPLES, 1, dtype=np.int16)
    stale = np.full(FRAME_SAMPLES, 2, dtype=np.int16)
    fresh = np.full(FRAME_SAMPLES, 3, dtype=np.int16)

    source._queue.put_nowait(oldest)
    source._queue.put_nowait(stale)
    source._enqueue_frame(fresh.tobytes())

    assert np.array_equal(source._queue.get_nowait(), stale)
    assert np.array_equal(source._queue.get_nowait(), fresh)


def test_mic_source_stop_unblocks_frames_when_no_audio_arrives(monkeypatch):
    entered = threading.Event()

    class FakeRawInputStream:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            entered.set()
            return self

        def __exit__(self, *_exc_info):
            return False

    monkeypatch.setitem(
        sys.modules, "sounddevice", SimpleNamespace(RawInputStream=FakeRawInputStream)
    )

    source = MicSource()
    stopped = threading.Event()

    def consume_frames():
        for _frame in source.frames():
            pass
        stopped.set()

    thread = threading.Thread(target=consume_frames, daemon=True)
    thread.start()

    assert entered.wait(timeout=1)
    source.stop()

    assert stopped.wait(timeout=1)
