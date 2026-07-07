"""Playback and output pipeline tests using fake sinks and fake TTS, so no
audio device or voice model is needed. A separate Piper test exercises the
real engine when installed.
"""

import importlib.util
import threading
import time
from collections.abc import Iterator

import numpy as np
import pytest

from earshot.audio.playback import Player
from earshot.config import Config
from earshot.output import OutputPipeline
from earshot.tts import create_backend


class FakeSink:
    """Collects written samples; can simulate real-time pacing."""

    def __init__(self, delay_per_write: float = 0.0):
        self.written: list[np.ndarray] = []
        self.aborted = 0
        self._delay = delay_per_write

    def write(self, samples):
        if self._delay:
            time.sleep(self._delay)
        self.written.append(np.asarray(samples))

    def abort(self):
        self.aborted += 1

    def close(self):
        pass

    def total_samples(self):
        return sum(len(w) for w in self.written)


class FakeTts:
    """Yields one marker chunk per synthesize() call and records timing."""

    sample_rate = 16000

    def __init__(self):
        self.calls: list[str] = []
        self.call_times: list[float] = []

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        self.calls.append(text)
        self.call_times.append(time.monotonic())
        yield np.full(1600, len(self.calls), dtype=np.int16)


class TestPlayer:
    def test_plays_everything_when_uninterrupted(self):
        sink = FakeSink()
        player = Player(sink)
        player.enqueue(np.ones(5000, dtype=np.int16))
        assert player.wait_until_idle(timeout=5)
        assert sink.total_samples() == 5000
        player.close()

    def test_stop_and_flush_halts_and_clears(self):
        sink = FakeSink(delay_per_write=0.01)
        player = Player(sink)
        # ~100 slices of 1024 at 10ms per write = ~1s of work
        for _ in range(10):
            player.enqueue(np.ones(10240, dtype=np.int16))
        time.sleep(0.05)  # let a few slices play
        player.stop_and_flush()
        written_at_stop = sink.total_samples()
        assert written_at_stop < 10 * 10240, "stop did not halt playback early"
        assert sink.aborted >= 1, "sink was never told to drop buffered audio"
        time.sleep(0.05)
        assert sink.total_samples() == written_at_stop, "audio kept playing after stop"
        player.close()

    def test_playable_again_after_stop(self):
        sink = FakeSink()
        player = Player(sink)
        player.enqueue(np.ones(2048, dtype=np.int16))
        player.stop_and_flush()
        before = sink.total_samples()
        player.enqueue(np.full(1024, 7, dtype=np.int16))
        assert player.wait_until_idle(timeout=5)
        assert sink.total_samples() == before + 1024
        player.close()

    def test_stop_drops_synthesis_that_yields_after_interrupt(self):
        sink = FakeSink()
        player = Player(sink)
        release = threading.Event()

        def slow_chunks():
            release.wait(timeout=5)
            yield np.ones(1024, dtype=np.int16)

        thread = threading.Thread(target=lambda: player.enqueue(slow_chunks()))
        thread.start()

        player.stop_and_flush()
        release.set()
        thread.join(timeout=5)
        time.sleep(0.05)

        assert sink.total_samples() == 0
        player.close()


class TestOutputPipeline:
    def make(self, code_blocks="summarize"):
        config = Config()
        config.code_blocks = code_blocks
        sink = FakeSink()
        tts = FakeTts()
        pipeline = OutputPipeline(config, player=Player(sink), tts=tts)
        return pipeline, tts, sink

    def test_streamed_markdown_is_spoken_clean(self):
        pipeline, tts, sink = self.make()
        stream = ["## Result", "s\n\nAll **42** tests pass. ", "The refactor is done.\n"]
        pipeline.speak_stream(stream)
        assert pipeline.wait_until_idle(timeout=5)
        spoken = " ".join(tts.calls)
        assert "#" not in spoken and "*" not in spoken
        assert "42 tests pass" in spoken
        assert sink.total_samples() == 1600 * len(tts.calls)

    def test_first_sentence_synthesized_before_stream_ends(self):
        pipeline, tts, _sink = self.make()

        first_sentence_done = threading.Event()
        seen_by_generator = []

        def stream():
            yield "First sentence is complete. "
            # Give the pipeline a beat to consume what was yielded.
            deadline = time.monotonic() + 2
            while not tts.calls and time.monotonic() < deadline:
                time.sleep(0.01)
            seen_by_generator.append(list(tts.calls))
            first_sentence_done.set()
            yield "Second sentence arrives much later."

        pipeline.speak_stream(stream())
        assert first_sentence_done.is_set()
        assert seen_by_generator[0], "first sentence was not synthesized before the stream ended"
        assert "First sentence is complete." in seen_by_generator[0][0]

    def test_code_block_modes(self):
        md = "Look:\n\n```python\nprint(1)\nprint(2)\n```\n\nDone.\n"
        for mode, must_contain, must_not in [
            ("summarize", "2 lines python code block", "print"),
            ("skip", "Done", "print"),
            ("read", "print(1)", None),
        ]:
            pipeline, tts, _ = self.make(code_blocks=mode)
            pipeline.speak_stream([md])
            spoken = " ".join(tts.calls)
            assert must_contain in spoken
            if must_not:
                assert must_not not in spoken

    def test_fence_split_across_stream_chunks(self):
        pipeline, tts, _ = self.make()
        pipeline.speak_stream(["Before.\n\n```py\nx", " = 1\n``", "`\n\nAfter.\n"])
        spoken = " ".join(tts.calls)
        assert "Before." in spoken
        assert "x = 1" not in spoken
        assert "1 line py code block" in spoken
        assert "After." in spoken

    def test_table_without_leading_pipe_is_buffered_until_complete(self):
        pipeline, tts, _ = self.make()
        pipeline.speak_stream(["Status | Notes\n--- | ---\nOK. | Done.\n\nAfter.\n"])

        assert all("|" not in call for call in tts.calls)
        assert "OK" in " ".join(tts.calls)
        assert "After." in " ".join(tts.calls)

    def test_stop_and_flush_passthrough(self):
        pipeline, _tts, sink = self.make()
        pipeline.speak("A sentence to interrupt.")
        pipeline.stop_and_flush()
        settled = sink.total_samples()
        time.sleep(0.05)
        assert sink.total_samples() == settled


def test_create_backend_rejects_unimplemented_local_engine():
    config = Config()
    config.tts.local.engine = "kokoro"

    with pytest.raises(NotImplementedError, match="kokoro"):
        create_backend(config)


@pytest.mark.skipif(importlib.util.find_spec("piper") is None, reason="piper not installed")
class TestPiperBackend:
    def test_synthesizes_offline(self, tmp_path):
        from earshot.tts.local_piper import VOICES_DIR, PiperBackend

        voices_dir = VOICES_DIR if (VOICES_DIR / "en_US-lessac-medium.onnx").exists() else None
        backend = PiperBackend(voice="en_US-lessac-medium", voices_dir=voices_dir)
        chunks = list(backend.synthesize("Hello from Earshot."))
        assert chunks and all(c.dtype == np.int16 for c in chunks)
        assert sum(len(c) for c in chunks) > backend.sample_rate // 4  # >0.25s of audio
