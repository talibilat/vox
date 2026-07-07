"""Barge-in tests: onset detection, the interrupt state machine, latency
instrumentation, the push-to-interrupt escape hatch, and no-self-interrupt.

The voice-path tests drive the real state machine with fixture audio, the
real Silero VAD, real whisper STT, and the real Player; only the speaker
sink, the TTS engine, and the agent are fakes.
"""

import time
import wave
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openwakeword", reason="voice deps not installed")

from earshot.agents.base import AgentAdapter  # noqa: E402
from earshot.audio import FRAME_SAMPLES, SAMPLE_RATE  # noqa: E402
from earshot.audio.capture import ArraySource  # noqa: E402
from earshot.audio.playback import Player  # noqa: E402
from earshot.barge import InterruptibleVoiceLoop, SpeechOnsetDetector  # noqa: E402
from earshot.config import Config  # noqa: E402
from earshot.output import OutputPipeline  # noqa: E402
from tests.test_output_pipeline import FakeSink, FakeTts  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parent.parent
WAKE_MODEL = REPO_ROOT / "spikes" / "models" / "hey_earshot.onnx"


def read_wav(name: str) -> np.ndarray:
    with wave.open(str(FIXTURES / name)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.int16)


class TestSpeechOnsetDetector:
    def test_fires_shortly_after_speech_starts(self):
        detector = SpeechOnsetDetector()
        audio = np.concatenate([silence(2.0), read_wav("command.wav")])
        fired_at = None
        for i, frame in enumerate(ArraySource(audio).frames()):
            if detector.onset(frame):
                fired_at = i
                break
        assert fired_at is not None, "onset never fired on real speech"
        true_onset_frame = int(2.0 * SAMPLE_RATE) // FRAME_SAMPLES
        lag_ms = (fired_at - true_onset_frame + 1) * FRAME_SAMPLES * 1000 // SAMPLE_RATE
        assert lag_ms <= 240, f"onset audio-time lag {lag_ms}ms is too slow for the budget"

    def test_silent_on_silence(self):
        detector = SpeechOnsetDetector()
        for frame in ArraySource(silence(4.0)).frames():
            assert not detector.onset(frame)

    def test_pre_roll_keeps_recent_frames(self):
        detector = SpeechOnsetDetector()
        frames = list(ArraySource(silence(2.0)).frames())
        for frame in frames:
            detector.onset(frame)
        pre = detector.pre_roll()
        assert 1 <= len(pre) <= 8
        assert all(len(p) == FRAME_SAMPLES for p in pre)


class SlowFakeAdapter(AgentAdapter):
    """Streams a long response slowly so RESPONDING lasts long enough to
    be interrupted."""

    def __init__(self):
        self.prompts = []
        self._alive = True

    def start(self):
        self._alive = True

    def stop(self):
        self._alive = False

    @property
    def alive(self):
        return self._alive

    def send(self, prompt):
        self.prompts.append(prompt)
        for i in range(50):
            yield f"This is sentence number {i} of a very long answer. "
            time.sleep(0.02)


def make_loop(source_audio, stt=None):
    config = Config()
    config.wake_word.model_path = str(WAKE_MODEL)
    config.wake_word.sensitivity = 0.9  # fixture margins; see test_input_pipeline
    config.wake_word.patience = 3

    adapter = SlowFakeAdapter()
    sink = FakeSink(delay_per_write=0.002)
    tts = FakeTts()
    output = OutputPipeline(config, player=Player(sink), tts=tts)
    if stt is None:
        pytest.importorskip("faster_whisper")
        from earshot.stt.local_whisper import LocalWhisperBackend

        stt = LocalWhisperBackend(model="tiny.en")
    loop = InterruptibleVoiceLoop(
        config, adapter, output, source=ArraySource(source_audio), stt=stt
    )
    return loop, adapter, sink, tts


class TestInterruptibleVoiceLoop:
    def test_voice_interrupt_full_cycle(self):
        # wake + command, then the user talks over the response; the
        # interrupting utterance must become the next command, no wake word.
        audio = np.concatenate(
            [
                read_wav("wake_then_command.wav"),
                read_wav("command.wav"),  # spoken over the response
                silence(1.5),
            ]
        )
        loop, adapter, sink, _tts = make_loop(audio)
        loop.run()

        assert len(adapter.prompts) == 2, f"expected 2 turns, got {adapter.prompts}"
        assert "test suite" in adapter.prompts[1].lower(), (
            "the interrupting utterance was not captured as the next command"
        )
        assert loop.interrupts == 1
        assert sink.aborted >= 1, "playback was never aborted"
        assert loop.latencies_ms and max(loop.latencies_ms) < 200, (
            f"interrupt latency exceeded the 200ms budget: {loop.latencies_ms}"
        )

    def test_no_stale_audio_after_interrupt(self):
        audio = np.concatenate(
            [read_wav("wake_then_command.wav"), read_wav("command.wav"), silence(1.5)]
        )
        loop, adapter, sink, _tts = make_loop(audio)

        # Freeze the second response so nothing new plays after the interrupt.
        original_send = adapter.send

        def send(prompt):
            if len(adapter.prompts) >= 1:
                adapter.prompts.append(prompt)
                return iter(())
            return original_send(prompt)

        adapter.send = send
        loop.run()
        settled = sink.total_samples()
        time.sleep(0.1)
        assert sink.total_samples() == settled, "stale audio played after the interrupt"

    def test_hotkey_interrupt_uses_same_path(self):
        # Silence after the first turn: only the escape hatch can interrupt.
        loop_holder = {}
        adapter_holder = {}

        class ScriptedSource:
            """Presses the hotkey between the first turn and more silence."""

            def frames(self):
                yield from ArraySource(read_wav("wake_then_command.wav")).frames()
                deadline = time.time() + 15
                while not adapter_holder["adapter"].prompts and time.time() < deadline:
                    time.sleep(0.01)
                time.sleep(0.05)  # let the loop settle into RESPONDING
                loop_holder["loop"].request_interrupt()
                yield from ArraySource(silence(2.0)).frames()

        loop, adapter, _sink, _tts = make_loop(silence(0.1))
        loop._source = ScriptedSource()
        loop_holder["loop"] = loop
        adapter_holder["adapter"] = adapter
        loop.run()

        assert loop.interrupts == 1, "the hotkey did not trigger the interrupt path"
        assert loop.latencies_ms and max(loop.latencies_ms) < 200

    def test_playback_alone_does_not_self_interrupt(self):
        # After the command, the mic hears only silence (the headset
        # assumption); the long response must play out uninterrupted.
        audio = np.concatenate([read_wav("wake_then_command.wav"), silence(4.0)])
        loop, adapter, _sink, tts = make_loop(audio)
        loop.run()
        assert loop.interrupts == 0, "silence triggered a self-interrupt"
        assert len(adapter.prompts) == 1
        assert tts.calls, "the response was never spoken"


def test_requires_wake_model():
    config = Config()
    with pytest.raises(ValueError, match="model_path"):
        InterruptibleVoiceLoop(config, SlowFakeAdapter(), object())
