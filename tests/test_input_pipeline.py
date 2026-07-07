"""Audio input pipeline tests: wake detection, end-of-speech, STT, and the
wake -> record -> transcribe flow end to end on committed fixtures.

The heavy voice dependencies (openwakeword, faster-whisper) are skipped
gracefully when not installed so the non-audio test suite stays runnable
anywhere; the no-mistakes/dev environment installs them via `pip install -e .`.
"""

import wave
from pathlib import Path

import numpy as np
import pytest

openwakeword = pytest.importorskip("openwakeword", reason="voice deps not installed")

from earshot.audio import FRAME_SAMPLES, SAMPLE_RATE  # noqa: E402
from earshot.audio.capture import ArraySource  # noqa: E402
from earshot.audio.endpointing import EndOfSpeechDetector  # noqa: E402
from earshot.config import Config  # noqa: E402
from earshot.input import InputPipeline  # noqa: E402
from earshot.wakeword import WakeWordDetector  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parent.parent
WAKE_MODEL = REPO_ROOT / "spikes" / "models" / "hey_earshot.onnx"


def read_wav(name: str) -> np.ndarray:
    with wave.open(str(FIXTURES / name)) as w:
        assert w.getframerate() == SAMPLE_RATE
        assert w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def feed(detector: WakeWordDetector, audio: np.ndarray) -> bool:
    fired = False
    for frame in ArraySource(audio).frames():
        if detector.detected(frame):
            fired = True
    return fired


# The committed feasibility-grade model scores this fixture voice at
# 0.94-0.95, right at its default 0.95 operating point (docs/latency-spike.md
# records the margins). Tests pin a slightly relaxed operating point so they
# exercise the detector logic deterministically instead of the model's edge:
# the positive fixture holds 3 consecutive windows above 0.9 (min 0.939) and
# the negative fixture never gets one window there (max 0.849).
TEST_SENSITIVITY = 0.9
TEST_PATIENCE = 3


@pytest.fixture(scope="module")
def wake_detector():
    return WakeWordDetector(
        model_path=str(WAKE_MODEL), sensitivity=TEST_SENSITIVITY, patience=TEST_PATIENCE
    )


class TestWakeWordDetector:
    def test_fires_on_wake_phrase(self, wake_detector):
        wake_detector.reset()
        assert feed(wake_detector, read_wav("wake_positive.wav"))

    def test_silent_on_other_speech(self, wake_detector):
        wake_detector.reset()
        assert not feed(wake_detector, read_wav("wake_negative.wav"))

    def test_silent_on_silence(self, wake_detector):
        wake_detector.reset()
        assert not feed(wake_detector, np.zeros(SAMPLE_RATE * 4, dtype=np.int16))


class TestEndOfSpeech:
    def frames(self, audio):
        return ArraySource(audio).frames()

    def test_ends_after_speech_then_silence(self):
        detector = EndOfSpeechDetector(silence_ms=600)
        audio = np.concatenate([read_wav("command.wav"), np.zeros(SAMPLE_RATE * 2, dtype=np.int16)])
        ended_at = None
        for i, frame in enumerate(self.frames(audio)):
            if detector.finished(frame):
                ended_at = i
                break
        assert ended_at is not None, "end of speech never detected"
        # It must end during the trailing silence, not mid-sentence.
        speech_frames = len(read_wav("command.wav")) // FRAME_SAMPLES
        trailing_silence = SAMPLE_RATE  # command.wav itself ends with 1s of silence
        assert ended_at >= speech_frames - trailing_silence // FRAME_SAMPLES

    def test_does_not_end_during_pure_silence_without_speech(self):
        detector = EndOfSpeechDetector(silence_ms=600, max_utterance_ms=10_000)
        for frame in self.frames(np.zeros(SAMPLE_RATE * 3, dtype=np.int16)):
            assert not detector.finished(frame)

    def test_max_duration_backstop(self):
        detector = EndOfSpeechDetector(silence_ms=600, max_utterance_ms=1_000)
        finished = [
            detector.finished(frame)
            for frame in self.frames(np.zeros(SAMPLE_RATE * 2, dtype=np.int16))
        ]
        assert any(finished), "max utterance backstop never fired"


@pytest.fixture(scope="module")
def whisper_backend():
    pytest.importorskip("faster_whisper")
    from earshot.stt.local_whisper import LocalWhisperBackend

    return LocalWhisperBackend(model="tiny.en")


class TestLocalWhisper:
    def test_transcribes_fixture(self, whisper_backend):
        text = whisper_backend.transcribe(read_wav("command.wav"), SAMPLE_RATE)
        assert "test suite" in text.lower()

    def test_rejects_wrong_sample_rate(self, whisper_backend):
        with pytest.raises(ValueError, match="16kHz"):
            whisper_backend.transcribe(np.zeros(8000, dtype=np.int16), 8000)


class TestInputPipeline:
    def make_config(self):
        config = Config()
        config.wake_word.model_path = str(WAKE_MODEL)
        config.wake_word.sensitivity = TEST_SENSITIVITY
        config.wake_word.patience = TEST_PATIENCE
        return config

    def test_wake_then_command_produces_transcript(self, whisper_backend):
        transcripts = []
        pipeline = InputPipeline(
            self.make_config(),
            on_transcript=transcripts.append,
            source=ArraySource(read_wav("wake_then_command.wav")),
            stt=whisper_backend,
        )
        pipeline.run()  # returns when the source is exhausted
        assert len(transcripts) == 1
        assert "test suite" in transcripts[0].lower()

    def test_no_wake_word_no_transcript(self, whisper_backend):
        transcripts = []
        pipeline = InputPipeline(
            self.make_config(),
            on_transcript=transcripts.append,
            source=ArraySource(read_wav("wake_negative.wav")),
            stt=whisper_backend,
        )
        pipeline.run()
        assert transcripts == []

    def test_requires_model_path(self):
        with pytest.raises(ValueError, match="model_path"):
            InputPipeline(
                Config(),
                on_transcript=lambda t: None,
                source=ArraySource(np.zeros(FRAME_SAMPLES, dtype=np.int16)),
            )
