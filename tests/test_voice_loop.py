"""The full voice loop on fixtures: wake word audio in -> transcript ->
agent (the fake serve process, spawned and owned by the adapter) -> markdown
response -> speakable text synthesized.

This is the automated version of the issue's end-to-end smoke test; the
only unreal parts are the audio device (fixture wav + fake speaker sink)
and the agent brain (a scripted HTTP server). Everything between them,
including process ownership and the HTTP/SSE transport, is the real code.
The manual counterpart with a live microphone and real opencode is
documented in docs/tickets/P1-05.md.
"""

import wave
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openwakeword", reason="voice deps not installed")

from earshot.agents import create_adapter  # noqa: E402
from earshot.audio.capture import ArraySource  # noqa: E402
from earshot.config import Config  # noqa: E402
from earshot.input import InputPipeline  # noqa: E402
from earshot.loop import ConversationLoop  # noqa: E402
from earshot.output import OutputPipeline  # noqa: E402
from tests.test_agents import fake_agent_config  # noqa: E402
from tests.test_output_pipeline import FakeSink, FakeTts  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parent.parent
WAKE_MODEL = REPO_ROOT / "spikes" / "models" / "hey_earshot.onnx"


def read_wav(name: str) -> np.ndarray:
    with wave.open(str(FIXTURES / name)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def test_wake_to_spoken_agent_response():
    pytest.importorskip("faster_whisper")
    from earshot.audio.playback import Player
    from earshot.stt.local_whisper import LocalWhisperBackend

    config = Config()
    config.wake_word.model_path = str(WAKE_MODEL)
    config.wake_word.sensitivity = 0.9  # fixture margins; see test_input_pipeline
    config.wake_word.patience = 3

    adapter = create_adapter("main", fake_agent_config())
    adapter.start()
    try:
        tts = FakeTts()
        output = OutputPipeline(config, player=Player(FakeSink()), tts=tts)
        loop = ConversationLoop(adapter, output)
        pipeline = InputPipeline(
            config,
            on_transcript=loop.handle_transcript,
            source=ArraySource(read_wav("wake_then_command.wav")),
            stt=LocalWhisperBackend(model="tiny.en"),
        )
        pipeline.run()  # returns when the fixture audio is exhausted

        spoken = " ".join(tts.calls)
        # The agent echoes the transcript back; hearing the instruction text
        # inside the spoken response proves the whole chain end to end.
        assert "you said" in spoken
        assert "test suite" in spoken.lower()
        assert "Turn 1" in spoken
    finally:
        adapter.stop()
