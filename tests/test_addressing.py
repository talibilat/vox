"""Addressing tests: the noisy-transcript corpus that drives the fuzzy-match
thresholds, plus router classification, active-agent behavior, ambiguous
clarification, and fleet-phrase protection.
"""

import pytest

import earshot.agents
from earshot.conductor import Fleet, Router
from earshot.conductor.addressing import ROUTE_THRESHOLD, extract_address
from earshot.config import AgentConfig, Config

NAMES = ["marvin", "olivia", "sebastian"]

# The corpus: (utterance, expected name or None, expected command).
# "Misheard" rows are hand-simulated STT noise for the configured names.
CORPUS_ROUTED = [
    # clean leading-name addressing
    ("marvin, run the tests", "marvin", "run the tests"),
    ("olivia, what changed in the diff", "olivia", "what changed in the diff"),
    ("sebastian, undo the last commit", "sebastian", "undo the last commit"),
    ("marvin run the tests", "marvin", "run the tests"),  # no comma
    ("Marvin, RUN the tests!", "marvin", "RUN the tests!"),  # case/punctuation
    # misheard names (vowel drift, cluster mangling)
    ("marvun, run the tests", "marvin", "run the tests"),
    ("marven run the tests", "marvin", "run the tests"),
    ("olivea, keep going", "olivia", "keep going"),
    ("alivia, keep going", "olivia", "keep going"),
    ("sebastien, try again", "sebastian", "try again"),
    ("sabastian rerun it", "sebastian", "rerun it"),
    # clipped leading word absorbed into the name window
    ("hey marvin, run the tests", "marvin", "run the tests"),
]

CORPUS_UNADDRESSED = [
    "run the tests",
    "what happened to the build",
    "keep going with that approach",
    "try a different library",
    "thanks, that looks right",
    "can you make it faster",
    "the file is in the src directory",
    "no, revert that change",
]


class TestExtractAddress:
    @pytest.mark.parametrize(("utterance", "name", "command"), CORPUS_ROUTED)
    def test_routes_noisy_names(self, utterance, name, command):
        address = extract_address(utterance, NAMES)
        assert address.name == name, f"{utterance!r} matched {address.name!r}"
        assert address.confidence >= ROUTE_THRESHOLD, (
            f"{utterance!r} only reached {address.confidence:.2f}"
        )
        assert address.command == command

    @pytest.mark.parametrize("utterance", CORPUS_UNADDRESSED)
    def test_unaddressed_stays_unaddressed(self, utterance):
        address = extract_address(utterance, NAMES)
        assert address.confidence < ROUTE_THRESHOLD, (
            f"{utterance!r} misrouted to {address.name!r} at {address.confidence:.2f}"
        )

    def test_name_only_utterance(self):
        address = extract_address("marvin", NAMES)
        assert address.name == "marvin"
        assert address.command == ""

    def test_empty_inputs(self):
        assert extract_address("", NAMES).name is None
        assert extract_address("marvin, hello", []).name is None


class RecordingAdapter:
    def __init__(self, name):
        self.name = name
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
        yield f"{self.name} did it. "


class RecordingOutput:
    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)

    def speak_stream(self, stream):
        self.spoken.append("".join(stream))

    def wait_until_idle(self, timeout=None):
        return True


@pytest.fixture()
def router(monkeypatch):
    adapters = {}

    def fake_create(name, _config):
        adapters[name] = RecordingAdapter(name)
        return adapters[name]

    monkeypatch.setattr(earshot.agents, "create_adapter", fake_create)
    config = Config()
    config.agents = {name: AgentConfig() for name in NAMES}
    fleet = Fleet(config, stagger_seconds=0)
    fleet.start_all()
    output = RecordingOutput()
    return Router(fleet, output), adapters, output, fleet


class TestRouter:
    def test_addressed_command_routes_and_sets_active(self, router):
        r, adapters, _out, fleet = router
        r.handle_transcript("olivia, run the linter")
        assert adapters["olivia"].prompts == ["run the linter"]
        assert r.active_agent == "olivia"
        assert fleet._active_name == "olivia"

    def test_unaddressed_follow_up_goes_to_active(self, router):
        r, adapters, _out, _fleet = router
        r.handle_transcript("olivia, run the linter")
        r.handle_transcript("now fix the warnings")
        assert adapters["olivia"].prompts == ["run the linter", "now fix the warnings"]

    def test_readdressing_switches_active(self, router):
        r, adapters, _out, _fleet = router
        r.handle_transcript("olivia, run the linter")
        r.handle_transcript("sebastian, write the docs")
        r.handle_transcript("make them shorter")
        assert adapters["sebastian"].prompts == ["write the docs", "make them shorter"]
        assert adapters["olivia"].prompts == ["run the linter"]

    def test_ambiguous_name_clarifies_never_misroutes(self, router):
        r, adapters, out, _fleet = router
        r.handle_transcript("marlon, run the tests")  # 0.70: in the clarify band
        assert all(not a.prompts for a in adapters.values()), "ambiguous input was routed"
        assert any("Did you mean" in s for s in out.spoken)

    def test_clarification_yes_routes_held_command(self, router):
        r, adapters, out, _fleet = router
        r.handle_transcript("marlon, run the tests")
        candidate = next(s for s in out.spoken if "Did you mean" in s)
        r.handle_transcript("yes")
        routed = [a for a in adapters.values() if a.prompts]
        assert len(routed) == 1
        assert routed[0].prompts == ["run the tests"]
        assert routed[0].name in candidate

    def test_clarification_other_utterance_falls_through(self, router):
        r, adapters, _out, _fleet = router
        r.handle_transcript("marlon, run the tests")
        r.handle_transcript("marvin, do something else")
        assert adapters["marvin"].prompts == ["do something else"]

    def test_fleet_phrase_never_reaches_an_agent(self, router):
        r, adapters, out, _fleet = router
        r.handle_transcript("agent status")
        assert all(not a.prompts for a in adapters.values()), "fleet phrase became a prompt"
        assert any("idle" in s for s in out.spoken)

    def test_read_request_uses_the_seam_not_the_agent(self, router):
        r, adapters, out, _fleet = router
        r.handle_transcript("olivia, what's your response")
        assert adapters["olivia"].prompts == [], "read request became a prompt"
        assert out.spoken, "read request produced no speech"

    def test_read_seam_callback_is_used(self, monkeypatch, router):
        r, _adapters, out, _fleet = router
        r._read_response = lambda name: f"{name} said forty two."
        r.handle_transcript("olivia, what did you say")
        assert "olivia said forty two." in out.spoken

    def test_dead_agent_is_reported_not_prompted(self, router):
        r, adapters, out, fleet = router
        fleet.get("marvin").mark("dead")
        r.handle_transcript("marvin, run the tests")
        assert adapters["marvin"].prompts == []
        assert any("not running" in s for s in out.spoken)

    def test_name_only_switches_active(self, router):
        r, adapters, out, _fleet = router
        r.handle_transcript("sebastian")
        assert r.active_agent == "sebastian"
        assert any("listening" in s for s in out.spoken)
        assert adapters["sebastian"].prompts == []


def test_voice_loop_with_router_end_to_end(monkeypatch):
    """Fixture wake audio -> STT -> Router -> the active agent of a
    three-agent fleet, through the real interruptible voice loop."""
    pytest.importorskip("openwakeword")
    pytest.importorskip("faster_whisper")
    import wave
    from pathlib import Path

    import numpy as np

    from earshot.audio.capture import ArraySource
    from earshot.barge import InterruptibleVoiceLoop
    from earshot.stt.local_whisper import LocalWhisperBackend

    adapters = {}

    def fake_create(name, _config):
        adapters[name] = RecordingAdapter(name)
        return adapters[name]

    monkeypatch.setattr(earshot.agents, "create_adapter", fake_create)
    repo = Path(__file__).resolve().parent.parent
    config = Config()
    config.wake_word.model_path = str(repo / "spikes" / "models" / "hey_earshot.onnx")
    config.wake_word.sensitivity = 0.9
    config.wake_word.patience = 3
    config.agents = {name: AgentConfig() for name in NAMES}

    fleet = Fleet(config, stagger_seconds=0)
    fleet.start_all()
    output = RecordingOutput()
    router = Router(fleet, output)
    with wave.open(str(repo / "tests" / "fixtures" / "wake_then_command.wav")) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    loop = InterruptibleVoiceLoop(
        config, router, output, source=ArraySource(audio), stt=LocalWhisperBackend(model="tiny.en")
    )
    loop.run()
    fleet.stop_all()

    # The fixture command has no agent name, so the active (first) agent
    # must receive it and speak its response.
    assert adapters[NAMES[0]].prompts, "the active agent never received the command"
    assert "test suite" in adapters[NAMES[0]].prompts[0].lower()
    assert any("did it" in s for s in output.spoken)
