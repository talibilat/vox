"""Watcher and status tests: silent buffering, active-only speech, read
acknowledgement, status phrasing, buffer caps, and failure isolation.
"""

import threading
import time

import pytest

import earshot.agents
import earshot.conductor.watchers as watcher_module
from earshot.agents import AgentError
from earshot.conductor import Fleet, Router, WatcherPool
from earshot.conductor.status import spoken_status
from earshot.config import AgentConfig, Config

NAMES = ["marvin", "olivia", "sebastian"]


class ScriptedAdapter:
    def __init__(self, name):
        self.name = name
        self.prompts = []
        self.release = threading.Event()
        self.release.set()  # respond immediately unless a test holds it
        self.fail_next = False
        self.raise_next = None
        self.starts = 0
        self._alive = True

    def start(self):
        self.starts += 1
        self._alive = True

    def stop(self):
        self._alive = False

    @property
    def alive(self):
        return self._alive

    def send(self, prompt):
        self.prompts.append(prompt)
        if self.fail_next:
            self.fail_next = False
            raise AgentError(f"{self.name} exploded")
        if self.raise_next is not None:
            error = self.raise_next
            self.raise_next = None
            raise error
        self.release.wait(timeout=10)
        yield f"{self.name} answer to "
        yield f"{prompt!r}. "


class RecordingOutput:
    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)

    def speak_stream(self, stream):
        self.spoken.append("".join(stream))

    def wait_until_idle(self, timeout=None):
        return True


def wait_for(predicate, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture()
def rig(monkeypatch):
    adapters = {}

    def fake_create(name, _config):
        adapters[name] = ScriptedAdapter(name)
        return adapters[name]

    monkeypatch.setattr(earshot.agents, "create_adapter", fake_create)
    config = Config()
    config.agents = {name: AgentConfig() for name in NAMES}
    fleet = Fleet(config, stagger_seconds=0)
    fleet.start_all()
    output = RecordingOutput()
    pool = WatcherPool(fleet, output)
    router = Router(
        fleet,
        output,
        read_response=pool.latest_response_text,
        fleet_status=pool.status_line,
        dispatch=pool.dispatch,
    )
    pool.set_active_probe(lambda name: router.active_agent == name)
    yield router, pool, adapters, output, fleet
    pool.stop()
    fleet.stop_all()


class TestSilentBuffering:
    def test_background_agents_never_speak_unprompted(self, rig):
        router, pool, adapters, output, fleet = rig
        router.handle_transcript("marvin, write the parser")  # marvin becomes active
        wait_for(lambda: fleet.get("marvin").status == "finished")
        spoken_before = list(output.spoken)
        # olivia works in the background: dispatch directly (as if previously routed)
        pool.dispatch("olivia", "run the benchmarks")
        assert wait_for(lambda: fleet.get("olivia").status == "finished")
        assert output.spoken == spoken_before, "a background agent was spoken aloud"

    def test_active_agent_response_is_spoken(self, rig):
        router, _pool, _adapters, output, fleet = rig
        router.handle_transcript("olivia, run the linter")
        assert wait_for(lambda: fleet.get("olivia").status == "finished")
        assert any("olivia answer" in s for s in output.spoken)

    def test_reading_one_agent_speaks_only_that_agent(self, rig):
        router, pool, _adapters, output, fleet = rig
        pool.dispatch("olivia", "task one")
        pool.dispatch("sebastian", "task two")
        assert wait_for(
            lambda: (
                fleet.get("olivia").status == "finished"
                and fleet.get("sebastian").status == "finished"
            )
        )
        router.handle_transcript("sebastian, what's your response")
        read_back = output.spoken[-1]
        assert "sebastian answer" in read_back
        assert "olivia" not in read_back

    def test_read_acknowledges_finished_to_idle(self, rig):
        router, pool, _adapters, _output, fleet = rig
        pool.dispatch("olivia", "task")
        assert wait_for(lambda: fleet.get("olivia").status == "finished")
        router.handle_transcript("olivia, what's your response")
        assert fleet.get("olivia").status == "idle"

    def test_nothing_to_read_says_so(self, rig):
        router, _pool, _adapters, output, _fleet = rig
        router.handle_transcript("sebastian, what did you say")
        assert any("has not said anything yet" in s for s in output.spoken)


class TestStatus:
    def test_mixed_status_summary(self, rig):
        router, pool, adapters, output, fleet = rig
        adapters["olivia"].release.clear()  # olivia stays busy
        pool.dispatch("olivia", "long task")
        pool.dispatch("sebastian", "quick task")
        assert wait_for(lambda: fleet.get("sebastian").status == "finished")
        assert wait_for(lambda: fleet.get("olivia").status == "busy")
        router.handle_transcript("agent status")
        line = output.spoken[-1]
        assert "sebastian" in line and "finished" in line
        assert "olivia" in line and "working" in line
        adapters["olivia"].release.set()

    def test_phrasing_cases(self):
        assert spoken_status({}) == "No agents are configured."
        assert spoken_status({"a": "finished"}) == "a has finished."
        assert (
            spoken_status({"a": "finished", "b": "finished", "c": "busy"})
            == "a and b have finished; c is still working."
        )
        assert spoken_status({"a": "idle", "b": "dead"}) == "a is idle; b is not running."
        assert spoken_status({"a": "busy", "b": "starting"}) == "a and b are still working."


class TestBufferLimits:
    def test_response_is_capped_before_buffering(self, rig, monkeypatch):
        _router, pool, adapters, _output, fleet = rig
        monkeypatch.setattr(watcher_module, "MAX_RESPONSE_CHARS", 10)
        original_buffer = watcher_module.AgentWatcher._buffer
        received_lengths = []

        def record_buffer_size(self, response):
            received_lengths.append(len(response))
            original_buffer(self, response)

        monkeypatch.setattr(watcher_module.AgentWatcher, "_buffer", record_buffer_size)

        def send(prompt):
            adapters["marvin"].prompts.append(prompt)
            yield "x" * 25
            yield "y" * 25

        adapters["marvin"].send = send
        pool.dispatch("marvin", "flood me")
        assert wait_for(lambda: fleet.get("marvin").status == "finished")
        assert max(received_lengths) <= 10 + len(" (response truncated)")

    def test_oversized_response_is_truncated(self, rig):
        _router, pool, adapters, _output, fleet = rig
        big = "x" * 500_000

        def send(prompt):
            adapters["marvin"].prompts.append(prompt)
            yield big

        adapters["marvin"].send = send
        pool.dispatch("marvin", "flood me")
        assert wait_for(lambda: fleet.get("marvin").status == "finished")
        stored = pool.latest_response_text("marvin")
        assert len(stored) < 200_000
        assert stored.endswith("(response truncated)")

    def test_ring_buffer_keeps_latest(self, rig):
        _router, pool, _adapters, _output, fleet = rig
        for i in range(12):  # deeper than the ring
            pool.dispatch("marvin", f"task {i}")
        assert wait_for(
            lambda: (
                fleet.get("marvin").status == "finished"
                and "task 11" in pool._watchers["marvin"]._responses[-1]
            )
        )
        assert "task 11" in pool.latest_response_text("marvin")
        assert len(pool._watchers["marvin"]._responses) <= 8


class TestFailureIsolation:
    def test_unexpected_turn_failure_is_buffered_as_finished_response(self, rig):
        _router, pool, adapters, _output, fleet = rig
        adapters["olivia"].raise_next = RuntimeError("tool crashed")
        pool.dispatch("olivia", "explode")
        assert wait_for(lambda: fleet.get("olivia").status == "finished")
        assert "failed" in pool.latest_response_text("olivia")
        assert "tool crashed" in pool.latest_response_text("olivia")

    def test_failed_turn_marks_agent_without_touching_others(self, rig):
        _router, pool, adapters, output, fleet = rig
        adapters["olivia"].fail_next = True
        pool.dispatch("olivia", "explode")
        pool.dispatch("sebastian", "carry on")
        assert wait_for(lambda: fleet.get("sebastian").status == "finished")
        assert wait_for(lambda: fleet.get("olivia").status in ("finished", "dead"))
        assert "failed" in pool.latest_response_text("olivia")
        assert "sebastian answer" in pool.latest_response_text("sebastian")
        # background failure stays silent (olivia was not active)
        assert not any("not responding" in s for s in output.spoken)

    def test_watcher_survives_for_next_turn(self, rig):
        _router, pool, adapters, _output, fleet = rig
        adapters["olivia"].fail_next = True
        pool.dispatch("olivia", "explode")
        pool.dispatch("olivia", "recover")
        assert wait_for(lambda: "olivia answer to 'recover'" in pool.latest_response_text("olivia"))


class TestWatcherSupervision:
    def test_watcher_mode_does_not_exempt_initial_active_agent(self, monkeypatch):
        monkeypatch.setattr("earshot.conductor.lifecycle.SUPERVISION_INTERVAL", 0.05)
        adapters = {}

        def fake_create(name, _config):
            adapters[name] = ScriptedAdapter(name)
            return adapters[name]

        monkeypatch.setattr(earshot.agents, "create_adapter", fake_create)
        config = Config()
        config.agents = {name: AgentConfig() for name in ["marvin", "olivia"]}
        fleet = Fleet(config, stagger_seconds=0)
        fleet.start_all()
        output = RecordingOutput()
        pool = WatcherPool(fleet, output)
        try:
            fleet.start_supervision()
            router = Router(
                fleet,
                output,
                read_response=pool.latest_response_text,
                fleet_status=pool.status_line,
                dispatch=pool.dispatch,
            )
            pool.set_active_probe(lambda name: router.active_agent == name)
            adapters["marvin"].stop()
            assert wait_for(lambda: adapters["marvin"].alive)
            assert adapters["marvin"].starts >= 2
        finally:
            pool.stop()
            fleet.stop_all()


def test_sixteen_agents_concurrent_no_soup(monkeypatch):
    adapters = {}

    def fake_create(name, _config):
        adapters[name] = ScriptedAdapter(name)
        return adapters[name]

    monkeypatch.setattr(earshot.agents, "create_adapter", fake_create)
    config = Config()
    names = [f"agent{i}" for i in range(16)]
    config.agents = {name: AgentConfig() for name in names}
    fleet = Fleet(config, stagger_seconds=0)
    fleet.start_all()
    output = RecordingOutput()
    pool = WatcherPool(fleet, output)  # active probe defaults to nobody-active
    for name in names:
        pool.dispatch(name, f"work for {name}")
    assert wait_for(
        lambda: all(status == "finished" for status in fleet.statuses().values()), timeout=20
    )
    assert output.spoken == [], "16 concurrent agents produced unprompted speech"
    assert "agent7 answer" in pool.latest_response_text("agent7")
    line = spoken_status(fleet.statuses())
    assert "have finished" in line
    pool.stop()
    fleet.stop_all()
