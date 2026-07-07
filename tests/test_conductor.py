"""Conductor tests: registry semantics, fleet lifecycle with scripted
adapters, supervision/restart policy, and the new config validation
(duplicate names rejected, phonetically risky names warned).
"""

import threading
import time

import pytest

import earshot.agents
from earshot.conductor import Fleet, Registry
from earshot.conductor.registry import AgentRecord
from earshot.config import AgentConfig, Config, ConfigError, load, validate


class ScriptedAdapter:
    """A controllable adapter double for lifecycle tests."""

    def __init__(self, name, fail_start=False):
        self.name = name
        self.fail_start = fail_start
        self.started_at: list[float] = []
        self.stops = 0
        self._alive = False

    def start(self):
        self.started_at.append(time.perf_counter())
        if self.fail_start:
            from earshot.agents import AgentError

            raise AgentError(f"{self.name} refuses to start")
        self._alive = True

    def stop(self):
        self.stops += 1
        self._alive = False

    def kill(self):
        self._alive = False

    @property
    def alive(self):
        return self._alive

    def send(self, prompt):
        return iter(())


@pytest.fixture()
def fleet_factory(monkeypatch):
    adapters: dict[str, ScriptedAdapter] = {}

    def factory(names, fail=(), stagger=0.0):
        def fake_create(name, _config):
            adapters[name] = ScriptedAdapter(name, fail_start=name in fail)
            return adapters[name]

        monkeypatch.setattr(earshot.agents, "create_adapter", fake_create)
        config = Config()
        config.agents = {name: AgentConfig() for name in names}
        return Fleet(config, stagger_seconds=stagger), adapters

    return factory


class TestRegistry:
    def test_lookup_and_statuses(self):
        registry = Registry()
        record = AgentRecord(name="rex", config=AgentConfig(), adapter=ScriptedAdapter("rex"))
        registry.add(record)
        assert registry.get("rex") is record
        assert registry.statuses() == {"rex": "starting"}
        record.mark("idle")
        assert registry.statuses() == {"rex": "idle"}

    def test_unknown_name_is_a_clear_error(self):
        with pytest.raises(KeyError, match="no agent named 'nemo'"):
            Registry().get("nemo")

    def test_duplicate_registration_rejected(self):
        registry = Registry()
        record = AgentRecord(name="rex", config=AgentConfig(), adapter=ScriptedAdapter("rex"))
        registry.add(record)
        with pytest.raises(ValueError, match="already registered"):
            registry.add(record)

    def test_invalid_status_rejected(self):
        record = AgentRecord(name="rex", config=AgentConfig(), adapter=ScriptedAdapter("rex"))
        with pytest.raises(ValueError, match="unknown agent status"):
            record.mark("confused")


class TestFleetLifecycle:
    def test_start_all_marks_idle_and_staggers(self, fleet_factory):
        fleet, adapters = fleet_factory(["alpha", "bravo", "charlie"], stagger=0.05)
        fleet.start_all()
        assert fleet.statuses() == {"alpha": "idle", "bravo": "idle", "charlie": "idle"}
        spawn_times = [adapters[n].started_at[0] for n in ["alpha", "bravo", "charlie"]]
        gaps = [b - a for a, b in zip(spawn_times[:-1], spawn_times[1:], strict=True)]
        assert all(gap >= 0.04 for gap in gaps), f"startup was not staggered: {gaps}"

    def test_one_failed_start_does_not_abort_the_fleet(self, fleet_factory):
        fleet, _ = fleet_factory(["alpha", "bravo", "charlie"], fail={"bravo"})
        fleet.start_all()
        assert fleet.statuses() == {"alpha": "idle", "bravo": "dead", "charlie": "idle"}

    def test_stop_all_stops_every_adapter(self, fleet_factory):
        fleet, adapters = fleet_factory(["alpha", "bravo"])
        fleet.start_all()
        fleet.stop_all()
        assert all(a.stops >= 1 for a in adapters.values())
        assert not any(a.alive for a in adapters.values())
        assert set(fleet.statuses().values()) == {"dead"}

    def test_sixteen_agent_fleet(self, fleet_factory):
        names = [f"agent{i}" for i in range(16)]
        fleet, adapters = fleet_factory(names)
        fleet.start_all()
        assert all(status == "idle" for status in fleet.statuses().values())
        fleet.stop_all()
        assert not any(a.alive for a in adapters.values())


class TestSupervision:
    def wait_for(self, predicate, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.05)
        return False

    @pytest.fixture(autouse=True)
    def fast_supervision(self, monkeypatch):
        monkeypatch.setattr("earshot.conductor.lifecycle.SUPERVISION_INTERVAL", 0.05)

    def test_death_detected_and_restarted_without_disturbing_others(self, fleet_factory):
        fleet, adapters = fleet_factory(["alpha", "bravo"])
        fleet.start_all()
        fleet.start_supervision()
        try:
            adapters["alpha"].kill()
            assert self.wait_for(lambda: adapters["alpha"].alive), "alpha was not restarted"
            assert fleet.get("alpha").status == "idle"
            assert fleet.get("bravo").status == "idle"
            assert len(adapters["bravo"].started_at) == 1, "bravo was disturbed"
        finally:
            fleet.stop_all()

    def test_restart_policy_off_leaves_agent_dead(self, fleet_factory, monkeypatch):
        fleet, adapters = fleet_factory(["alpha"])
        fleet.get("alpha").config.restart_on_death = False
        fleet.start_all()
        fleet.start_supervision()
        try:
            adapters["alpha"].kill()
            assert self.wait_for(lambda: fleet.get("alpha").status == "dead")
            time.sleep(0.2)
            assert len(adapters["alpha"].started_at) == 1, "restart policy was ignored"
        finally:
            fleet.stop_all()

    def test_active_agent_is_left_to_the_conversation_loop(self, fleet_factory):
        fleet, adapters = fleet_factory(["alpha", "bravo"])
        fleet.start_all()
        fleet.start_supervision(active_name="alpha")
        try:
            adapters["alpha"].kill()
            assert self.wait_for(lambda: fleet.get("alpha").status == "dead")
            time.sleep(0.2)
            assert len(adapters["alpha"].started_at) == 1, (
                "the supervisor restarted the active agent; that is the loop's job"
            )
        finally:
            fleet.stop_all()


class TestMultiAgentConfig:
    def test_duplicate_agent_names_rejected(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("agents:\n  rex:\n    harness: opencode\n  rex:\n    harness: codex\n")
        with pytest.raises(ConfigError, match="duplicate key 'rex'"):
            load(path)

    def test_short_name_warns(self, caplog):
        config = Config()
        config.agents = {"rex": AgentConfig()}
        with caplog.at_level("WARNING", logger="earshot.config"):
            validate(config)
        assert any("short for speech recognition" in r.message for r in caplog.records)

    def test_similar_names_warn(self, caplog):
        config = Config()
        config.agents = {"marvin": AgentConfig(), "martin": AgentConfig()}
        with caplog.at_level("WARNING", logger="earshot.config"):
            validate(config)
        assert any("sound alike" in r.message for r in caplog.records)

    def test_distinct_names_do_not_warn(self, caplog):
        config = Config()
        config.agents = {"marvin": AgentConfig(), "olivia": AgentConfig()}
        with caplog.at_level("WARNING", logger="earshot.config"):
            validate(config)
        assert not [r for r in caplog.records if "sound alike" in r.message]

    def test_restart_on_death_validated(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("agents:\n  marvin:\n    restart_on_death: sometimes\n")
        with pytest.raises(ConfigError, match=r"agents\.marvin\.restart_on_death"):
            load(path)


def test_supervisor_thread_stops_cleanly(fleet_factory):
    fleet, _ = fleet_factory(["alpha"])
    fleet.start_all()
    fleet.start_supervision()
    fleet.stop_all()
    assert not any(t.name == "fleet-supervisor" and t.is_alive() for t in threading.enumerate())
