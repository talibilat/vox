"""tmux fallback transport tests, against a real tmux with a scripted REPL
standing in for a harness CLI. Skipped where tmux is not installed.
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

import earshot.agents
from earshot.agents import AgentError, create_adapter
from earshot.agents.tmux_fallback import TmuxAgentAdapter, _strip_ansi
from earshot.config import AgentConfig, Config

pytestmark = pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux not installed")

REPO_ROOT = Path(__file__).resolve().parent.parent
FAKE_REPL = f"{sys.executable} -u {Path(__file__).resolve().parent / 'fake_pane_agent.py'}"


def tmux_config(session: str, **overrides) -> AgentConfig:
    fields = {
        "harness": "opencode",
        "command": FAKE_REPL,
        "workdir": str(REPO_ROOT),
        "tmux_pane": session,
    }
    fields.update(overrides)
    return AgentConfig(**fields)


@pytest.fixture()
def adapter(request):
    session = f"earshot-test-{request.node.name[:20]}"
    adapter = create_adapter("fallback", tmux_config(session))
    adapter.start()
    yield adapter
    adapter.stop()


def has_session(session: str) -> bool:
    return (
        subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0
    )


def bare_adapter() -> TmuxAgentAdapter:
    adapter = object.__new__(TmuxAgentAdapter)
    adapter._name = "fallback"
    adapter._session = "earshot-test-bare"
    return adapter


def test_tmux_pane_field_selects_the_fallback_adapter():
    adapter = create_adapter("fallback", tmux_config("earshot-test-select"))
    assert type(adapter) is TmuxAgentAdapter


def test_spawns_owns_and_kills_the_session():
    session = "earshot-test-lifecycle"
    adapter = create_adapter("fallback", tmux_config(session))
    adapter.start()
    assert adapter.alive
    assert has_session(session)
    adapter.stop()
    assert not adapter.alive
    assert not has_session(session)
    adapter.stop()  # idempotent


def test_turn_returns_clean_text(adapter):
    response = "".join(adapter.send("hello fallback"))
    assert "GOT: hello fallback" in response
    assert "done." in response
    assert "\x1b" not in response, "ANSI escape codes reached the response"
    assert "hello fallback\n" not in response.split("GOT:")[0], "the echoed prompt was read back"


def test_special_characters_arrive_intact(adapter):
    tricky = "quotes \"double\" and 'single', $VAR, `ticks`, 100% & a;semi"
    response = "".join(adapter.send(tricky))
    assert f"GOT: {tricky}" in response


def test_multiline_prompt_via_paste_buffer(adapter):
    prompt = "first line\nsecond line with $pecial\nthird"
    response = "".join(adapter.send(prompt))
    for line in prompt.splitlines():
        assert f"GOT: {line}" in response


def test_prompt_delivery_failure_raises_immediately(monkeypatch):
    adapter = bare_adapter()

    def fail_paste(*_args):
        return subprocess.CompletedProcess(["tmux"], 1, "", "no pane")

    monkeypatch.setattr(adapter, "_tmux", fail_paste)
    with pytest.raises(AgentError, match="deliver prompt"):
        adapter._deliver("hello")


def test_multiline_load_buffer_failure_raises_immediately(monkeypatch):
    adapter = bare_adapter()

    def fail_load_buffer(*_args, **_kwargs):
        return subprocess.CompletedProcess(["tmux"], 1, "", "buffer failed")

    monkeypatch.setattr(subprocess, "run", fail_load_buffer)
    with pytest.raises(AgentError, match="deliver prompt"):
        adapter._deliver("hello\nthere")


def test_send_yields_each_stable_wait_chunk(monkeypatch):
    adapter = bare_adapter()
    monkeypatch.setattr(TmuxAgentAdapter, "alive", property(lambda _self: True))
    monkeypatch.setattr(adapter, "_capture", lambda: "before")
    monkeypatch.setattr(adapter, "_deliver", lambda _prompt: None)
    monkeypatch.setattr(
        adapter,
        "_wait_for_stable_output",
        lambda _before, _prompt: iter(["first chunk", "second chunk"]),
    )

    assert list(adapter.send("hello")) == ["first chunk", "second chunk"]


def test_extract_response_uses_overlap_when_capture_window_slides():
    adapter = bare_adapter()
    before = "\n".join(f"old line {index}" for index in range(2000))
    after = "\n".join([*(f"old line {index}" for index in range(1, 2000)), "new answer"])

    assert adapter._extract_response(before, after, "prompt") == "new answer"


def test_strips_osc_sequences_terminated_by_st():
    assert _strip_ansi("before\x1b]0;pane title\x1b\\after") == "beforeafter"


def test_multi_turn_session_persists(adapter):
    first = "".join(adapter.send("turn one"))
    second = "".join(adapter.send("turn two"))
    assert "GOT: turn one" in first
    assert "GOT: turn two" in second
    assert "turn one" not in second, "previous turn output leaked into the new turn"


def test_death_mid_turn_raises(adapter):
    def kill_soon():
        time.sleep(0.6)
        subprocess.run(["tmux", "kill-session", "-t", adapter._session], capture_output=True)

    import threading

    threading.Thread(target=kill_soon, daemon=True).start()
    with pytest.raises(AgentError, match="died|lost"):
        "".join(adapter.send("this will be interrupted"))


def test_mixed_fleet_behaves_uniformly(monkeypatch):
    """A fleet with one native (scripted) agent and one tmux agent, driven
    through the same WatcherPool the voice layer uses."""
    from earshot.conductor import Fleet, WatcherPool
    from tests.test_watchers import RecordingOutput, ScriptedAdapter, wait_for

    scripted = {}
    real_create = earshot.agents.create_adapter

    def fake_create(name, agent_config):
        if agent_config.tmux_pane:
            return real_create(name, agent_config)
        scripted[name] = ScriptedAdapter(name)
        return scripted[name]

    monkeypatch.setattr(earshot.agents, "create_adapter", fake_create)
    config = Config()
    config.agents = {
        "native": AgentConfig(),
        "fallback": tmux_config("earshot-test-mixed"),
    }
    fleet = Fleet(config, stagger_seconds=0)
    fleet.start_all()
    output = RecordingOutput()
    pool = WatcherPool(fleet, output)
    try:
        pool.dispatch("native", "native task")
        pool.dispatch("fallback", "fallback task")
        assert wait_for(
            lambda: (
                fleet.get("native").status == "finished"
                and fleet.get("fallback").status == "finished"
            ),
            timeout=30,
        ), f"statuses: {fleet.statuses()}"
        assert "native answer" in pool.latest_response_text("native")
        assert "GOT: fallback task" in pool.latest_response_text("fallback")
        assert output.spoken == [], "a fleet agent spoke unprompted"
    finally:
        pool.stop()
        fleet.stop_all()
