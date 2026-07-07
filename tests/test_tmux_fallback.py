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
from earshot.agents.tmux_fallback import TmuxAgentAdapter
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
