"""Cross-harness adapter matrix: the same behavioral contract, asserted
against all three adapters, each driving its fake harness as a real spawned
child process (tests/fake_opencode_serve.py, tests/fake_claude_code.py,
tests/fake_codex_appserver.py).

Real-binary integration tests live at the bottom, skipped when a harness is
not installed; they prove the transports against the actual CLIs.
"""

import shutil
import sys
from pathlib import Path

import pytest

from earshot.agents import AgentError, create_adapter
from earshot.agents.claude_code import ClaudeCodeAdapter
from earshot.agents.codex import CodexAdapter
from earshot.agents.opencode import OpencodeAdapter
from earshot.config import AgentConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent

HARNESSES = ["opencode", "claude-code", "codex"]
ADAPTER_CLASSES = {
    "opencode": OpencodeAdapter,
    "claude-code": ClaudeCodeAdapter,
    "codex": CodexAdapter,
}
FAKE_COMMANDS = {
    "opencode": f"{sys.executable} {TESTS / 'fake_opencode_serve.py'} serve",
    "claude-code": f"{sys.executable} {TESTS / 'fake_claude_code.py'}",
    "codex": f"{sys.executable} {TESTS / 'fake_codex_appserver.py'}",
}


def harness_config(harness: str, **overrides) -> AgentConfig:
    fields = {
        "harness": harness,
        "command": FAKE_COMMANDS[harness],
        "workdir": str(REPO_ROOT),
    }
    fields.update(overrides)
    return AgentConfig(**fields)


@pytest.fixture(autouse=True)
def _fake_claude_state(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_STATE", str(tmp_path / "claude-state"))


@pytest.fixture(params=HARNESSES)
def harness(request):
    return request.param


@pytest.fixture()
def adapter(harness):
    adapter = create_adapter("main", harness_config(harness))
    adapter.start()
    yield adapter
    adapter.stop()


def test_registry_selects_adapter_by_harness(harness):
    adapter = create_adapter("main", harness_config(harness))
    assert type(adapter) is ADAPTER_CLASSES[harness]


class TestAdapterContract:
    def test_start_and_stop_cleanly(self, harness):
        adapter = create_adapter("main", harness_config(harness))
        adapter.start()
        assert adapter.alive
        adapter.stop()
        adapter.stop()  # idempotent

    def test_turn_streams_text_until_completion(self, adapter):
        chunks = list(adapter.send("hello there"))
        assert "you said 'hello there'" in "".join(chunks)
        assert len(chunks) >= 2, "response did not stream in multiple chunks"

    def test_multi_turn_persists_session(self, adapter):
        first = "".join(adapter.send("first question"))
        second = "".join(adapter.send("a follow-up"))
        assert "Turn 1" in first
        assert "Turn 2" in second, "second turn did not land in the same session"

    def test_in_band_error_raises_speakable_agent_error(self, adapter):
        with pytest.raises(AgentError, match="error|failure"):
            list(adapter.send("please error"))
        assert adapter.alive, "an in-band error must not kill the adapter"
        # And the session must still be usable afterward.
        assert "you said" in "".join(adapter.send("still there?"))

    def test_death_mid_turn_raises_not_hangs(self, adapter):
        with pytest.raises(AgentError):
            list(adapter.send("please die"))

    def test_unlaunchable_command_raises(self, harness):
        adapter = create_adapter(
            "main", harness_config(harness, command="/nonexistent/binary serve")
        )
        with pytest.raises(AgentError, match="could not launch|not found"):
            adapter.start()


@pytest.mark.skipif(shutil.which("claude") is None, reason="claude not installed")
class TestRealClaudeCode:
    def test_two_turn_conversation(self):
        adapter = create_adapter(
            "main",
            AgentConfig(harness="claude-code", workdir=str(REPO_ROOT), model="haiku"),
        )
        adapter.start()
        try:
            first = "".join(adapter.send("Reply with exactly: CLAUDE_ADAPTER_OK"))
            assert "CLAUDE_ADAPTER_OK" in first
            second = "".join(
                adapter.send("Reply with exactly the marker from your previous reply.")
            )
            assert "CLAUDE_ADAPTER_OK" in second, "multi-turn persistence failed (--resume)"
        finally:
            adapter.stop()


@pytest.mark.skipif(shutil.which("codex") is None, reason="codex not installed")
class TestRealCodex:
    def test_two_turn_conversation(self):
        adapter = create_adapter("main", AgentConfig(harness="codex", workdir=str(REPO_ROOT)))
        adapter.start()
        try:
            first = "".join(adapter.send("Reply with exactly: CODEX_ADAPTER_OK"))
            assert "CODEX_ADAPTER_OK" in first
            second = "".join(
                adapter.send("Reply with exactly the marker from your previous reply.")
            )
            assert "CODEX_ADAPTER_OK" in second, "multi-turn persistence failed (thread reuse)"
        finally:
            adapter.stop()
