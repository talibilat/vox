"""Adapter and conversation-loop tests.

The opencode adapter runs against tests/fake_opencode_serve.py (spawned as
the adapter's own child process via the config `command` override), so the
full spawn -> session -> prompt -> stream -> completion path is exercised
without the real opencode binary. A separate integration test drives the
real binary when it is installed.
"""

import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from earshot.agents import AgentError, create_adapter, first_agent
from earshot.agents import opencode as opencode_module
from earshot.agents.base import AgentAdapter
from earshot.agents.opencode import OpencodeAdapter
from earshot.config import AgentConfig, Config
from earshot.loop import ConversationLoop

REPO_ROOT = Path(__file__).resolve().parent.parent
FAKE_SERVE = Path(__file__).resolve().parent / "fake_opencode_serve.py"


def fake_agent_config(**overrides):
    fields = {
        "harness": "opencode",
        "command": f"{sys.executable} {FAKE_SERVE} serve",
        "workdir": str(REPO_ROOT),
    }
    fields.update(overrides)
    return AgentConfig(**fields)


@pytest.fixture()
def adapter():
    adapter = create_adapter("main", fake_agent_config())
    adapter.start()
    yield adapter
    adapter.stop()


class TestOpencodeAdapter:
    def test_spawns_owns_and_stops_the_process(self):
        adapter = create_adapter("main", fake_agent_config())
        adapter.start()
        assert adapter.alive
        adapter.stop()
        assert not adapter.alive
        adapter.stop()  # idempotent

    def test_turn_streams_text_until_completion(self, adapter):
        chunks = list(adapter.send("hello there"))
        text = "".join(chunks)
        assert "you said 'hello there'" in text
        assert len(chunks) >= 2, "response did not stream in multiple chunks"

    def test_multi_turn_persists_session(self, adapter):
        first = "".join(adapter.send("first question"))
        second = "".join(adapter.send("a follow-up"))
        assert "Turn 1" in first
        assert "Turn 2" in second, "second turn did not land in the same session"

    def test_model_is_pinned_from_config(self):
        adapter = create_adapter("main", fake_agent_config(model="opencode/some-model"))
        adapter.start()
        try:
            assert "".join(adapter.send("check"))  # session creation succeeded
        finally:
            adapter.stop()

    def test_in_band_error_raises_speakable_agent_error(self, adapter):
        with pytest.raises(AgentError, match="error"):
            list(adapter.send("please error"))
        assert adapter.alive, "an in-band error must not kill the process"

    def test_agent_death_mid_turn_raises_not_hangs(self, adapter):
        started = time.monotonic()
        with pytest.raises(AgentError):
            list(adapter.send("please die"))
        assert time.monotonic() - started < 30, "death detection hung"

    def test_send_after_death_raises_immediately(self, adapter):
        with pytest.raises(AgentError):
            list(adapter.send("please die"))
        with pytest.raises(AgentError, match="not running"):
            list(adapter.send("anything"))

    def test_unlaunchable_command_raises(self):
        adapter = create_adapter("main", fake_agent_config(command="/nonexistent/binary serve"))
        with pytest.raises(AgentError, match="could not launch"):
            adapter.start()

    def test_start_cleans_up_process_when_startup_fails_after_spawn(self, monkeypatch):
        command = f"{sys.executable} -c 'import time; time.sleep(60)'"
        adapter = OpencodeAdapter("main", fake_agent_config(command=command))

        def fail_ready():
            raise AgentError("not ready")

        monkeypatch.setattr(adapter, "_wait_ready", fail_ready)
        with pytest.raises(AgentError, match="not ready"):
            adapter.start()

        assert not adapter.alive
        assert adapter._proc is None

    @pytest.mark.parametrize("payload", [{}, {"data": {}}, {"data": None}])
    def test_malformed_session_response_raises_agent_error(self, monkeypatch, payload):
        adapter = OpencodeAdapter("main", fake_agent_config(model="opencode/some-model"))
        monkeypatch.setattr(adapter, "_api", lambda *args, **kwargs: payload)

        with pytest.raises(AgentError, match="could not create a session"):
            adapter._create_session()

    def test_sse_keepalives_do_not_prevent_stall_timeout(self, monkeypatch):
        adapter = OpencodeAdapter("main", fake_agent_config())
        adapter._session_id = "ses_fake0000000000000000000000"
        adapter._proc = SimpleNamespace(poll=lambda: None)
        monkeypatch.setattr(opencode_module, "TURN_STALL_TIMEOUT", 0.02)

        def keepalives():
            for _ in range(5):
                time.sleep(0.01)
                yield b": keepalive\n"

        with pytest.raises(AgentError, match="stalled mid-turn"):
            list(adapter._stream_turn(keepalives()))

    def test_prompt_api_failure_raises_agent_error(self, adapter):
        adapter._session_id = "missing"
        with pytest.raises(AgentError, match="could not prompt"):
            list(adapter.send("hello"))

    def test_unknown_harness_is_rejected(self):
        with pytest.raises(NotImplementedError, match="tmux"):
            create_adapter("main", AgentConfig(harness="tmux"))

    def test_first_agent_selection(self):
        config = Config()
        config.agents = {"six": AgentConfig(), "seven": AgentConfig()}
        name, _ = first_agent(config)
        assert name == "six"


class FakeAdapter(AgentAdapter):
    """Scriptable adapter for loop tests."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.prompts = []
        self.restarts = 0
        self._alive = True

    def start(self):
        self._alive = True
        self.restarts += 1

    def stop(self):
        self._alive = False

    @property
    def alive(self):
        return self._alive

    def send(self, prompt):
        self.prompts.append(prompt)
        action = self._turns.pop(0)
        if isinstance(action, str):
            yield action
            return
        if action == ("die",):
            self._alive = False
            raise AgentError("process died")
        raise action


class FakeOutput:
    """Stands in for OutputPipeline; records everything spoken."""

    def __init__(self):
        self.spoken = []

    def speak_stream(self, stream):
        self.spoken.append("".join(stream))

    def speak(self, text):
        self.spoken.append(text)

    def wait_until_idle(self, timeout=None):
        return True


class TestConversationLoop:
    def test_normal_turn_speaks_the_response(self):
        adapter = FakeAdapter(["The tests **pass**."])
        output = FakeOutput()
        ConversationLoop(adapter, output).handle_transcript("run the tests")
        assert adapter.prompts == ["run the tests"]
        assert output.spoken == ["The tests **pass**."]

    def test_agent_error_with_live_agent_speaks_feedback(self):
        adapter = FakeAdapter([AgentError("timeout")])
        output = FakeOutput()
        ConversationLoop(adapter, output).handle_transcript("hello")
        assert any("not responding" in s for s in output.spoken)

    def test_dead_agent_restarts_and_retries(self):
        adapter = FakeAdapter([("die",), "Recovered answer."])
        adapter.restarts = 0
        output = FakeOutput()
        ConversationLoop(adapter, output).handle_transcript("do the thing")
        assert adapter.restarts == 1
        assert adapter.prompts == ["do the thing", "do the thing"]
        assert output.spoken[-1] == "Recovered answer."
        assert any("Restarting" in s for s in output.spoken)

    def test_dead_agent_uses_supplied_restart_callback_before_retrying(self):
        adapter = FakeAdapter([("die",), "Recovered answer."])
        output = FakeOutput()
        restarted = []

        def restart():
            restarted.append(True)
            adapter.start()
            return True

        ConversationLoop(adapter, output, restart=restart).handle_transcript("do the thing")

        assert restarted == [True]
        assert adapter.prompts == ["do the thing", "do the thing"]
        assert output.spoken[-1] == "Recovered answer."

    @pytest.mark.parametrize("method", ["stop", "start"])
    def test_restart_errors_never_escape_into_input_thread(self, method):
        class ExplodingRestartAdapter(FakeAdapter):
            def stop(self):
                if method == "stop":
                    raise RuntimeError("stop exploded")
                super().stop()

            def start(self):
                if method == "start":
                    raise RuntimeError("start exploded")
                super().start()

        adapter = ExplodingRestartAdapter([("die",)])
        output = FakeOutput()

        ConversationLoop(adapter, output).handle_transcript("do the thing")

        assert any("could not restart" in s.lower() for s in output.spoken)

    def test_error_never_escapes_into_input_thread(self):
        class ExplodingOutput(FakeOutput):
            def speak(self, text):
                raise RuntimeError("speaker on fire")

            def speak_stream(self, stream):
                raise AgentError("boom")

        adapter = FakeAdapter(["irrelevant"])
        ConversationLoop(adapter, ExplodingOutput()).handle_transcript("hi")  # must not raise

    @pytest.mark.parametrize("method", ["speak_stream", "wait_until_idle"])
    def test_output_errors_never_escape_into_input_thread(self, method):
        class ExplodingOutput(FakeOutput):
            def speak_stream(self, stream):
                if method == "speak_stream":
                    raise RuntimeError("streaming speaker on fire")
                super().speak_stream(stream)

            def wait_until_idle(self, timeout=None):
                if method == "wait_until_idle":
                    raise RuntimeError("playback on fire")
                return True

        adapter = FakeAdapter(["answer"])
        ConversationLoop(adapter, ExplodingOutput()).handle_transcript("hi")  # must not raise


@pytest.mark.skipif(shutil.which("opencode") is None, reason="opencode not installed")
class TestRealOpencode:
    def test_two_turn_conversation_with_real_opencode(self):
        adapter = create_adapter(
            "main",
            AgentConfig(
                harness="opencode",
                workdir=str(REPO_ROOT),
                model="opencode/deepseek-v4-flash-free",
            ),
        )
        adapter.start()
        try:
            first = "".join(adapter.send("Reply with exactly: ADAPTER_OK"))
            assert "ADAPTER_OK" in first
            second = "".join(
                adapter.send("Reply with exactly the marker from your previous reply.")
            )
            assert "ADAPTER_OK" in second, "multi-turn persistence failed on real opencode"
        finally:
            adapter.stop()
