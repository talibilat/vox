"""Daemon lifecycle tests, driven through the real CLI."""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from earshot import daemon
from earshot.config import Config

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def isolated_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
daemon:
  log_file: {tmp_path / "earshot.log"}
  pid_file: {tmp_path / "earshot.pid"}
"""
    )
    return config_path


@pytest.fixture()
def daemon_config(tmp_path):
    config = Config()
    config.daemon.log_file = str(tmp_path / "earshot.log")
    config.daemon.pid_file = str(tmp_path / "earshot.pid")
    return config


def cli(*args, config=None):
    cmd = [sys.executable, "-m", "earshot.cli"]
    if config:
        cmd += ["--config", str(config)]
    cmd += list(args)
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    return subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=REPO_ROOT, timeout=30)


def test_start_status_stop_cycle(isolated_config, tmp_path):
    result = cli("start", config=isolated_config)
    assert result.returncode == 0, result.stderr
    assert "started" in result.stdout
    try:
        status = cli("status", config=isolated_config)
        assert status.returncode == 0
        assert "running" in status.stdout

        # PID file exists and points at a live process
        pid = int((tmp_path / "earshot.pid").read_text())
        os.kill(pid, 0)
    finally:
        stopped = cli("stop", config=isolated_config)
    assert stopped.returncode == 0, stopped.stderr
    assert "stopped" in stopped.stdout
    assert not (tmp_path / "earshot.pid").exists()
    assert "daemon started" in (tmp_path / "earshot.log").read_text()

    status = cli("status", config=isolated_config)
    assert status.returncode == 1
    assert "not running" in status.stdout


def test_double_start_refused(isolated_config):
    assert cli("start", config=isolated_config).returncode == 0
    try:
        second = cli("start", config=isolated_config)
        assert second.returncode == 1
        assert "already running" in second.stderr
    finally:
        cli("stop", config=isolated_config)


def test_stop_without_daemon(isolated_config):
    result = cli("stop", config=isolated_config)
    assert result.returncode == 1
    assert "not running" in result.stderr


def test_interrupt_without_daemon(isolated_config):
    result = cli("interrupt", config=isolated_config)
    assert result.returncode == 1
    assert "not running" in result.stderr


def test_interrupt_does_not_kill_daemon_without_voice_loop(isolated_config):
    # SIGUSR1's default action terminates a process; the daemon must install
    # a handler even when no wake model is configured, or `earshot interrupt`
    # would kill it instead of being a no-op.
    assert cli("start", config=isolated_config).returncode == 0
    try:
        result = cli("interrupt", config=isolated_config)
        assert result.returncode == 0
        assert "interrupt sent" in result.stdout
        time.sleep(0.3)
        status = cli("status", config=isolated_config)
        assert status.returncode == 0, "SIGUSR1 killed the daemon"
    finally:
        cli("stop", config=isolated_config)


def test_stale_pid_file_is_cleaned(isolated_config, tmp_path):
    # A PID that cannot exist as a live process
    (tmp_path / "earshot.pid").write_text("99999999")
    status = cli("status", config=isolated_config)
    assert status.returncode == 1
    assert "not running" in status.stdout
    assert not (tmp_path / "earshot.pid").exists()


def test_foreground_runs_and_handles_sigterm(isolated_config, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    cmd = [sys.executable, "-m", "earshot.cli", "--config", str(isolated_config)]
    proc = subprocess.Popen(
        cmd + ["start", "--foreground"],
        env=env,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.time() + 10
        pid_file = tmp_path / "earshot.pid"
        while time.time() < deadline and not pid_file.exists():
            time.sleep(0.05)
        assert pid_file.exists(), "foreground daemon never wrote its PID file"
        assert int(pid_file.read_text()) == proc.pid
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    assert proc.returncode == 0
    assert not pid_file.exists()


def test_foreground_refused_when_daemon_running(isolated_config, tmp_path):
    assert cli("start", config=isolated_config).returncode == 0
    try:
        pid_before = (tmp_path / "earshot.pid").read_text()
        second = cli("start", "--foreground", config=isolated_config)
        assert second.returncode == 1
        assert "already running" in second.stderr
        # The refused foreground process must not touch the owner's PID file.
        assert (tmp_path / "earshot.pid").read_text() == pid_before
        assert cli("status", config=isolated_config).returncode == 0
    finally:
        cli("stop", config=isolated_config)


def test_live_but_unowned_pid_treated_as_stale(isolated_config, tmp_path):
    # A live process that is not an earshot daemon (PID-number reuse case).
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        (tmp_path / "earshot.pid").write_text(str(sleeper.pid))
        status = cli("status", config=isolated_config)
        assert status.returncode == 1
        assert "not running" in status.stdout
        assert not (tmp_path / "earshot.pid").exists()
        assert sleeper.poll() is None  # the foreign process must not be touched
    finally:
        sleeper.kill()
        sleeper.wait()


def test_pid_identity_does_not_accept_earshot_substring(monkeypatch):
    class Result:
        returncode = 0
        stdout = "tail -f /tmp/earshot.log\n"

    monkeypatch.setattr(daemon.subprocess, "run", lambda *args, **kwargs: Result())

    assert not daemon._looks_like_earshot(12345)


def test_stale_pid_cleanup_preserves_replaced_pid_file(monkeypatch, daemon_config, tmp_path):
    pid_file = tmp_path / "earshot.pid"
    pid_file.write_text("12345")

    def kill_replaces_pid_file(pid, sig):
        assert pid == 12345
        assert sig == 0
        pid_file.write_text("67890")
        raise ProcessLookupError

    monkeypatch.setattr(daemon.os, "kill", kill_replaces_pid_file)

    assert daemon.read_pid(daemon_config) is None
    assert pid_file.read_text() == "67890"


def test_detached_start_waits_for_ready_signal_not_pid_file(monkeypatch, daemon_config):
    class RunningProc:
        pid = 12345
        returncode = None

        def poll(self):
            return None

    now = 0.0

    def time_now():
        return now

    def sleep(seconds):
        nonlocal now
        now += seconds

    spawned = False

    def popen(*_args, **_kwargs):
        nonlocal spawned
        spawned = True
        return RunningProc()

    def read_pid(_config):
        return 12345 if spawned else None

    monkeypatch.setattr(daemon.subprocess, "Popen", popen)
    monkeypatch.setattr(daemon, "read_pid", read_pid)
    monkeypatch.setattr(daemon.time, "time", time_now)
    monkeypatch.setattr(daemon.time, "sleep", sleep)

    with pytest.raises(RuntimeError, match="did not report ready"):
        daemon.start(daemon_config, config_path=None)


def test_config_error_exit_code(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("stt: {backend: remote}")
    result = cli("status", config=bad)
    assert result.returncode == 2
    assert "config error" in result.stderr
    assert "stt.backend" in result.stderr


def test_daemon_stops_when_input_pipeline_thread_fails(monkeypatch, daemon_config):
    import earshot.agents
    import earshot.barge

    pipeline_failed = threading.Event()
    stopped = threading.Event()
    agent_stopped = threading.Event()

    class FakeAdapter:
        alive = True

        def start(self):
            pass

        def stop(self):
            agent_stopped.set()

        def send(self, _prompt):
            return iter(())

    monkeypatch.setattr(earshot.agents, "create_adapter", lambda _name, _cfg: FakeAdapter())

    import earshot.output

    class FakeOutputPipeline:
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(earshot.output, "OutputPipeline", FakeOutputPipeline)

    class FailingPipeline:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self):
            pipeline_failed.set()
            raise RuntimeError("microphone failed")

        def stop(self):
            stopped.set()

    sleep_calls_after_failure = 0

    def sleep(_seconds):
        nonlocal sleep_calls_after_failure
        if pipeline_failed.wait(timeout=0.01):
            sleep_calls_after_failure += 1
        if sleep_calls_after_failure > 5:
            raise AssertionError("daemon kept running after input pipeline failure")

    daemon_config.wake_word.model_path = "model.onnx"
    monkeypatch.setattr(earshot.barge, "InterruptibleVoiceLoop", FailingPipeline)
    monkeypatch.setattr(daemon.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(daemon.time, "sleep", sleep)

    with pytest.raises(RuntimeError, match="input pipeline failed"):
        daemon.run(daemon_config)

    assert stopped.is_set()
    assert agent_stopped.is_set(), "the owned agent process was not shut down"
    assert not Path(daemon_config.daemon.pid_file).exists()
