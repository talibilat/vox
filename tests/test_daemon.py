"""Daemon lifecycle tests, driven through the real CLI."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

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


def test_config_error_exit_code(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("stt: {backend: remote}")
    result = cli("status", config=bad)
    assert result.returncode == 2
    assert "config error" in result.stderr
    assert "stt.backend" in result.stderr
