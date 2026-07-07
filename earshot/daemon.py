"""Daemon lifecycle: start, stop, status, and the foreground run loop.

The daemon is the long-lived process that will own the audio pipeline and
every agent process (later issues). This module only implements lifecycle
plumbing: PID-file management, detached start, clean stop, and logging.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from earshot.config import Config

logger = logging.getLogger("earshot")


def _pid_path(config: Config) -> Path:
    return Path(config.daemon.pid_file).expanduser()


def _log_path(config: Config) -> Path:
    return Path(config.daemon.log_file).expanduser()


def read_pid(config: Config) -> int | None:
    """Return the running daemon's PID, or None. Cleans up stale PID files."""
    pid_file = _pid_path(config)
    try:
        pid = int(pid_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    try:
        os.kill(pid, 0)  # existence check only
    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        return None
    except PermissionError:
        return pid
    return pid


def start(config: Config, config_path: str | None) -> int:
    """Spawn the daemon as a detached child. Returns its PID."""
    existing = read_pid(config)
    if existing is not None:
        raise RuntimeError(f"daemon already running (pid {existing})")
    log_file = _log_path(config)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "earshot.cli"]
    if config_path:
        cmd += ["--config", config_path]
    cmd += ["run"]
    with open(log_file, "ab") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # survive the parent's terminal
        )
    # The child writes its own PID file once its loop is up; wait briefly so
    # `earshot start && earshot status` behaves as expected.
    deadline = time.time() + 5
    while time.time() < deadline:
        if read_pid(config) == proc.pid:
            return proc.pid
        if proc.poll() is not None:
            raise RuntimeError(
                f"daemon exited immediately (exit code {proc.returncode}); see {log_file}"
            )
        time.sleep(0.05)
    raise RuntimeError(f"daemon did not report ready within 5s; see {log_file}")


def stop(config: Config) -> int:
    """SIGTERM the daemon and wait for it to exit. Returns the stopped PID."""
    pid = read_pid(config)
    if pid is None:
        raise RuntimeError("daemon is not running")
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 10
    while time.time() < deadline:
        if read_pid(config) is None:
            return pid
        time.sleep(0.05)
    raise RuntimeError(f"daemon (pid {pid}) did not exit within 10s")


def run(config: Config) -> None:
    """The daemon main loop, in the current process (foreground mode uses
    this directly; `start` runs it in a detached child)."""
    log_file = _log_path(config)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.FileHandler(log_file)]
    if sys.stderr.isatty():
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )

    pid_file = _pid_path(config)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))

    stopping = False

    def _handle_term(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)

    logger.info(
        "earshot daemon started (pid %d, agents: %s)", os.getpid(), ", ".join(config.agents)
    )
    try:
        # Placeholder loop: the audio pipeline (#5, #6, #7) and agent
        # lifecycle (#8, #11) plug in here.
        while not stopping:
            time.sleep(0.2)
    finally:
        pid_file.unlink(missing_ok=True)
        logger.info("earshot daemon stopped")
