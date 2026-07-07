"""Daemon lifecycle: start, stop, status, and the foreground run loop.

The daemon is the long-lived process that owns lifecycle plumbing, the audio
pipeline, and the configured Phase 1 agent process when the voice loop is
enabled.
"""

from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from earshot.config import Config

logger = logging.getLogger("earshot")
START_READY_TIMEOUT_SECONDS = 30


def _pid_path(config: Config) -> Path:
    return Path(config.daemon.pid_file).expanduser()


def _ready_path(config: Config) -> Path:
    return _pid_path(config).with_suffix(_pid_path(config).suffix + ".ready")


def _log_path(config: Config) -> Path:
    return Path(config.daemon.log_file).expanduser()


def _looks_like_earshot(pid: int) -> bool:
    """Best-effort check that a PID actually belongs to an earshot process,
    so a recycled PID number is not mistaken for a running daemon."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return True  # cannot verify; err on the side of "running"
    if result.returncode != 0:
        return False
    try:
        argv = shlex.split(result.stdout.strip())
    except ValueError:
        return False
    return _is_earshot_daemon_argv(argv)


def _is_earshot_daemon_argv(argv: list[str]) -> bool:
    try:
        module_index = argv.index("-m")
    except ValueError:
        return False
    if module_index + 1 >= len(argv) or argv[module_index + 1] != "earshot.cli":
        return False
    command_args = argv[module_index + 2 :]
    if "--config" in command_args:
        config_index = command_args.index("--config")
        if config_index + 1 >= len(command_args):
            return False
        command_args = command_args[:config_index] + command_args[config_index + 2 :]
    return command_args == ["run"] or command_args == ["start", "--foreground"]


def _unlink_pid_if_matches(pid_file: Path, pid: int) -> None:
    try:
        if pid_file.read_text().strip() == str(pid):
            pid_file.unlink(missing_ok=True)
    except FileNotFoundError:
        pass


def read_pid(config: Config) -> int | None:
    """Return the running daemon's PID, or None. Cleans up stale PID files,
    including a live-but-unowned PID left behind by PID-number reuse."""
    pid_file = _pid_path(config)
    try:
        pid = int(pid_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    try:
        os.kill(pid, 0)  # existence check only
    except ProcessLookupError:
        _unlink_pid_if_matches(pid_file, pid)
        return None
    except PermissionError:
        pass  # process exists; fall through to the identity check
    if not _looks_like_earshot(pid):
        _unlink_pid_if_matches(pid_file, pid)
        return None
    return pid


def start(config: Config, config_path: str | None) -> int:
    """Spawn the daemon as a detached child. Returns its PID."""
    existing = read_pid(config)
    if existing is not None:
        raise RuntimeError(f"daemon already running (pid {existing})")
    ready_file = _ready_path(config)
    ready_file.unlink(missing_ok=True)
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
    # The child writes its ready file after startup completes so `start` only
    # returns once the daemon can handle work.
    deadline = time.time() + START_READY_TIMEOUT_SECONDS
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"daemon exited immediately (exit code {proc.returncode}); see {log_file}"
            )
        if read_pid(config) == proc.pid:
            try:
                if int(ready_file.read_text().strip()) == proc.pid:
                    return proc.pid
            except (FileNotFoundError, ValueError):
                pass
        time.sleep(0.05)
    raise RuntimeError(
        f"daemon did not report ready within {START_READY_TIMEOUT_SECONDS}s; see {log_file}"
    )


def interrupt(config: Config) -> int:
    """Send the push-to-interrupt signal (SIGUSR1) to the running daemon."""
    pid = read_pid(config)
    if pid is None:
        raise RuntimeError("daemon is not running")
    os.kill(pid, signal.SIGUSR1)
    return pid


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
    existing = read_pid(config)
    if existing is not None and existing != os.getpid():
        raise RuntimeError(f"daemon already running (pid {existing})")

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
    ready_file = _ready_path(config)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    ready_file.unlink(missing_ok=True)
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

    pipeline = None
    pipeline_thread = None
    pipeline_error: BaseException | None = None
    fleet = None
    # SIGUSR1 is the push-to-interrupt escape hatch (`earshot interrupt`).
    # Registered unconditionally and BEFORE the voice loop exists: the
    # default SIGUSR1 action would otherwise terminate a daemon that is
    # running without a wake model.
    signal.signal(signal.SIGUSR1, lambda *_: pipeline and pipeline.request_interrupt())
    if config.wake_word.model_path:
        from earshot.agents import first_agent
        from earshot.barge import InterruptibleVoiceLoop
        from earshot.conductor import Fleet
        from earshot.output import OutputPipeline

        # Single-agent operation is the degenerate case of a one-agent
        # fleet: one code path from Phase 1 through 16 agents.
        fleet = Fleet(config)
        fleet.start_all()
        active_name, _ = first_agent(config)
        active = fleet.get(active_name)
        if active.status == "dead":
            raise RuntimeError(f"the active agent {active_name!r} failed to start")
        # The conversation loop owns turn-triggered recovery for the active
        # agent; the supervisor owns everyone else, and the router moves the
        # exemption as spoken addressing switches agents.
        fleet.start_supervision(active_name=active_name)

        from earshot.conductor import Router

        output = OutputPipeline(config)
        router = Router(fleet, output)
        pipeline = InterruptibleVoiceLoop(config, router, output)

        def _run_pipeline() -> None:
            nonlocal pipeline_error
            try:
                pipeline.run()
            except Exception as exc:
                pipeline_error = exc
                logger.exception("voice loop failed")

        pipeline_thread = threading.Thread(target=_run_pipeline, daemon=True, name="voice-loop")
        pipeline_thread.start()
        logger.info(
            "voice loop listening (wake word: %r, active agent: %s)",
            config.wake_word.phrase,
            active_name,
        )
    else:
        logger.info("voice loop disabled (wake_word.model_path is not set)")

    ready_file.write_text(str(os.getpid()))

    try:
        while not stopping:
            if pipeline_error is not None:
                raise RuntimeError("input pipeline failed") from pipeline_error
            time.sleep(0.2)
    finally:
        if pipeline is not None:
            pipeline.stop()
            pipeline_thread.join(timeout=5)
        if fleet is not None:
            fleet.stop_all()
        # Only remove the PID file this process owns; never clobber a file
        # that another daemon has since written.
        try:
            if pid_file.read_text().strip() == str(os.getpid()):
                pid_file.unlink()
        except (FileNotFoundError, ValueError):
            pass
        try:
            if ready_file.read_text().strip() == str(os.getpid()):
                ready_file.unlink()
        except (FileNotFoundError, ValueError):
            pass
        logger.info("earshot daemon stopped")
