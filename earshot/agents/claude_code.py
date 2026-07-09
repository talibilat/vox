"""The Claude Code adapter: drives `claude --print` in stream-json mode.

Process model differs from opencode: Claude Code runs ONE PROCESS PER TURN.
The adapter owns each turn's process and persists the conversation across
turns with `--resume <session_id>` (the session id is captured from the
first turn's events). `alive` therefore means "the adapter is started and
usable", not "a process is currently running"; a turn failure surfaces as
AgentError from send(), never as a dead long-lived process.

Event grammar observed live on claude 2.1.197:
- {"type":"system","subtype":"init",...,"session_id":...} first
- {"type":"assistant","message":{"content":[{"type":"text","text":...}]}}
  one event per content block (thinking blocks arrive separately and are
  skipped)
- {"type":"result","subtype":"success","is_error":false,"result":...,
   "session_id":...} terminates the turn
- hook/system noise events are interleaved and ignored
"""

from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

from earshot.agents.base import AgentAdapter, AgentError, _stop_process
from earshot.config import AgentConfig

logger = logging.getLogger("earshot.agents.claude_code")

TURN_TIMEOUT = 600.0  # coding turns can run long; the stall guard is per-line
LINE_STALL_TIMEOUT = 120.0  # max quiet time between output lines


def _decode_json_event(line: str) -> dict | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _assistant_text(event: dict) -> Iterator[str]:
    for block in event.get("message", {}).get("content", []):
        if block.get("type") == "text" and block.get("text"):
            yield block["text"]


class ClaudeCodeAdapter(AgentAdapter):
    def __init__(self, name: str, config: AgentConfig):
        self._name = name
        self._config = config
        self._session_id: str | None = None
        self._started = False
        self._turn_proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    @property
    def alive(self) -> bool:
        return self._started

    def start(self) -> None:
        binary = shlex.split(self._config.command)[0] if self._config.command else "claude"
        if shutil.which(binary) is None:
            raise AgentError(f"could not launch agent {self._name}: {binary} not found")
        self._session_id = None
        self._started = True
        logger.info("agent %s (claude-code) ready; sessions start on first turn", self._name)

    def stop(self) -> None:
        self._started = False
        with self._lock:
            proc = self._turn_proc
        if proc is not None:
            _stop_process(proc)

    def send(self, prompt: str) -> Iterator[str]:
        if not self._started:
            raise AgentError(f"agent {self._name} is not running")
        cmd = self._build_command()
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=Path(self._config.workdir).expanduser(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError as exc:
            raise AgentError(f"could not launch agent {self._name}: {exc}") from exc
        with self._lock:
            self._turn_proc = proc
        try:
            message = {
                "type": "user",
                "message": {"role": "user", "content": [{"type": "text", "text": prompt}]},
            }
            try:
                proc.stdin.write(json.dumps(message) + "\n")
                proc.stdin.close()
            except (OSError, ValueError, AttributeError) as exc:
                _stop_process(proc)
                raise AgentError(f"agent {self._name} is not accepting input: {exc}") from exc
            yield from self._stream_turn(proc)
        finally:
            with self._lock:
                self._turn_proc = None
            if proc.poll() is None:
                _stop_process(proc)

    # --- internals -----------------------------------------------------

    def _build_command(self) -> list[str]:
        if self._config.command:
            cmd = shlex.split(self._config.command)
        else:
            cmd = ["claude"]
        cmd += [
            "--print",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            # Voice-driven turns cannot answer interactive permission
            # prompts; acceptEdits keeps file edits flowing. Full tool
            # permission policy is the user's to widen via `command`.
            "--permission-mode",
            "acceptEdits",
        ]
        if self._config.model:
            cmd += ["--model", self._config.model]
        if self._session_id:
            cmd += ["--resume", self._session_id]
        return cmd

    def _stream_turn(self, proc: subprocess.Popen) -> Iterator[str]:
        got_result = False
        for event in self._turn_events(proc):
            etype = event.get("type")
            if etype == "assistant":
                yield from _assistant_text(event)
            elif etype == "result":
                got_result = True
                self._raise_for_result_error(event)
                break
        exit_code = proc.wait(timeout=10)
        if not got_result:
            raise AgentError(
                f"agent {self._name} turn ended without a result (exit code {exit_code})"
            )

    def _turn_events(self, proc: subprocess.Popen) -> Iterator[dict]:
        for line in self._lines_with_stall_guard(proc):
            event = _decode_json_event(line)
            if event is None:
                continue
            if session := event.get("session_id"):
                self._session_id = session
            yield event

    def _raise_for_result_error(self, event: dict) -> None:
        if event.get("is_error"):
            raise AgentError(
                f"agent {self._name} reported an error: {str(event.get('result', ''))[:200]}"
            )

    def _lines_with_stall_guard(self, proc: subprocess.Popen) -> Iterator[str]:
        """Yield stdout lines, raising AgentError if output stalls."""
        result: dict = {}

        def _read():
            result["line"] = proc.stdout.readline()

        while True:
            reader = threading.Thread(target=_read, daemon=True)
            reader.start()
            reader.join(timeout=LINE_STALL_TIMEOUT)
            if reader.is_alive():
                proc.kill()
                raise AgentError(f"agent {self._name} stalled mid-turn")
            line = result.get("line", "")
            if not line:
                if proc.poll() is not None and proc.returncode != 0:
                    raise AgentError(
                        f"agent {self._name} died mid-turn (exit code {proc.returncode})"
                    )
                return
            yield line.strip()
