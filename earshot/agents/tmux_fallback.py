"""The tmux fallback transport: insurance for harnesses without a healthy
native surface.

Per docs/control-plane-verdicts.md, NO current harness needs this (all
three native surfaces passed validation), so this is the issue's reduced
scope: a minimal, generic adapter kept for whatever harness comes next.
It is just another AgentAdapter: the Conductor creates and owns a tmux
session, launches the harness CLI inside it, delivers prompts with
send-keys (literal mode; paste-buffer for multiline), captures output with
capture-pane plus ANSI stripping, and detects completion by output
stability (the capture-pane idle pattern from NTM/Tmux-Orchestrator).

Activation: setting `tmux_pane` on an agent in config selects this adapter;
the value names the tmux session the Conductor owns for it.

Known limits of a best-effort text transport (accepted for insurance-tier
use): completion is quiescence-based (a harness that pauses mid-answer for
longer than the stability window ends the turn early), and TUI redraws are
only cleaned up as far as ANSI stripping goes.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

from earshot.agents.base import AgentAdapter, AgentError
from earshot.config import AgentConfig

logger = logging.getLogger("earshot.agents.tmux")

DEFAULT_COMMANDS = {"opencode": "opencode", "claude-code": "claude", "codex": "codex"}
POLL_SECONDS = 0.4
STABLE_SECONDS = 2.0  # unchanged output for this long = turn finished
TURN_TIMEOUT = 300.0
_ANSI = re.compile(
    r"\x1b\].*?(?:\x07|\x1b\\)|\x1b\[[0-9;?]*[a-zA-Z]|\x1b[=>]|[\x00-\x08\x0b-\x1f\x7f]"
)


def _strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def _fresh_line_start(before_lines: list[str], after_lines: list[str]) -> int:
    common = 0
    for old, new in zip(before_lines, after_lines, strict=False):
        if old != new:
            break
        common += 1
    if common:
        return common
    for overlap in range(min(len(before_lines), len(after_lines)), 0, -1):
        if before_lines[-overlap:] == after_lines[:overlap]:
            return overlap
    return 0


def _stable_wait_state(
    before: str, current: str, stable_since: float | None
) -> tuple[float | None, bool]:
    if current == before:
        return stable_since, False
    now = time.time()
    if stable_since is None:
        return now, False
    return stable_since, now - stable_since >= STABLE_SECONDS


class TmuxAgentAdapter(AgentAdapter):
    def __init__(self, name: str, config: AgentConfig):
        if shutil.which("tmux") is None:
            raise AgentError("tmux is not installed; the fallback transport needs it")
        self._name = name
        self._config = config
        self._session = config.tmux_pane or f"earshot-{name}"

    @property
    def alive(self) -> bool:
        return (
            subprocess.run(
                ["tmux", "has-session", "-t", self._session],
                capture_output=True,
                timeout=10,
            ).returncode
            == 0
        )

    def start(self) -> None:
        command = self._config.command or DEFAULT_COMMANDS.get(self._config.harness)
        if not command:
            raise AgentError(f"agent {self._name} has no command for the tmux fallback")
        self.stop()  # never adopt a stale session; the Conductor owns it
        result = self._tmux(
            "new-session",
            "-d",
            "-s",
            self._session,
            "-x",
            "220",
            "-y",
            "50",
            "-c",
            str(Path(self._config.workdir).expanduser()),
            command,
        )
        if result.returncode != 0:
            raise AgentError(
                f"could not launch agent {self._name} in tmux: {result.stderr.strip()}"
            )
        time.sleep(0.5)  # give the CLI a beat to draw its prompt
        if not self.alive:
            raise AgentError(f"agent {self._name} exited during startup (tmux session gone)")
        logger.info("agent %s ready in tmux session %s", self._name, self._session)

    def stop(self) -> None:
        subprocess.run(
            ["tmux", "kill-session", "-t", self._session], capture_output=True, timeout=10
        )

    def send(self, prompt: str) -> Iterator[str]:
        if not self.alive:
            raise AgentError(f"agent {self._name} is not running")
        before = self._capture()
        self._deliver(prompt)
        yield from self._wait_for_stable_output(before, prompt)

    # --- internals -----------------------------------------------------

    def _tmux(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["tmux", *args], capture_output=True, text=True, timeout=15)

    def _deliver(self, prompt: str) -> None:
        if "\n" in prompt:
            # Multiline goes through a paste buffer; send-keys would submit
            # on the first newline.
            self._check_delivery(
                subprocess.run(
                    ["tmux", "load-buffer", "-b", f"earshot-{self._name}", "-"],
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=15,
                )
            )
            self._check_delivery(
                self._tmux("paste-buffer", "-d", "-b", f"earshot-{self._name}", "-t", self._session)
            )
        else:
            # Literal mode so the prompt's characters are never interpreted
            # as key names.
            self._check_delivery(self._tmux("send-keys", "-t", self._session, "-l", "--", prompt))
        self._check_delivery(self._tmux("send-keys", "-t", self._session, "Enter"))

    def _check_delivery(self, result: subprocess.CompletedProcess) -> None:
        if result.returncode != 0:
            detail = (
                result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
            )
            raise AgentError(f"could not deliver prompt to agent {self._name}: {detail}")

    def _capture(self) -> str:
        result = self._tmux("capture-pane", "-t", self._session, "-p", "-S", "-2000")
        if result.returncode != 0:
            raise AgentError(f"lost the tmux session for agent {self._name}")
        return result.stdout

    def _wait_for_stable_output(self, before: str, prompt: str) -> Iterator[str]:
        deadline = time.time() + TURN_TIMEOUT
        last = before
        stable_since: float | None = None
        while time.time() < deadline:
            time.sleep(POLL_SECONDS)
            if not self.alive:
                raise AgentError(f"agent {self._name} died mid-turn")
            current = self._capture()
            if current != last:
                chunk = self._extract_response(last, current, prompt)
                last = current
                stable_since = None
                if chunk:
                    yield chunk
                continue
            stable_since, is_stable = _stable_wait_state(before, current, stable_since)
            if is_stable:
                return
        raise AgentError(f"agent {self._name} stalled mid-turn")

    def _extract_response(self, before: str, after: str, prompt: str) -> str:
        before_lines = before.rstrip("\n").splitlines()
        after_lines = after.rstrip("\n").splitlines()
        common = _fresh_line_start(before_lines, after_lines)
        fresh = [_strip_ansi(line) for line in after_lines[common:]]
        # Drop echoed prompt lines so the user's own words are not read back.
        prompt_lines = {line.strip() for line in prompt.splitlines() if line.strip()}
        cleaned = [line for line in fresh if line.strip() and line.strip() not in prompt_lines]
        return "\n".join(cleaned).strip()
