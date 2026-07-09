"""The adapter contract every harness implements.

Designed against all three known control surfaces so #9 does not force
churn: opencode serve (HTTP + SSE events), codex app-server/mcp-server
(JSON-RPC over stdio), and Claude Code --print stream-json with --resume.
All three reduce to the same shape: own a child process, send one prompt
into a persistent session, stream text back, and know when the turn ended.

Nothing harness-specific may leak above this interface: the conversation
loop sees markdown text chunks and exceptions, and only those.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from collections.abc import Iterator


class AgentError(Exception):
    """The agent failed mid-turn: process died, API timed out, or the
    response stream stalled. The message is speakable to the user."""


def _stop_process(proc: subprocess.Popen, timeout: float = 10) -> None:
    """Terminate a child process, escalating to kill if it will not exit."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


class AgentAdapter(ABC):
    @abstractmethod
    def start(self) -> None:
        """Spawn and own the agent process; create the persistent session.

        Raises AgentError when the harness cannot be started.
        """

    @abstractmethod
    def send(self, prompt: str) -> Iterator[str]:
        """Send one prompt into the session and yield markdown text chunks
        as they stream back. The iterator ends when the turn is complete.
        Consecutive send() calls continue the same session (multi-turn).

        Raises AgentError mid-iteration on process death, timeout, or an
        in-band harness error.
        """

    @abstractmethod
    def stop(self) -> None:
        """Shut the agent process down cleanly. Idempotent."""

    @property
    @abstractmethod
    def alive(self) -> bool:
        """True while the owned agent process is running."""
