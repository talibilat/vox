"""Per-agent output watchers: buffer everything, speak only on request.

The anti-audio-soup rule: with 16 agents working, nothing is EVER spoken
unprompted. Each agent gets a watcher thread that executes its dispatched
turns, buffers the full response (size-capped ring buffer), and updates the
registry status (busy while a turn runs, finished when a response is
waiting, idle once it has been read). Only the ACTIVE agent's turns are
also streamed to the speaker as they arrive; everyone else works silently.

Idle/finished detection rides the adapters' native completion signals (the
turn stream ending is the harness's own turn-complete event, per
docs/control-plane-verdicts.md); no polling heuristics are needed because
no current harness was flagged for the fallback path.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from collections.abc import Callable

from earshot.agents import AgentError
from earshot.conductor.lifecycle import Fleet
from earshot.conductor.registry import AgentRecord
from earshot.output import OutputPipeline

logger = logging.getLogger("earshot.watchers")

MAX_BUFFERED_RESPONSES = 8  # ring depth per agent
MAX_RESPONSE_CHARS = 100_000  # per-response cap; 16 chatty agents stay bounded
TRUNCATED_SUFFIX = " (response truncated)"


class _BoundedResponse:
    def __init__(self) -> None:
        self._chunks: list[str] = []
        self._chars = 0
        self._truncated = False

    def append(self, chunk: str) -> None:
        if self._chars < MAX_RESPONSE_CHARS:
            remaining = MAX_RESPONSE_CHARS - self._chars
            self._chunks.append(chunk[:remaining])
            self._chars += min(len(chunk), remaining)
            if len(chunk) > remaining:
                self._truncated = True
        else:
            self._truncated = True

    def text(self) -> str:
        response = "".join(self._chunks)
        if self._truncated:
            response += TRUNCATED_SUFFIX
        return response


class AgentWatcher:
    def __init__(
        self,
        record: AgentRecord,
        output: OutputPipeline,
        is_active: Callable[[str], bool],
    ):
        self._record = record
        self._output = output
        self._is_active = is_active
        self._commands: queue.Queue[str | None] = queue.Queue()
        self._responses: deque[str] = deque(maxlen=MAX_BUFFERED_RESPONSES)
        self._lock = threading.Lock()
        self._unread = False
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"watcher-{record.name}"
        )
        self._thread.start()

    def dispatch(self, command: str) -> None:
        """Queue one turn for this agent."""
        self._commands.put(command)

    def latest_response(self) -> str | None:
        """The newest buffered response; reading acknowledges it (finished
        becomes idle)."""
        with self._lock:
            latest = self._responses[-1] if self._responses else None
            self._unread = False
        if self._record.status == "finished":
            self._record.mark("idle")
        return latest

    @property
    def has_unread(self) -> bool:
        return self._unread

    def stop(self) -> None:
        self._commands.put(None)
        self._thread.join(timeout=5)

    # --- internals -----------------------------------------------------

    def _run(self) -> None:
        while True:
            command = self._commands.get()
            if command is None:
                return
            self._record.mark("busy")
            try:
                self._run_turn(command)
                self._record.mark("finished")
            except AgentError as error:
                logger.warning("agent %s turn failed: %s", self._record.name, error)
                self._record_failure(error)
            except Exception as error:
                logger.exception("watcher for %s crashed on a turn", self._record.name)
                self._record_failure(error)

    def _run_turn(self, command: str) -> None:
        response = _BoundedResponse()

        def tee():
            for chunk in self._record.adapter.send(command):
                response.append(chunk)
                yield chunk

        if self._is_active(self._record.name):
            self._output.speak_stream(tee())
            self._output.wait_until_idle()
        else:
            for _ in tee():
                pass  # buffer silently; nothing is spoken unprompted
        self._buffer(response.text())

    def _record_failure(self, error: Exception) -> None:
        self._buffer(f"The last request failed: {error}")
        self._record.mark("dead" if not self._record.adapter.alive else "finished")
        if self._is_active(self._record.name):
            self._say(f"{self._record.name} is not responding.")

    def _buffer(self, response: str) -> None:
        if len(response) > MAX_RESPONSE_CHARS:
            response = response[:MAX_RESPONSE_CHARS] + TRUNCATED_SUFFIX
        with self._lock:
            self._responses.append(response)
            self._unread = True

    def _say(self, sentence: str) -> None:
        try:
            self._output.speak(sentence)
            self._output.wait_until_idle()
        except Exception:
            logger.exception("could not speak watcher message")


class WatcherPool:
    """One watcher per fleet member; the router's dispatch/read/status API."""

    def __init__(self, fleet: Fleet, output: OutputPipeline):
        self._fleet = fleet
        self._output = output
        self._active_probe: Callable[[str], bool] = lambda _name: False
        self._watchers = {
            record.name: AgentWatcher(record, output, self._is_active)
            for record in fleet.registry.records()
        }

    def set_active_probe(self, probe: Callable[[str], bool]) -> None:
        """Wired after the Router exists (the router owns active-agent state)."""
        self._active_probe = probe

    def dispatch(self, name: str, command: str) -> None:
        self._watchers[name].dispatch(command)

    def latest_response_text(self, name: str) -> str:
        latest = self._watchers[name].latest_response()
        if latest is None:
            return f"{name} has not said anything yet."
        return latest

    def status_line(self) -> str:
        from earshot.conductor.status import spoken_status

        return spoken_status(self._fleet.statuses())

    def stop(self) -> None:
        for watcher in self._watchers.values():
            watcher.stop()

    def _is_active(self, name: str) -> bool:
        return self._active_probe(name)
