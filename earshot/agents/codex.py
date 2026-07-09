"""The codex adapter: drives `codex app-server` over JSON-RPC on stdio.

app-server was chosen over mcp-server after prototyping both surfaces:
app-server is a long-lived process with persistent threads, true streaming
deltas, and a turn/interrupt method, which matches the adapter contract
directly; mcp-server would wrap every turn in an MCP tool call with no
incremental output.

Protocol observed live on codex-cli 0.142.4:
- request  initialize {clientInfo} once
- request  thread/start {cwd, model?} -> result.thread.id
- request  turn/start {threadId, input: [{type: "text", text}]}
- notify   item/agentMessage/delta {threadId, delta}   (streaming text)
- notify   turn/completed {threadId, turn: {status, error}}
- notify   error {message} on failures
"""

from __future__ import annotations

import json
import logging
import queue
import shlex
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

from earshot.agents.base import AgentAdapter, AgentError, _stop_process
from earshot.config import AgentConfig

logger = logging.getLogger("earshot.agents.codex")

REQUEST_TIMEOUT = 30.0
TURN_STALL_TIMEOUT = 120.0  # max quiet time mid-turn before we call it stalled


class CodexAdapter(AgentAdapter):
    def __init__(self, name: str, config: AgentConfig):
        self._name = name
        self._config = config
        self._proc: subprocess.Popen | None = None
        self._thread_id: str | None = None
        self._next_id = 0
        self._lock = threading.Lock()
        self._responses: dict[int, queue.Queue] = {}
        self._notifications: queue.Queue = queue.Queue()
        self._reader: threading.Thread | None = None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        cmd = shlex.split(self._config.command) if self._config.command else ["codex", "app-server"]
        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=Path(self._config.workdir).expanduser(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError as exc:
            raise AgentError(f"could not launch agent {self._name}: {exc}") from exc
        try:
            self._start_reader()
            self._request("initialize", {"clientInfo": {"name": "earshot", "version": "0.1"}})
            self._thread_id = self._start_thread()
        except Exception:
            self.stop()
            raise
        logger.info(
            "agent %s ready (pid %s, thread %s)", self._name, self._proc.pid, self._thread_id
        )

    def stop(self) -> None:
        if self._proc is not None:
            _stop_process(self._proc)
        self._proc = None

    def send(self, prompt: str) -> Iterator[str]:
        if not self.alive:
            raise AgentError(f"agent {self._name} is not running")
        self._drain_notifications()
        self._request(
            "turn/start",
            {"threadId": self._thread_id, "input": [{"type": "text", "text": prompt}]},
        )
        for note in self._turn_notifications():
            method = note.get("method", "")
            params = note.get("params", {})
            if method == "item/agentMessage/delta":
                yield params.get("delta") or ""
            elif method == "turn/completed":
                turn = params.get("turn", {})
                error = turn.get("error")
                if error or turn.get("status") == "failed":
                    raise AgentError(f"agent {self._name} reported an error: {str(error)[:200]}")
                return
            elif method == "error":
                raise AgentError(
                    f"agent {self._name} reported an error: {json.dumps(params)[:200]}"
                )

    def _turn_notifications(self) -> Iterator[dict]:
        while True:
            try:
                note = self._notifications.get(timeout=TURN_STALL_TIMEOUT)
            except queue.Empty:
                raise AgentError(f"agent {self._name} stalled mid-turn") from None
            if note is None:  # reader thread saw EOF
                raise AgentError(f"agent {self._name} died mid-turn")
            params = note.get("params", {})
            if params.get("threadId") in (None, self._thread_id):
                yield note

    # --- internals -----------------------------------------------------

    def _request(self, method: str, params: dict) -> dict:
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            self._responses[request_id] = queue.Queue()
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        try:
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
        except (OSError, ValueError, AttributeError) as exc:
            raise AgentError(f"agent {self._name} is not accepting requests: {exc}") from exc
        try:
            response = self._responses[request_id].get(timeout=REQUEST_TIMEOUT)
        except queue.Empty:
            raise AgentError(f"agent {self._name} did not answer {method}") from None
        finally:
            with self._lock:
                self._responses.pop(request_id, None)
        if response is None:
            raise AgentError(f"agent {self._name} died before answering {method}")
        if "error" in response:
            raise AgentError(
                f"agent {self._name} rejected {method}: {json.dumps(response['error'])[:200]}"
            )
        return response.get("result", {})

    def _start_reader(self) -> None:
        # Fresh per-process state: a previous dead process's reader wakes on EOF
        # asynchronously, so give each process its own waiters and notifications.
        self._responses = {}
        self._notifications = queue.Queue()
        self._reader = threading.Thread(
            target=self._pump,
            args=(self._proc.stdout, self._responses, self._notifications),
            daemon=True,
            name=f"codex-{self._name}",
        )
        self._reader.start()

    def _start_thread(self) -> str:
        params: dict = {"cwd": str(Path(self._config.workdir).expanduser())}
        if self._config.model:
            params["model"] = self._config.model
        result = self._request("thread/start", params)
        thread_id = result.get("thread", {}).get("id") or result.get("threadId")
        if not thread_id:
            raise AgentError(f"agent {self._name} did not return a thread id")
        return thread_id

    def _pump(self, stream, responses: dict, notifications: queue.Queue) -> None:
        """Reader thread: route responses to callers, notifications to send().

        Operates only on the state it was handed at spawn time, so after a
        restart the dead process's EOF cannot poison the new process's
        requests.
        """
        for line in stream:
            message = _decode_pump_message(line)
            if message is None:
                continue
            self._route_pump_message(message, responses, notifications)
        self._wake_pump_waiters(responses, notifications)

    def _route_pump_message(self, message, responses: dict, notifications: queue.Queue) -> None:
        if "id" in message and ("result" in message or "error" in message):
            with self._lock:
                waiter = responses.get(message["id"])
            if waiter is not None:
                waiter.put(message)
        elif "method" in message:
            notifications.put(message)

    def _wake_pump_waiters(self, responses: dict, notifications: queue.Queue) -> None:
        # EOF: wake up everything that might be waiting on THIS process.
        with self._lock:
            waiters = list(responses.values())
        for waiter in waiters:
            waiter.put(None)
        notifications.put(None)

    def _drain_notifications(self) -> None:
        try:
            while True:
                if self._notifications.get_nowait() is None:
                    return
        except queue.Empty:
            pass


def _decode_pump_message(line: str):
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None
