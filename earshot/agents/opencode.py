"""The opencode adapter: drives `opencode serve` over its native HTTP + SSE
surface, exactly as proven by the Phase 0 control-plane spike
(docs/control-plane-spike.md). No tmux, no keystroke injection.

Spike findings this implementation encodes:
- readiness is polling GET /api/health after spawning the child;
- a session must pin its model explicitly or opencode may resolve an
  unauthorized default provider (observed HTTP 401 turns);
- the turn-completion signal is `session.next.step.ended` with
  finish == "stop" on GET /api/event (payload under `data`; `session.idle`
  exists in the schema but never fires);
- provider failures arrive in-band as events, not as failed POSTs.
"""

from __future__ import annotations

import json
import logging
import shlex
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from earshot.agents.base import AgentAdapter, AgentError, _stop_process
from earshot.config import AgentConfig

logger = logging.getLogger("earshot.agents.opencode")

READY_TIMEOUT = 30.0
TURN_STALL_TIMEOUT = 120.0  # max quiet time mid-turn before we call it stalled


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class OpencodeAdapter(AgentAdapter):
    def __init__(self, name: str, config: AgentConfig):
        self._name = name
        self._config = config
        self._port = _free_port()
        self._base = f"http://127.0.0.1:{self._port}"
        self._proc: subprocess.Popen | None = None
        self._session_id: str | None = None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        if self._config.command:
            cmd = shlex.split(self._config.command) + ["--port", str(self._port)]
        else:
            cmd = ["opencode", "serve", "--port", str(self._port)]
        workdir = Path(self._config.workdir).expanduser()
        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=workdir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise AgentError(f"could not launch agent {self._name}: {exc}") from exc
        try:
            self._wait_ready()
            self._session_id = self._create_session()
        except Exception:
            self.stop()
            raise
        logger.info(
            "agent %s ready (pid %s, session %s)", self._name, self._proc.pid, self._session_id
        )

    def stop(self) -> None:
        if self._proc is not None:
            _stop_process(self._proc)
        self._proc = None

    def send(self, prompt: str) -> Iterator[str]:
        if not self.alive:
            raise AgentError(f"agent {self._name} is not running")
        # Subscribe before prompting so no event can be missed.
        events = self._open_events()
        try:
            try:
                self._api(
                    "POST", f"/api/session/{self._session_id}/prompt", {"prompt": {"text": prompt}}
                )
            except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError) as exc:
                raise AgentError(f"could not prompt agent {self._name}: {exc}") from exc
            yield from self._stream_turn(events)
        finally:
            events.close()

    # --- internals -----------------------------------------------------

    def _wait_ready(self) -> None:
        deadline = time.time() + READY_TIMEOUT
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise AgentError(
                    f"agent {self._name} exited during startup (exit code {self._proc.returncode})"
                )
            try:
                self._api("GET", "/api/health", timeout=2)
                return
            except (urllib.error.URLError, OSError):
                time.sleep(0.25)
        raise AgentError(f"agent {self._name} did not become ready within {READY_TIMEOUT:.0f}s")

    def _create_session(self) -> str:
        # The spike showed opencode can resolve an unauthorized provider when
        # no model is pinned, so this adapter always pins one. The default
        # lives HERE because it is opencode-specific and must not leak into
        # other harnesses through the shared AgentConfig.
        from earshot.config import DEFAULT_OPENCODE_MODEL

        model = self._config.model or DEFAULT_OPENCODE_MODEL
        provider, separator, model_id = model.partition("/")
        if separator != "/" or not provider.strip() or not model_id.strip() or "/" in model_id:
            raise AgentError(f"agent {self._name} has invalid model pin {model!r}")
        body = {"model": {"providerID": provider, "id": model_id}}
        try:
            data = self._api("POST", "/api/session", body)
            return data["data"]["id"]
        except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise AgentError(f"could not create a session on agent {self._name}: {exc}") from exc

    def _api(self, method: str, path: str, body: dict | None = None, timeout: float = 30):
        request = urllib.request.Request(
            self._base + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return json.loads(raw) if raw else None

    def _open_events(self):
        request = urllib.request.Request(self._base + "/api/event")
        try:
            return urllib.request.urlopen(request, timeout=TURN_STALL_TIMEOUT)
        except (urllib.error.URLError, OSError) as exc:
            raise AgentError(
                f"could not open the event stream for agent {self._name}: {exc}"
            ) from exc

    def _stream_turn(self, events) -> Iterator[str]:
        """Yield text deltas for our session until the final step ends."""
        progress_deadline = time.monotonic() + TURN_STALL_TIMEOUT
        try:
            for raw_line in events:
                if time.monotonic() >= progress_deadline:
                    raise AgentError(f"agent {self._name} stalled mid-turn")
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    if not self.alive:
                        raise AgentError(f"agent {self._name} died mid-turn")
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                payload = event.get("data") or event.get("properties") or {}
                session = payload.get("sessionID") or (event.get("durable") or {}).get(
                    "aggregateID"
                )
                if session and session != self._session_id:
                    continue
                etype = event.get("type", "")
                if etype == "session.next.text.delta":
                    progress_deadline = time.monotonic() + TURN_STALL_TIMEOUT
                    yield payload.get("delta") or ""
                elif etype in ("session.error", "session.next.step.failed"):
                    raise AgentError(
                        f"agent {self._name} reported an error: {json.dumps(payload)[:200]}"
                    )
                elif etype == "session.next.step.ended":
                    progress_deadline = time.monotonic() + TURN_STALL_TIMEOUT
                    finish = payload.get("finish")
                    if finish == "stop":
                        return
                    if finish not in (None, "tool-calls"):
                        raise AgentError(f"agent {self._name} turn ended abnormally ({finish})")
        except TimeoutError as exc:
            raise AgentError(f"agent {self._name} stalled mid-turn") from exc
        except OSError as exc:
            if not self.alive:
                raise AgentError(f"agent {self._name} died mid-turn") from exc
            raise AgentError(f"lost the event stream for agent {self._name}: {exc}") from exc
        # SSE stream ended without a turn-end event: the server went away.
        raise AgentError(f"agent {self._name} closed the event stream mid-turn")
