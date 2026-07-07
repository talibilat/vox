#!/usr/bin/env python3
"""P0-02 control-plane spike: drive opencode end-to-end via `opencode serve`.

Throwaway code, stdlib only. Proves the Conductor-owns-lifecycle model:
spawn the server, create a session over HTTP, prompt it, stream events,
detect turn completion from the event stream, prompt again in the same
session, tear down cleanly.

Usage:
  python3 spikes/opencode_serve_spike.py            # one two-turn run
  python3 spikes/opencode_serve_spike.py --loop 10  # reliability loop
  python3 spikes/opencode_serve_spike.py --kill-restart
"""

import argparse
import json
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error

MODEL = {"providerID": "opencode", "id": "deepseek-v4-flash-free"}
TURN_TIMEOUT = 120.0
READY_TIMEOUT = 30.0


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def api(base, method, path, body=None, timeout=30):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else None


class Server:
    """Owns the opencode serve child process end to end."""

    def __init__(self):
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.proc = None

    def start(self):
        self.proc = subprocess.Popen(
            ["opencode", "serve", "--port", str(self.port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + READY_TIMEOUT
        while time.time() < deadline:
            try:
                api(self.base, "GET", "/api/health", timeout=2)
                return
            except (urllib.error.URLError, OSError):
                if self.proc.poll() is not None:
                    raise RuntimeError("opencode serve exited during startup")
                time.sleep(0.25)
        raise RuntimeError("opencode serve not ready within %ss" % READY_TIMEOUT)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def kill_hard(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGKILL)
            self.proc.wait()


class EventStream:
    """Background SSE reader for GET /api/event."""

    def __init__(self, base):
        self.base = base
        self.events = queue.Queue()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        req = urllib.request.Request(self.base + "/api/event")
        try:
            with urllib.request.urlopen(req, timeout=TURN_TIMEOUT + 30) as resp:
                for raw in resp:
                    if self._stop.is_set():
                        return
                    line = raw.decode("utf-8", "replace").strip()
                    if line.startswith("data: "):
                        try:
                            self.events.put(json.loads(line[6:]))
                        except json.JSONDecodeError:
                            pass
        except Exception:
            pass  # server going away mid-read is expected in the kill test

    def stop(self):
        self._stop.set()


def run_turn(base, events, session_id, text):
    """Send one prompt and stream until the final step ends.

    Completion signal: `session.next.step.ended` with finish == "stop".
    On `/api/event` the finish value is under `data`; on legacy `/event`, it
    is under `properties`. (`session.idle` exists in the schema but was never
    observed on either event stream in opencode 1.17.14, so it cannot be relied on.)
    """
    api(base, "POST", f"/api/session/{session_id}/prompt", {"prompt": {"text": text}})
    chunks = []
    deadline = time.time() + TURN_TIMEOUT
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError(f"turn did not finish within {TURN_TIMEOUT}s")
        try:
            ev = events.events.get(timeout=remaining)
        except queue.Empty:
            continue
        etype = ev.get("type", "")
        # /api/event wraps the payload in `data`; the legacy /event stream uses `properties`.
        props = ev.get("data") or ev.get("properties") or {}
        ev_session = props.get("sessionID") or (ev.get("durable") or {}).get("aggregateID") or ""
        if ev_session and ev_session != session_id:
            continue
        if etype == "session.next.text.delta":
            chunks.append(props.get("delta") or "")
        elif etype in ("session.error", "session.next.step.failed"):
            raise RuntimeError(f"turn failed: {json.dumps(ev)[:300]}")
        elif etype == "session.next.step.ended":
            finish = props.get("finish")
            if finish == "stop":
                break
            if finish not in (None, "tool-calls"):
                raise RuntimeError(f"turn ended with finish={finish!r}")
    streamed = "".join(chunks)
    # Cross-check the durable record, not just the stream.
    # /message returns newest-first, so take the most recently created match.
    msgs = api(base, "GET", f"/api/session/{session_id}/message")["data"]
    final = None
    newest = -1
    for m in msgs:
        if m.get("type") == "assistant" and m.get("finish") == "stop":
            created = (m.get("time") or {}).get("created", 0)
            text = "".join(c.get("text", "") for c in m.get("content", []) if c.get("type") == "text")
            if created > newest:
                newest = created
                final = text
    if final is None:
        raise RuntimeError("no completed assistant message found after idle")
    if streamed and streamed.strip() != final.strip():
        print(f"  note: streamed text differs from stored text ({streamed!r} vs {final!r})")
    return final


def two_turn_run(tag=""):
    srv = Server()
    srv.start()
    try:
        events = EventStream(srv.base)
        sid = api(srv.base, "POST", "/api/session", {"model": MODEL})["data"]["id"]
        t0 = time.time()
        r1 = run_turn(srv.base, events, sid, "Reply with exactly: TURN_ONE_OK")
        t1 = time.time()
        # Persistence proof: the second turn must recall first-turn content
        # that this prompt deliberately does not repeat.
        r2 = run_turn(srv.base, events, sid, "Reply with exactly the marker you replied with in your previous message.")
        t2 = time.time()
        print(f"{tag}turn1 ({t1 - t0:.1f}s): {r1!r}")
        print(f"{tag}turn2 ({t2 - t1:.1f}s): {r2!r}")
        ok = "TURN_ONE_OK" in r1 and "TURN_ONE_OK" in r2
        print(f"{tag}multi-turn persistence: {'OK' if ok else 'SUSPECT'}")
        events.stop()
        return ok
    finally:
        srv.stop()


def kill_restart_run():
    """Kill the server mid-turn, restart, and see what survives."""
    srv = Server()
    srv.start()
    events = EventStream(srv.base)
    sid = api(srv.base, "POST", "/api/session", {"model": MODEL})["data"]["id"]
    r1 = run_turn(srv.base, events, sid, "Reply with exactly: BEFORE_KILL_OK")
    print(f"pre-kill turn: {r1!r}")

    # Fire a prompt and SIGKILL the server while it is (probably) mid-turn.
    api(srv.base, "POST", f"/api/session/{sid}/prompt", {"prompt": {"text": "Count slowly from 1 to 30, one number per line."}})
    time.sleep(1.0)
    srv.kill_hard()
    events.stop()
    print("server SIGKILLed mid-turn")

    srv2 = Server()
    srv2.start()
    try:
        events2 = EventStream(srv2.base)
        # Does the old session still exist for the new process?
        try:
            info = api(srv2.base, "GET", f"/api/session/{sid}")
            print(f"session survived restart: id={info['data']['id']}")
        except urllib.error.HTTPError as e:
            print(f"session NOT visible after restart: HTTP {e.code}")
            return
        msgs = api(srv2.base, "GET", f"/api/session/{sid}/message")["data"]
        print(f"messages visible after restart: {len(msgs)}")
        interrupted = [m for m in msgs if m.get("type") == "assistant" and m.get("finish") not in ("stop", None)]
        print(f"interrupted assistant messages: {[m.get('finish') for m in interrupted]}")
        r = run_turn(srv2.base, events2, sid, "Reply with exactly: AFTER_RESTART_OK")
        print(f"post-restart turn: {r!r}")
        events2.stop()
    finally:
        srv2.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="run N consecutive two-turn runs")
    ap.add_argument("--kill-restart", action="store_true")
    args = ap.parse_args()

    if args.kill_restart:
        kill_restart_run()
        return
    if args.loop:
        passed = 0
        for i in range(1, args.loop + 1):
            print(f"--- run {i}/{args.loop}")
            try:
                if two_turn_run(tag=f"[{i}] "):
                    passed += 1
            except Exception as e:
                print(f"[{i}] FAILED: {e}")
        print(f"=== {passed}/{args.loop} runs passed")
        sys.exit(0 if passed == args.loop else 1)
    ok = two_turn_run()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
