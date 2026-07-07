"""A minimal stand-in for `opencode serve`, for adapter tests.

Implements just the surface the adapter uses: /api/health, POST
/api/session, POST /api/session/{id}/prompt, and the /api/event SSE stream
with the v2 envelope (payload under `data`). The response depends on the
prompt text so tests can drive every failure path:

- contains "please error"  -> a session.error event
- contains "please stall"  -> no events at all (client should time out)
- contains "please die"    -> the whole server process exits mid-turn
- anything else            -> two text deltas + step.ended finish=stop,
                              echoing the prompt and the turn number so
                              multi-turn session persistence is observable.

Run: python fake_opencode_serve.py serve --port N
"""

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SESSION_ID = "ses_fake0000000000000000000000"
STATE = {"turns": 0, "prompt": None, "event": threading.Event()}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/health":
            self._json({})
        elif self.path == "/api/event":
            self._serve_events()
        else:
            self._json({"error": "not found"}, status=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/session":
            self._json({"data": {"id": SESSION_ID, "model": body.get("model")}})
        elif self.path == f"/api/session/{SESSION_ID}/prompt":
            STATE["turns"] += 1
            STATE["prompt"] = body["prompt"]["text"]
            STATE["event"].set()
            self._json({"data": {"id": "msg_fake", "sessionID": SESSION_ID}})
        else:
            self._json({"error": "unknown session"}, status=404)

    def _serve_events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def emit(event_type, payload):
            data = {"type": event_type, "data": {"sessionID": SESSION_ID, **payload}}
            self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()

        while True:
            if not STATE["event"].wait(timeout=30):
                return
            STATE["event"].clear()
            prompt = STATE["prompt"] or ""
            if "please stall" in prompt:
                continue  # never respond; the client's stall timeout fires
            if "please die" in prompt:
                emit("session.next.text.delta", {"delta": "dying "})
                self.wfile.flush()
                time.sleep(0.1)
                sys.stderr.close()
                import os

                os._exit(1)
            if "please error" in prompt:
                emit("session.error", {"message": "provider exploded"})
                continue
            turn = STATE["turns"]
            emit("session.next.text.delta", {"delta": f"Turn {turn}: "})
            emit("session.next.text.delta", {"delta": f"you said {prompt!r}. "})
            emit("session.next.step.ended", {"finish": "stop"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["serve"])
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
