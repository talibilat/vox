"""A minimal stand-in for `codex app-server`: JSON-RPC over stdio.

Implements initialize, thread/start, and turn/start with the same event
grammar the real server was observed to use (item/agentMessage/delta
notifications, then turn/completed). Prompt-driven failure modes:

- "please error"  -> turn/completed with turn.error set
- "please stall"  -> no events after the turn/start response
- "please die"    -> the process exits 1 mid-turn
"""

import json
import sys

THREAD_ID = "thread-fake-0001"


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def notify(method, params):
    emit({"jsonrpc": "2.0", "method": method, "params": params})


def main():
    turns = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        method = request.get("method")
        request_id = request.get("id")
        if method == "initialize":
            emit({"jsonrpc": "2.0", "id": request_id, "result": {"userAgent": "fake-codex"}})
        elif method == "thread/start":
            emit(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "thread": {"id": THREAD_ID, "model": request["params"].get("model")}
                    },
                }
            )
        elif method == "turn/start":
            turns += 1
            prompt = request["params"]["input"][0]["text"]
            emit({"jsonrpc": "2.0", "id": request_id, "result": {"turn": {"id": f"turn-{turns}"}}})
            if "please stall" in prompt:
                continue
            if "please die" in prompt:
                notify(
                    "item/agentMessage/delta",
                    {"threadId": THREAD_ID, "delta": "dying "},
                )
                sys.exit(1)
            if "please error" in prompt:
                notify(
                    "turn/completed",
                    {
                        "threadId": THREAD_ID,
                        "turn": {"status": "failed", "error": {"message": "simulated failure"}},
                    },
                )
                continue
            notify("item/agentMessage/delta", {"threadId": THREAD_ID, "delta": f"Turn {turns}: "})
            notify(
                "item/agentMessage/delta",
                {"threadId": THREAD_ID, "delta": f"you said {prompt!r}. "},
            )
            notify(
                "turn/completed",
                {"threadId": THREAD_ID, "turn": {"status": "completed", "error": None}},
            )
        else:
            emit({"jsonrpc": "2.0", "id": request_id, "result": {}})


if __name__ == "__main__":
    main()
