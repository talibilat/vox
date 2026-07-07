"""A minimal stand-in for `claude --print` in stream-json mode.

One process per turn, like the real thing. Session continuity across turns
is proven the same way the real CLI does it: state lives outside the
process (a file per session id under FAKE_CLAUDE_STATE), and the adapter
must pass --resume <session_id> for turn two to find turn one's state.

Prompt-driven failure modes:
- "please error"  -> result event with is_error true
- "please stall"  -> no further output (the adapter's stall guard fires)
- "please die"    -> exits 1 mid-stream without a result event
"""

import json
import os
import sys
import time
import uuid
from pathlib import Path

STATE_DIR = Path(os.environ.get("FAKE_CLAUDE_STATE", "/tmp/fake_claude_state"))


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    args = sys.argv[1:]
    assert "--print" in args, "adapter must run print mode"
    assert "stream-json" in args, "adapter must use stream-json"
    session_id = None
    if "--resume" in args:
        session_id = args[args.index("--resume") + 1]
    message = json.loads(sys.stdin.readline())
    prompt = message["message"]["content"][0]["text"]

    if session_id is None:
        session_id = str(uuid.uuid4())
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_file = STATE_DIR / f"{session_id}.turns"
    turn = int(state_file.read_text()) + 1 if state_file.exists() else 1
    state_file.write_text(str(turn))

    emit({"type": "system", "subtype": "init", "session_id": session_id})
    emit({"type": "system", "subtype": "hook_started", "session_id": session_id})  # noise

    if "please stall" in prompt:
        time.sleep(3600)
    if "please die" in prompt:
        emit(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "dying "}]},
                "session_id": session_id,
            }
        )
        sys.exit(1)
    if "please error" in prompt:
        emit(
            {
                "type": "result",
                "subtype": "error",
                "is_error": True,
                "result": "simulated provider failure",
                "session_id": session_id,
            }
        )
        return

    emit(
        {
            "type": "assistant",
            "message": {"content": [{"type": "thinking", "thinking": "hmm"}]},
            "session_id": session_id,
        }
    )
    emit(
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": f"Turn {turn}: "}]},
            "session_id": session_id,
        }
    )
    emit(
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": f"you said {prompt!r}. "}]},
            "session_id": session_id,
        }
    )
    emit(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": f"Turn {turn}: you said {prompt!r}. ",
            "session_id": session_id,
        }
    )


if __name__ == "__main__":
    main()
