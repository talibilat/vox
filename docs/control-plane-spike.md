# P0-02 Control-Plane Spike: opencode serve

Date: 2026-07-07.
opencode version tested: 1.17.14 on macOS (darwin arm64).
Spike script: `spikes/opencode_serve_spike.py` (throwaway code, stdlib-only Python).

## Verdict

**GO: the native `opencode serve` surface is fit to be Phase 1's primary transport.**
The full Conductor-owns-lifecycle loop works end to end: spawn the server process, create a session over HTTP, send a prompt, stream the response over SSE, detect completion from the event stream, send a follow-up into the same session, and tear the process down cleanly.
The reliability acceptance bar was met: 10 consecutive full-lifecycle runs (fresh server spawn, two-turn conversation, teardown) passed with zero hangs and zero truncated reads.
Turn latency was 1.4 to 6.3 seconds per turn against the free hosted model, including server spawn amortized into the first turn.
The tmux fallback path does not need promoting; it stays a fallback.

## Lifecycle model proven

- `opencode serve --port <N>` is spawned as a child process with no pre-started state.
- Readiness is detected by polling `GET /api/health` (the server binds within roughly 1 to 3 seconds).
- Teardown is a plain SIGTERM; the process exits promptly.
- Nothing was started by hand at any point, matching the plan's Conductor-owns-lifecycle decision.

## API surface actually used

The server self-describes at `GET /doc` (OpenAPI 3.1, 162 paths in this version).
The endpoints the spike drives:

| Purpose | Endpoint | Notes |
|---|---|---|
| Readiness | `GET /api/health` | Poll until 200 during startup. |
| Create session | `POST /api/session` | Body `{"model": {"providerID": "...", "id": "..."}}`. Returns `data.id` (`ses_...`). Model MUST be pinned (see quirks). |
| Send prompt | `POST /api/session/{id}/prompt` | Body `{"prompt": {"text": "..."}}`. Returns immediately with an admission record (`msg_...` id); the turn runs async. |
| Event stream | `GET /api/event` | SSE. Global stream, all sessions; filter by session id. Heartbeats keep it alive. |
| Read messages | `GET /api/session/{id}/message` | Durable record. Returns newest-first with cursor pagination. |
| Session info | `GET /api/session/{id}` | Also works from a freshly restarted server process (sessions are durable). |
| Interrupt | `POST /api/session/{id}/interrupt` | Present; not exercised in this spike but directly relevant to barge-in later. |

## Event shapes

Two SSE streams exist and carry the same event types with DIFFERENT envelopes:

- `GET /api/event` (v2): payload under a `data` object, plus a `durable` envelope (`aggregateID` = session id, `seq` for ordering).
- `GET /event` (legacy): payload under a `properties` object, and extra startup noise (`plugin.added` events, 45 of them on this install).

A turn emits this sequence on either stream:

```
session.next.prompt.admitted
session.next.prompted
session.next.step.started
session.next.reasoning.started / session.next.reasoning.delta (xN) / session.next.reasoning.ended
session.next.text.started / session.next.text.delta (xN) / session.next.text.ended
session.next.step.ended        <- carries finish: "stop" when the turn is done
```

Sample `session.next.step.ended` payload from `/api/event`:

```json
{
  "id": "evt_...",
  "type": "session.next.step.ended",
  "durable": {"aggregateID": "ses_...", "seq": 16, "version": 2},
  "data": {
    "timestamp": 1783383919276,
    "sessionID": "ses_...",
    "assistantMessageID": "msg_...",
    "finish": "stop",
    "cost": 0,
    "tokens": {"input": 19, "output": 6, "reasoning": 28, "cache": {"read": 3584, "write": 0}}
  }
}
```

Sample `session.next.text.delta` payload (legacy envelope shown; v2 nests the same fields under `data`):

```json
{
  "type": "session.next.text.delta",
  "properties": {"sessionID": "ses_...", "assistantMessageID": "msg_...", "textID": "text-0", "delta": "P"}
}
```

## Completion detection method

The reliable turn-finished signal is `session.next.step.ended` with `finish == "stop"`.
Multi-step turns (tool calls) emit multiple `step.ended` events; only `finish: "stop"` ends the turn, so treat other finish values (`tool-calls`) as continue.
After the event fires, the spike cross-checks the durable record (`GET .../message`) for a completed assistant message and uses that as the authoritative text.
`session.next.step.failed` and `session.error` are the failure signals.
This detection ran 10 consecutive full runs (20 turns) without a hang or a truncated read, plus the kill-restart run.

## Multi-turn persistence

Proven: turn two asked the model to repeat "the marker you replied with in your previous message" without restating it, and the model correctly echoed turn one's `TURN_ONE_OK`.
Session state is durable on disk, not process-local (see the kill test).

## Kill and restart behavior

The spike SIGKILLs the server one second after admitting a long-running prompt, then starts a fresh `opencode serve` process:

- The session is fully visible to the new process (`GET /api/session/{id}` returns it; all committed messages intact).
- The in-flight assistant response is silently dropped: no partial assistant message and no interrupted/error marker appears in the durable record.
- The orphaned user prompt persists, and the next prompt after restart makes the model answer BOTH the orphaned instruction and the new one in a single reply.
- Practical consequence for the Conductor: after restarting a server it must inspect the tail of the message list for an unanswered user message and decide whether to re-drive or clear it, rather than assuming a clean slate.

## Quirks and gotchas

1. **Model resolution must be explicit.** A session created without a `model` picked an unauthorized provider on this machine (`tencent-token-plan/hy3`) and every turn failed with HTTP 401 inside an assistant message with `finish: "error"`. Always pin `{"providerID", "id"}` from `GET /api/model` at session create.
2. **`POST /api/session/{id}/wait` is not usable.** It returns 503 `{"_tag": "ServiceUnavailableError", "message": "Session wait is not available yet"}` in this version, so completion detection must come from the event stream (or polling messages).
3. **`session.idle` never fires.** The event type exists in the OpenAPI schema, but across all captures on both streams it was never emitted; `session.next.step.ended` with `finish: "stop"` is the real signal.
4. **Two envelope formats.** Code must read the payload from `data` on `/api/event` and from `properties` on `/event`; a client written against one silently gets empty payloads on the other.
5. **Message list is newest-first.** `GET .../message` returns descending creation order; picking "the last element" gives the OLDEST message.
6. **Prompt errors land in-band.** Provider failures do not fail the prompt POST (it returns 200 with an admission record); the error appears later as an assistant message with `finish: "error"` and an `error` object. Watch the stream or the record, not the POST status.
7. **Global event stream.** `/api/event` carries every session on the server; a multi-agent Conductor must filter by `sessionID` (or `durable.aggregateID`). This is actually convenient for the Phase 2 fan-in model: one SSE connection can watch all 16 agents.

## Implications for #8 (adapter layer)

- Implemented in #8: `AgentAdapter.start()` owns the child process and creates the persistent session, `send(prompt)` yields markdown chunks until the turn completes, `stop()` shuts the process down, and `alive` reports whether the owned process is still running.
- One process can host many sessions and one SSE connection can watch them all, so the Phase 2 Conductor does not need a process per agent for opencode.
- Phase 1 restart recovery chooses a fresh session after a dead process, announces that session loss to the user, and retries the spoken request once; a future multi-agent Conductor can add orphaned-prompt inspection if it preserves sessions across restarts.
