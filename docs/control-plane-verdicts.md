# Per-Harness Control-Plane Verdicts

Date: 2026-07-07.
Versions validated: opencode 1.17.14, Claude Code 2.1.197, codex-cli 0.142.4, all on macOS arm64.
Method: each surface was prototyped in isolation to capture real event shapes, implemented against the `AgentAdapter` contract, exercised by the shared cross-harness behavioral test matrix (spawn/own/stop, streaming, multi-turn persistence, in-band errors, mid-turn death, launch failure) against scripted fake harnesses, and then proven with a real-binary two-turn conversation. The full voice chain (fixture wake audio -> STT -> adapter -> spoken response) runs against all three via `tests/test_voice_loop.py`, switching only the config `harness:` field.

## Verdict summary

| Harness | Surface | Verdict | tmux fallback needed? |
|---|---|---|---|
| opencode | `opencode serve` HTTP + SSE | **Solid.** Primary transport confirmed (P0-02 spike + P1-05 adapter + this matrix). | No |
| Claude Code | `claude --print` stream-json + `--resume` | **Solid, with per-turn process cost.** | No |
| codex | `codex app-server` JSON-RPC over stdio | **Solid.** | No |

**No harness needs the tmux fallback (#14) at current versions.** All three native surfaces passed the same behavioral contract and real two-turn conversations. #14 stays dormant unless a future version regresses.

## opencode (`opencode serve`)

- Long-lived HTTP server owned by the adapter; sessions durable across restarts.
- Streaming via SSE `session.next.text.delta`; completion via `session.next.step.ended` with `finish: "stop"`.
- Quirks (full list in docs/control-plane-spike.md): model must be pinned per session or an unauthorized default provider can be resolved, so the adapter always pins one (default lives in the adapter, not the shared config, to keep it from leaking into other harnesses); `/wait` endpoint returns 503; `session.idle` never fires.
- Conductor notes (#11): one server can host many sessions; one SSE connection can watch them all.

## Claude Code (`claude --print` stream-json)

- ONE PROCESS PER TURN: the adapter spawns `claude --print --input-format stream-json --output-format stream-json --verbose` per turn and persists the conversation with `--resume <session_id>` captured from the first turn's events. `alive` means "adapter started", not "process running".
- Event grammar: `system/init` (carries session_id), one `assistant` event per content block (thinking blocks skipped), terminal `result` event with `is_error` as the completion signal; hook events interleave as noise and must be ignored.
- Text granularity is per content block, not token deltas (fine for sentence-chunked TTS; `--include-partial-messages` exists if finer streaming is ever needed).
- Quirks the Conductor must know: per-turn process startup is heavy (~5s observed with user hooks firing on every spawn), which makes Claude Code the slowest harness to first token by far; permission prompts cannot be answered over voice, so the adapter passes `--permission-mode acceptEdits` (users widen policy via the `command` override); a `rate_limit_event` warning surfaces in-stream near quota limits.
- Cost note: every turn re-creates prompt cache (observed 18k cache-write tokens on a trivial turn).

## codex (`codex app-server`)

- app-server chosen over mcp-server after prototyping: long-lived process, persistent threads, TRUE streaming deltas, and a `turn/interrupt` method that maps directly onto future barge-in needs; mcp-server would wrap turns in MCP tool calls with no incremental output.
- Protocol: JSON-RPC over stdio. `initialize` -> `thread/start {cwd, model?}` -> `turn/start {threadId, input: [{type: "text", text}]}`; streaming via `item/agentMessage/delta` notifications; completion via `turn/completed` with `turn.status`/`turn.error`.
- Quirks: the protocol is marked experimental by codex and is schema-generated (`codex app-server generate-json-schema`), so version pinning matters; benign notifications (`remoteControl/status/changed`, rate-limit updates) interleave and must be filtered; approval-requiring tool calls surface as server requests the adapter does not yet answer (same MVP posture as the other harnesses: trivial voice turns do not trigger them).

## Cross-harness bug this validation caught

`AgentConfig.model` had defaulted to the opencode free-tier model, which leaked an opencode-specific value into every harness; codex rejected it with an invalid_request_error on the first real turn.
The default now lives inside the opencode adapter, `model: null` means "the harness's own default", and the cross-harness matrix pins the regression.

## Manual voice matrix (needs a human with a microphone)

For each harness, set `agents.main.harness` in the config and run the P1-05 checklist (wake -> instruct -> spoken response -> in-session follow-up). The automated matrix covers everything except the physical microphone and the real agent brains; the two-turn real-binary tests cover those separately.
