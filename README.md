# Vox

Earshot is a voice-to-voice control project for terminal coding agents.

## Earshot Scaffold

The project currently provides the installable Python package `earshot-cli`, which exposes the `earshot` console command.
The scaffold covers configuration loading, daemon lifecycle, the first audio-input pipeline, the first speech-output pipeline, and the harness-backed voice loop: wake word detection, end-of-speech detection, local faster-whisper or OpenAI-compatible API STT, markdown-to-speakable text, local Piper or OpenAI-compatible API TTS, streamed agent responses, voice addressing, per-agent output watchers, spoken fleet status, and barge-in interruption while the agent is speaking.
Phase 2 adds the Conductor core: the daemon starts every configured agent as a supervised fleet, staggers startup, tracks per-agent lifecycle status, restarts dead agents according to each agent's `restart_on_death` policy, routes spoken turns by addressed agent name, and buffers non-active agents' responses until they are read aloud on request.
The implemented agent harnesses are `opencode`, `claude-code`, and `codex`.

Install for development with:

```sh
uv pip install -e ".[dev]"
```

Run the daemon lifecycle commands with:

```sh
earshot start
earshot status
earshot interrupt
earshot stop
```

Use `earshot start --foreground` for a foreground development run.
Use `earshot --config PATH ...` to point at a non-default config file.
The voice loop starts only when `wake_word.model_path` points at a trained openWakeWord `.onnx` model; without it, the daemon logs that the voice loop is disabled.
The committed feasibility model is `spikes/models/hey_earshot.onnx`; it is useful for development but not production-ready.
While the daemon is responding, speaking over playback interrupts the agent, stops synthesis and speaker output, and records the interruption as the next command without requiring the wake word again.
Use `earshot interrupt` as the push-to-interrupt escape hatch; bind that command in your OS or launcher if you want a one-keystroke hotkey.

On first run without `--config`, Earshot creates `~/.config/earshot/config.yaml` from the same template committed as `config.example.yaml`.
Every config key is optional, unknown keys and duplicate YAML keys are rejected with key-path errors, and the schema covers wake word, STT, TTS, code-block handling, agent harnesses, restart policy, barge-in, and daemon paths.
Configure named agents under `agents` with `agents.<name>.harness` set to `opencode`, `claude-code`, or `codex`, plus an `agents.<name>.workdir`; Earshot owns the harness processes, starts the full fleet when the voice loop is enabled, and routes addressed utterances like `<name>, run the tests` to that agent's watcher.
Set `agents.<name>.model` only when you want to override a harness default; `model: null` uses the adapter default, and opencode validates explicit overrides in `provider/model-id` form.
Agent names are spoken names, so validation warns when names are short or too similar for reliable speech recognition; fuzzy matching handles common vowel-level mishearings, ambiguous matches ask for confirmation aloud, and unaddressed follow-ups go to the active agent.
Only the active agent's turn streams to the speaker as it arrives; other agents work silently, move to `finished` when an unread response is buffered, and are read aloud with requests like `<name>, what's your response`.
Fleet-status phrases such as `agent status` speak a grouped roll-call of finished, working, idle, and not-running agents.
If an agent fails mid-turn, Earshot buffers a speakable failure response and speaks feedback only when that agent is active; if the process died, the Conductor supervisor applies the agent's `restart_on_death` policy.
The Conductor supervisor owns restarts for all daemon-managed agents, active included; the older active-agent restart exemption applies only to the legacy direct `ConversationLoop` fallback.
Local STT is implemented with faster-whisper, and local TTS is implemented with Piper.
API STT and TTS use OpenAI-compatible `/audio/transcriptions` and `/audio/speech` endpoints, read the API key from the environment variable named by `stt.api.api_key_env` or `tts.api.api_key_env`, and can fall back to the local backend when `fallback_to_local` is true; Kokoro remains a reserved local TTS engine and raises today.
Speech output converts streamed Markdown to speakable text sentence-by-sentence; `code_blocks` controls whether fenced code blocks are summarized, skipped, or read aloud.

## Docs

- [Issue dependency graph](docs/dependency-graph.md)
- [P1-01 repo scaffold notes](docs/tickets/P1-01.md)
- [P1-02 audio input notes](docs/tickets/P1-02.md)
- [P1-03 speech output notes](docs/tickets/P1-03.md)
- [P1-04 barge-in notes](docs/tickets/P1-04.md)
- [P1-05 agent adapter notes](docs/tickets/P1-05.md)
- [P1-07 API backend notes](docs/tickets/P1-07.md)
- [P1-06 harness adapter notes](docs/tickets/P1-06.md)
- [P2-01 Conductor core notes](docs/tickets/P2-01.md)
- [P2-02 voice addressing notes](docs/tickets/P2-02.md)
- [P2-03 output watcher notes](docs/tickets/P2-03.md)
- [Example Earshot config](config.example.yaml)
- [P0-02 control-plane spike](docs/control-plane-spike.md)
- [Per-harness control-plane verdicts](docs/control-plane-verdicts.md)
- [P0-02 process notes](docs/tickets/P0-02.md)

## Project Documentation

- [Issue dependency graph](docs/dependency-graph.md) maps the implementation order across project phases.
- [P1-01 repo scaffold notes](docs/tickets/P1-01.md) record the package, daemon, config schema, and validation work.
- [P1-02 audio input notes](docs/tickets/P1-02.md) record the wake-word, endpointing, microphone, and local STT pipeline work.
- [P1-03 speech output notes](docs/tickets/P1-03.md) record the markdown-to-speech, local Piper TTS, and interruptible playback work.
- [P1-04 barge-in notes](docs/tickets/P1-04.md) record the VAD interruption loop, push-to-interrupt command, and target-hardware latency validation work.
- [P1-05 agent adapter notes](docs/tickets/P1-05.md) record the opencode adapter, original single-agent voice loop, and daemon agent-process ownership work.
- [P1-07 API backend notes](docs/tickets/P1-07.md) record the OpenAI-compatible STT/TTS backends, API failure handling, and optional local fallback work.
- [P1-06 harness adapter notes](docs/tickets/P1-06.md) record the Claude Code and codex adapters plus the three-harness validation matrix.
- [P2-01 Conductor core notes](docs/tickets/P2-01.md) record the multi-agent fleet lifecycle, per-agent restart policy, duplicate-name rejection, phonetic naming warnings, and live 16-agent validation.
- [P2-02 voice addressing notes](docs/tickets/P2-02.md) record fuzzy leading-name routing, active-agent switching, clarification prompts, fleet phrase protection, and the router handoff from the voice loop.
- [P2-03 output watcher notes](docs/tickets/P2-03.md) record per-agent watcher threads, silent background buffering, readback requests, natural fleet status, and watcher-owned failure isolation.
- [Example Earshot config](config.example.yaml) shows the complete YAML schema and defaults.
- [P0-01 license gate](docs/licenses.md) records dependency license verdicts and the Earshot license recommendation.
- [P0-01 VoiceMode notes](docs/voicemode-notes.md) record the VoiceMode design review and local Claude Code MCP smoke test.
- [P0-02 control-plane spike](docs/control-plane-spike.md) records the `opencode serve` transport verdict, event shapes, and adapter implications.
- [Per-harness control-plane verdicts](docs/control-plane-verdicts.md) record the current native transport verdicts for opencode, Claude Code, and codex.
- [P0-03 voice-stack spike](docs/latency-spike.md) records STT, TTS, VAD, and wake-word latency results plus the committed feasibility model.

## Ticket Notes

- [P0-01 process notes](docs/tickets/P0-01.md)
- [P0-02 process notes](docs/tickets/P0-02.md)
- [P0-03 process notes](docs/tickets/P0-03.md)
- [P1-02 process notes](docs/tickets/P1-02.md)
- [P1-03 process notes](docs/tickets/P1-03.md)
- [P1-04 process notes](docs/tickets/P1-04.md)
- [P1-05 process notes](docs/tickets/P1-05.md)
- [P1-07 process notes](docs/tickets/P1-07.md)
- [P1-06 process notes](docs/tickets/P1-06.md)
- [P2-01 process notes](docs/tickets/P2-01.md)
- [P2-02 process notes](docs/tickets/P2-02.md)
- [P2-03 process notes](docs/tickets/P2-03.md)
