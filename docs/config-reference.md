# Earshot Config Reference

Config lives at `~/.config/earshot/config.yaml` (override with `earshot --config PATH ...`).
Every key is optional; missing keys use the defaults below.
Unknown keys are hard errors with the offending key path, so typos never silently fall back to defaults.
Duplicate mapping keys (for example two agents with the same name) are rejected at load.

## wake_word

| Key | Default | Meaning |
|---|---|---|
| `phrase` | `hey earshot` | Informational label for the wake phrase (detection comes from the model). |
| `model_path` | `null` | Path to a trained openWakeWord `.onnx` model. The voice loop is disabled while this is null. |
| `sensitivity` | `0.9` | Detection threshold, 0..1. Tuned value; see docs/tuning-protocol.md. |
| `patience` | `3` | Consecutive 80ms windows above the threshold before firing; suppresses one-frame confusions. |

## stt

| Key | Default | Meaning |
|---|---|---|
| `backend` | `local` | `local` (faster-whisper, offline) or `api` (OpenAI-compatible). |
| `local.model` | `base.en` | faster-whisper model size. `tiny.en` halves latency at some accuracy cost. |
| `local.device` | `cpu` | Inference device passed to faster-whisper. |
| `local.compute_type` | `int8` | faster-whisper quantization. |
| `api.base_url` | `https://api.openai.com/v1` | Any OpenAI-compatible `/audio/transcriptions` server. |
| `api.api_key_env` | `OPENAI_API_KEY` | NAME of the environment variable holding the key; never a literal key. |
| `api.model` | `whisper-1` | Hosted model name. |
| `api.fallback_to_local` | `false` | On API failure, transparently transcribe with the local backend instead. |

## tts

| Key | Default | Meaning |
|---|---|---|
| `backend` | `local` | `local` (Piper) or `api` (OpenAI-compatible `/audio/speech`). |
| `local.engine` | `piper` | Only `piper` is implemented; it won the latency spike by 6-20x. |
| `local.voice` | `en_US-lessac-medium` | Piper voice, auto-downloaded to `~/.local/share/earshot/voices` on first use. |
| `local.speed` | `1.0` | Speaking speed multiplier. |
| `api.base_url` | `https://api.openai.com/v1` | Any OpenAI-compatible `/audio/speech` server. |
| `api.api_key_env` | `OPENAI_API_KEY` | NAME of the environment variable holding the key; never a literal key. |
| `api.model` | `tts-1` | Hosted synthesis model name. |
| `api.voice` | `alloy` | Hosted synthesis voice name. Audio streams as 24kHz PCM. |
| `api.fallback_to_local` | `false` | On API failure, synthesize locally (resampled to match the open stream). |

## code_blocks

`summarize` (default) speaks "A 12 lines python code block."; `skip` omits code entirely; `read` reads the code text aloud.

## agents

A map of spoken name to agent definition; the map order matters only in that the first agent starts as the active one.

| Key | Default | Meaning |
|---|---|---|
| `harness` | `opencode` | `opencode`, `claude-code`, or `codex`. Selects the native adapter. |
| `command` | `null` | Explicit launch command override (the harness default otherwise). |
| `workdir` | `~` | Working directory the agent runs in. |
| `model` | `null` | `provider/model-id` (or the harness's own naming). `null` uses the harness default; the opencode adapter always pins one internally. |
| `restart_on_death` | `true` | The fleet supervisor restarts this agent if its process dies. |
| `tmux_pane` | `null` | Setting this switches the agent to the generic tmux fallback transport; the value names the tmux session Earshot creates and owns. |

Naming guidance: use phonetically distinct, multi-syllable names; validation warns about single-syllable names and sound-alike pairs because they transcribe unreliably.

## barge_in

| Key | Default | Meaning |
|---|---|---|
| `vad_threshold` | `0.6` | Silero VAD speech probability for both barge-in onset and end-of-speech detection. Tuned value; see docs/tuning-protocol.md. |
| `interrupt_hotkey` | `null` | Informational label. The actual escape hatch is `earshot interrupt` (SIGUSR1 to the daemon); bind that command to a system hotkey. |

## daemon

| Key | Default | Meaning |
|---|---|---|
| `log_file` | `~/.local/state/earshot/earshot.log` | Daemon log (includes per-interrupt latency lines; grep "barge-in"). |
| `pid_file` | `~/.local/state/earshot/earshot.pid` | Daemon PID file. |

## CLI

```sh
earshot start [--foreground]   # spawn the daemon (and every configured agent)
earshot status                 # is it running
earshot interrupt              # push-to-interrupt escape hatch
earshot stop                   # clean shutdown, agents included
earshot --config PATH ...      # any command against a non-default config
```
