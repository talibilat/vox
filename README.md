# Earshot

Voice-to-voice control for terminal coding agents.
You speak to your agents; they speak back.
No keyboard, no reading walls of markdown, and it scales from one agent to a named fleet of sixteen you conduct by voice.

> Release status: source-ready; the public demo recording and PyPI publication are release-gate tasks tracked in `docs/tickets/P3-02.md`.

## What it does

- **Voice-to-voice loop**: say the wake word ("Hey Earshot"), speak an instruction, the agent executes, and the response is spoken back as natural speech.
- **Barge-in**: talk over the agent and playback stops (measured stop path: median 124ms, worst 140ms on an Apple M4 Pro), and what you said becomes the next command; no wake word needed.
- **Speakable output**: markdown is converted before it is voiced; you will never hear "hash hash bold star star". Code blocks are summarized, skipped, or read, per config.
- **Agent-agnostic**: works with opencode, Claude Code, and codex through each harness's native headless surface, plus a generic tmux fallback for whatever comes next.
- **The Conductor**: configure many named agents; "marvin, run the tests" routes to marvin, follow-ups stick to whoever you addressed last, "agent status" answers with a spoken roll-call, and background agents stay silent until you ask "olivia, what's your response".
- **Local-first**: the default stack is fully offline (openWakeWord, faster-whisper, Silero VAD, Piper). Hosted OpenAI-compatible STT/TTS are opt-in per backend, with optional automatic fallback to local.

## Install

From source today:

```sh
git clone https://github.com/talibilat/earshot && cd earshot
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

After PyPI publication, install from PyPI with the package name `earshot-cli`.

Runtime requirements: Python 3.10+, a microphone and speaker (PortAudio), and at least one agent CLI on your PATH (`opencode`, `claude`, or `codex`).
On macOS, `brew install portaudio` if sounddevice cannot find it.

## Quickstart (one agent)

1. Get a trained openWakeWord model for your wake phrase. The repo ships a development-grade "Hey Earshot" model at `spikes/models/hey_earshot.onnx` (train a better one with `spikes/train_wakeword.py`).
2. Create `~/.config/earshot/config.yaml` (running `earshot start` once generates a commented default):

```yaml
wake_word:
  model_path: /path/to/hey_earshot.onnx
agents:
  main:
    harness: opencode        # or claude-code | codex
    workdir: ~/projects/myrepo
```

3. `earshot start`, put on a headset, and say: "Hey Earshot" ... "list the files in this directory and tell me what you see".
4. Talk over the answer whenever you want to redirect it. `earshot interrupt` is the push-to-interrupt escape hatch (bind it to a system hotkey); `earshot stop` shuts everything down, including every agent process Earshot spawned.

## Everyday commands

| Goal | Say or run |
| --- | --- |
| Start Earshot | `earshot start` |
| Give the active agent an instruction | “Hey Earshot, run the tests and tell me what failed.” |
| Redirect a spoken answer | Talk over the answer, or run `earshot interrupt` from a shell. |
| Hear the fleet’s current state | “agent status” |
| Stop Earshot and its agent processes | `earshot stop` |

## The fleet (multi-agent)

```yaml
agents:
  marvin:
    harness: opencode
    workdir: ~/projects/backend
  olivia:
    harness: claude-code
    workdir: ~/projects/frontend
  sebastian:
    harness: codex
    workdir: ~/projects/infra
```

- "marvin, run the tests" routes to marvin and makes him the active agent; bare follow-ups keep going to him.
- Everyone else works in silence; nothing is ever spoken unprompted.
- "olivia, what's your response" reads back exactly olivia's latest output.
- "agent status" answers like "marvin and olivia have finished; sebastian is still working."
- A garbled name gets "Did you mean marvin?" instead of a silent misroute; answer "yes" to confirm.
- Earshot spawns and supervises every agent itself (staggered startup, per-agent `restart_on_death` policy); nothing is started by hand.
- Pick phonetically distinct, multi-syllable names; multi-agent config validation warns about risky ones.

## Local vs API mode

Local is the default and works fully offline.
Any backend can be switched independently:

```yaml
stt:
  backend: api               # hosted transcription
  api:
    api_key_env: OPENAI_API_KEY
    fallback_to_local: true  # degrade to faster-whisper on API failure
tts:
  backend: local             # keep the local Piper voice
```

API keys are only ever read from the environment variable you name; never put a key in YAML.

## Tuned defaults

The shipped thresholds come from a measured sweep (see `docs/tuning-protocol.md` for the full tables and the reproduction command):

- Barge-in VAD threshold 0.6: zero false interrupts from music and keyboard noise in the sweep, with no onset-latency cost over the old default.
- Wake word at sensitivity 0.9 / patience 3: zero false fires across the adversarial scenario set while detecting every positive.
- Interrupt stop path re-validated post-tuning: median 124ms, worst 140ms.
- Voice addressing: 16/16 spoken commands routed correctly through real STT in the tuning run.
- Assumption to know about: barge-in listens while the agent talks, so use a headset (or modest speaker volume); a VAD cannot tell your voice from a nearby conversation.

## Documentation

- [Config reference](docs/config-reference.md) covers every knob in the schema.
- [Tuning protocol and numbers](docs/tuning-protocol.md)
- [Control-plane verdicts per harness](docs/control-plane-verdicts.md)
- [Latency and wake-word spike numbers](docs/latency-spike.md)
- [opencode serve transport notes](docs/control-plane-spike.md)
- [Dependency license audit](docs/licenses.md)
- [Demo recording script](docs/demo-script.md)
- Per-ticket engineering notes live under `docs/tickets/`.

## Relationship to Claude Code's voice mode

Claude Code's native voice input is push-to-talk speech-to-text only; there is no spoken response, wake word, or barge-in, and open feature requests (anthropics/claude-code #34305, #36745) ask for exactly the loop Earshot provides.
Earshot treats that as validation: it is the wrapper that gives any terminal agent a full voice conversation, not a competitor to any single harness.

## Parking lot (deliberate non-features, for now)

- Per-agent TTS voices (instant recognizability in a fleet).
- A notification chime when a background agent finishes.
- "Read the last thing again."

Earshot is a personal tool first: no SLAs, no multi-user, no production hardening.

## Development

```sh
git clone https://github.com/talibilat/earshot && cd earshot
uv venv && uv pip install -e ".[dev]"
pytest -q        # 255 tests; heavier voice tests skip without the audio deps
ruff check earshot tests
```

Development notes: `earshot start --foreground` runs in the terminal; the voice loop needs `wake_word.model_path` set (the committed model at `spikes/models/hey_earshot.onnx` is development-grade).
The name situation: the product is Earshot, the pip package is `earshot-cli` (`earshot` was taken on PyPI), and the public repository URL is `github.com/talibilat/earshot`.

## License

MIT (see LICENSE). Dependency licensing was audited up front; see docs/licenses.md.
