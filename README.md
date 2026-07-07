# Vox

Earshot is a voice-to-voice control project for terminal coding agents.

## Earshot Scaffold

The project currently provides the installable Python package `earshot-cli`, which exposes the `earshot` console command.
The scaffold covers configuration loading, daemon lifecycle, and the first audio-input pipeline: wake word detection, end-of-speech detection, and local faster-whisper STT.
Speech output and agent adapters land in later phase issues.

Install for development with:

```sh
uv pip install -e ".[dev]"
```

Run the daemon lifecycle commands with:

```sh
earshot start
earshot status
earshot stop
```

Use `earshot start --foreground` for a foreground development run.
Use `earshot --config PATH ...` to point at a non-default config file.
Audio input starts only when `wake_word.model_path` points at a trained openWakeWord `.onnx` model; without it, the daemon logs that the input pipeline is disabled.
The committed feasibility model is `spikes/models/hey_earshot.onnx`; it is useful for development but not production-ready.

On first run without `--config`, Earshot creates `~/.config/earshot/config.yaml` from the same template committed as `config.example.yaml`.
Every config key is optional, unknown keys are rejected with key-path errors, and the schema already reserves fields for wake word, STT, TTS, code-block handling, agent harnesses, barge-in, and daemon paths.
Local STT is implemented with faster-whisper; the API STT backend is reserved for issue #10 and raises until that lands.

## Docs

- [Issue dependency graph](docs/dependency-graph.md)
- [P1-01 repo scaffold notes](docs/tickets/P1-01.md)
- [P1-02 audio input notes](docs/tickets/P1-02.md)
- [Example Earshot config](config.example.yaml)
- [P0-02 control-plane spike](docs/control-plane-spike.md)
- [P0-02 process notes](docs/tickets/P0-02.md)

## Project Documentation

- [Issue dependency graph](docs/dependency-graph.md) maps the implementation order across project phases.
- [P1-01 repo scaffold notes](docs/tickets/P1-01.md) record the package, daemon, config schema, and validation work.
- [P1-02 audio input notes](docs/tickets/P1-02.md) record the wake-word, endpointing, microphone, and local STT pipeline work.
- [Example Earshot config](config.example.yaml) shows the complete YAML schema and defaults.
- [P0-01 license gate](docs/licenses.md) records dependency license verdicts and the Earshot license recommendation.
- [P0-01 VoiceMode notes](docs/voicemode-notes.md) record the VoiceMode design review and local Claude Code MCP smoke test.
- [P0-02 control-plane spike](docs/control-plane-spike.md) records the `opencode serve` transport verdict, event shapes, and adapter implications.
- [P0-03 voice-stack spike](docs/latency-spike.md) records STT, TTS, VAD, and wake-word latency results plus the committed feasibility model.

## Ticket Notes

- [P0-01 process notes](docs/tickets/P0-01.md)
- [P0-02 process notes](docs/tickets/P0-02.md)
- [P0-03 process notes](docs/tickets/P0-03.md)
- [P1-02 process notes](docs/tickets/P1-02.md)
