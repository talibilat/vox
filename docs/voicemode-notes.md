# P0-01 VoiceMode Notes

Date: 2026-07-06.

This file records the VoiceMode design review and local Claude Code smoke test for issue #1.

## Local Run Evidence

Claude Code was available at `/opt/homebrew/bin/claude`.
The installed Claude Code version was `2.1.197`.
`uvx` was available at `/Library/Frameworks/Python.framework/Versions/3.12/bin/uvx`.
The installed `uvx` version was `0.9.30`.

VoiceMode was installed and invoked with `uvx --from voice-mode voicemode --help`.
That command installed the package dependencies and printed the VoiceMode CLI help.

A temporary MCP config was written outside the repo at `/private/tmp/earshot-p0-01-licenses/voicemode-mcp.json`.
The config exposed only the `converse` tool and set `VOICEMODE_SKIP_TTS=true`.
The first Claude Code MCP smoke test after reset failed because `ffmpeg` was missing.
`ffmpeg` was then installed with `brew install ffmpeg`, matching VoiceMode's macOS requirement.
The final Claude Code smoke test succeeded with this command shape:

```sh
claude -p --mcp-config /private/tmp/earshot-p0-01-licenses/voicemode-mcp.json --strict-mcp-config --allowedTools mcp__voicemode__converse --permission-mode auto --max-budget-usd 2.00 "<prompt asking Claude Code to call VoiceMode converse>"
```

Claude Code reported that the VoiceMode `converse` tool call succeeded with `wait_for_response=false` and `skip_tts=true`.
That verifies the local Claude Code to VoiceMode MCP path without requiring live microphone input.

## Patterns Worth Borrowing

VoiceMode treats local STT and TTS services as OpenAI-compatible endpoints.
Its configuration uses ordered base URL lists such as `VOICEMODE_STT_BASE_URLS` and `VOICEMODE_TTS_BASE_URLS`.
That is the strongest pattern for Earshot because it lets the same caller switch among local Whisper, local Kokoro, and cloud OpenAI-compatible APIs without changing core loop code.

VoiceMode separates provider discovery from the actual voice turn.
`voice_mode/provider_discovery.py` classifies endpoints, seeds default models and voices, and records endpoint metadata.
`voice_mode/simple_failover.py` then walks configured endpoint lists and returns detailed attempted-endpoint errors.
Earshot should borrow this split: configuration and discovery should not be tangled into audio capture or agent transport code.

VoiceMode's local-first fallback model is a good fit.
Connection failures on local services are treated as cheap and fast, while cloud endpoints can keep normal retry behavior.
Earshot should preserve that distinction so offline-first setups do not feel slow when a local service is down.

VoiceMode's silence detection and VAD flow is worth studying for #5 and #7.
It uses WebRTC VAD in `record_audio_with_silence_detection`, with a minimum recording duration, silence threshold, and explicit control-channel checks for stop and skip events.
Earshot should adapt the state-machine idea, but should use its own implementation boundaries.

VoiceMode's saved audio and debug-file recovery pattern is useful.
The docs describe `latest-STT.wav`, `latest-TTS.mp3`, and saved transcription artifacts.
Earshot should offer similar recovery hooks because STT failures are otherwise opaque.

VoiceMode's CLI and MCP parity is useful.
The same `converse` behavior can be entered through a CLI subcommand or an MCP tool.
Earshot should keep a testable CLI path for every voice-loop stage, even when the primary user flow is through a daemon or agent adapter.

## Patterns To Avoid Or Rework

VoiceMode is fundamentally an MCP server that lets an LLM choose and invoke voice tools.
Earshot's conductor needs to own the event loop, routing, active speaker, barge-in policy, and terminal-agent transport directly.
That means VoiceMode should remain a design reference, not a runtime dependency.

VoiceMode's per-session MCP architecture resists a single-process multi-agent conductor.
The `conch` system coordinates speaking through shared state and session IDs, which is useful for separate Claude Code sessions but not the same as a central conductor with first-class agent identities and routing.
Earshot should model speakers and addressed agents directly instead of layering a conductor on top of MCP tool calls.

VoiceMode's MCP request lifecycle can be a poor fit for long or interruptible turns.
Its own docs describe hard caps for remote MCP waits and heartbeat-based remote liveness.
Earshot should make interrupt, idle detection, and output watching daemon-owned behaviors rather than MCP request behaviors.

VoiceMode's tool namespace and configuration are broad because it is a general voice MCP product.
Earshot should start narrower: wake or push-to-talk input, STT, agent send, output watcher, TTS, interrupt.
Additional service management tools can come later if they do not blur the core loop.

## Design Decision

Borrow VoiceMode's backend abstraction shape, not its product architecture.
The concrete pattern to carry forward is: `VoiceBackend` style interfaces configured by ordered OpenAI-compatible endpoints, with local-first discovery and cloud fallback.
The concrete pattern to avoid is: making the terminal agent responsible for orchestrating voice turns by choosing MCP tools at runtime.
