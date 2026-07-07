# P0-01 License Gate

Date: 2026-07-06.

This file records the Phase 0 license check for Earshot.
Each source was cloned into `/private/tmp/earshot-p0-01-licenses` and checked from the repository license file where one exists.
The GitHub issue requires direct license-file verification before Earshot adapts design patterns or runs a project at runtime.

## Summary

Earshot uses MIT for its own repository.
MIT is compatible with the permissive projects reviewed here and keeps the project simple for later release work.
Apache-2.0 dependencies are also acceptable for runtime use when NOTICE and license obligations are preserved.
MPL-2.0 is acceptable only as a separate dependency with MPL notices preserved and without copying MPL-covered source into Earshot.
Noncommercial model artifacts are not acceptable as default runtime dependencies for a general MIT project.
Projects without a repository license file, or with non-standard restrictions, must not be copied or used as runtime dependencies until a maintainer or legal review clears them.

## Dependency Verdicts

| Dependency | Source checked | License evidence | Pattern borrowing verdict | Runtime dependency verdict | Notes |
|---|---|---|---|---|---|
| VoiceMode | `mbailey/voicemode` at `391681e` | `LICENSE`: MIT License | OK to borrow design patterns with attribution awareness. | Do not use as a runtime dependency for Earshot because the issue locks VoiceMode as a design reference only. | Good reference for OpenAI-compatible STT and TTS endpoints, provider failover, local Whisper and Kokoro services, silence detection, and MCP integration boundaries. |
| agent-tts | `kiliman/agent-tts` at `2905104` | `LICENSE`: MIT License | OK to borrow patterns. | OK as a runtime dependency from a license standpoint if technically useful. | This repository was selected because it exactly matches the scoped description: real-time text-to-speech for Claude, OpenCode, and other agents. |
| NTM | `Dicklesworthstone/ntm` at `591f4da` | `LICENSE`: MIT License with OpenAI and Anthropic rider | Do not borrow implementation patterns without legal review. | Not OK as a runtime dependency for Earshot. | The rider denies rights to OpenAI, Anthropic, affiliates, and people acting for them, and it extends to derivative works and use. Treat as incompatible with this project until counsel says otherwise. |
| Tmux-Orchestrator | `Jedward23/Tmux-Orchestrator` at `7193530` | No `LICENSE` or `COPYING` file found. README says MIT. | Do not copy code or adapt specific implementation patterns until a real license file is added or written permission is obtained. | Not OK as a runtime dependency until the repository has a license file. | README-only license text is not enough for this ticket because the issue requires checking license files directly. |
| claude-tmux-orchestration | `primeline-ai/claude-tmux-orchestration` at `a511f1c` | `LICENSE`: MIT License | OK to borrow patterns. | OK as a runtime dependency from a license standpoint if technically useful. | License is permissive and conventional. |
| openWakeWord | `dscripka/openWakeWord` at `368c037` | `LICENSE`: Apache License 2.0. README says bundled pretrained models are CC BY-NC-SA 4.0. | OK to borrow code architecture patterns from Apache-2.0 source. | Code is OK as a runtime dependency with Apache-2.0 notices, but bundled pretrained models are not OK as default runtime artifacts. | Any future wake-word issue should either train or source permissively licensed models, or make noncommercial models explicit opt-in user downloads. |
| Piper | `rhasspy/piper` at `73c04d8` | `LICENSE.md`: MIT License | OK to borrow patterns. | OK as a runtime dependency from a license standpoint. | License is permissive and conventional. |
| Kokoro | `hexgrad/kokoro` at `dfb907a` | `LICENSE`: Apache License 2.0 | OK to borrow patterns. | OK as a runtime dependency with Apache-2.0 notice handling. | Repository documentation also describes the weights as Apache-licensed. Keep model provenance visible in release docs. |
| Coqui TTS | `coqui-ai/TTS` at `dbf1a08` | `LICENSE.txt`: Mozilla Public License 2.0 | OK to learn high-level patterns. | Use only as an external dependency with MPL-2.0 obligations preserved. | Do not copy MPL-covered source into Earshot unless Earshot is prepared to publish those covered-file modifications under MPL-2.0. |
| Silero VAD | `snakers4/silero-vad` at `b163605` | `LICENSE`: MIT License | OK to borrow patterns. | OK as a runtime dependency from a license standpoint. | Repository also contains dataset documentation with CC BY-NC-SA 4.0 for datasets, so avoid redistributing training datasets or derived noncommercial artifacts without a separate check. |
| faster-whisper | `SYSTRAN/faster-whisper` at `ed9a06c` | `LICENSE`: MIT License | OK to borrow patterns. | OK as a runtime dependency from a license standpoint. | License is permissive and conventional. |
| NumPy | `numpy/numpy` at `634b462` | `LICENSE.txt`: BSD 3-Clause License | OK to borrow patterns. | OK as a runtime dependency with BSD notice handling. | Added by the audio input pipeline for frame buffers and sample conversion. |
| sounddevice | `spatialaudio/python-sounddevice` at `88de286` | `LICENSE`: MIT License | OK to borrow patterns. | OK as a runtime dependency from a license standpoint. | Added by the audio input pipeline for PortAudio microphone capture. |
| markdown-it-py | `executablebooks/markdown-it-py` at `2c3f71e` | `LICENSE`: MIT License | OK to borrow patterns. | OK as a runtime dependency from a license standpoint. | Added by the speech output pipeline for Markdown tokenization. |
| SciPy | `scipy/scipy` v1.10.0 | `LICENSE.txt`: BSD 3-Clause License | OK to borrow patterns. | OK as a runtime dependency with BSD notice handling. | Added explicitly by the API TTS fallback path for resampling fallback audio to the API backend's advertised sample rate. |

## Loud Flags

NTM is blocked for Earshot pattern copying and runtime use until legal review because its license has a non-standard rider targeting OpenAI and Anthropic.
Tmux-Orchestrator is blocked for copying or runtime dependency use because the cloned repository has no top-level license file.
openWakeWord source code is Apache-2.0, but its included pretrained models are CC BY-NC-SA 4.0 and should not be default runtime artifacts for an MIT project.
Coqui TTS is MPL-2.0, so runtime use is possible but source copying has file-level copyleft obligations.

## Earshot License

Earshot itself is licensed under MIT in the top-level `LICENSE` file.
That matches the project plan expectation, keeps compatibility simple with the MIT dependencies, and remains compatible with Apache-2.0 runtime dependencies when notices are preserved.
If Earshot later vendors or modifies MPL-2.0 code, keep those files clearly separated and comply with MPL-2.0 for the covered files.
Do not vendor NTM or Tmux-Orchestrator code under the current findings.
