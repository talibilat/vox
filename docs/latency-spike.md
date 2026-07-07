# P0-03 Voice-Stack Spike: Latency Benchmarks and Wake-Word Model

Date: 2026-07-07.
Machine: Apple M4 Pro, 24 GB RAM, macOS (darwin arm64). All engines on CPU (int8 for faster-whisper); no GPU/ANE tuning attempted yet, so these are conservative numbers.
Benchmark scripts: `spikes/bench_stt.py`, `spikes/bench_tts.py`, `spikes/bench_vad.py`, `spikes/train_wakeword.py`.
Every measurement is 10 runs unless stated; medians and worst cases reported.

## Decisions made from these numbers

| Decision | Choice | Why |
|---|---|---|
| STT model | faster-whisper `base.en` (default), `tiny.en` (low-latency option) | base.en transcribes 10s of speech in 340ms median with perfect accuracy on the test clip; tiny.en is 187ms if we ever need it |
| TTS engine | **Piper** (`en_US-lessac-medium`) | 46ms median first-audio vs Kokoro's 277ms (sentence) / 1005ms (paragraph); 6 to 20x faster to first sound |
| Kokoro's role | Optional quality voice, sentence-at-a-time only | 277ms sentence-level first-audio is usable but pays a noticeable beat; never feed it whole paragraphs |
| Wake word | "Hey Earshot" | Model trained and evaluated below; phrase is phonetically distinct and matched the product name |
| Interrupt budget | Achievable: ~68ms estimated of the 200ms budget used | See VAD section |

## STT: faster-whisper transcription latency

Input: a 10.0s spoken instruction clip (16kHz mono), synthesized with a natural macOS voice, transcribed with beam_size=1, language pinned to en.
Latency = wall-clock of a full `transcribe()` call, which is exactly the end-of-speech to transcript-available wait.

| Model | Median | Worst | Best | Transcript quality |
|---|---|---|---|---|
| tiny.en | 187ms | 380ms | 184ms | Perfect on test clip |
| base.en | 340ms | 350ms | 337ms | Perfect on test clip |
| small.en | 1027ms | 1049ms | 1021ms | Perfect on test clip |

Verdict: `base.en` as the default (accuracy headroom for noisy real microphone audio at a third of a second), `tiny.en` as the configurable fast option, `small.en` only if real-world accuracy demands it.
Worst cases are within a few ms of medians after warm-up; the one 380ms outlier for tiny.en was the first post-load run.

## TTS: first-audio latency, Piper vs Kokoro

First-audio latency = time from submitting text to the first audio chunk available for playback.
Input paragraph: 4 sentences, 43 words. Sentence test uses its first sentence.

| Engine | Paragraph first-audio (median/worst) | Paragraph full synth | First sentence, streaming (median/worst) |
|---|---|---|---|
| Piper en_US-lessac-medium | 46ms / 74ms | 208ms | 47ms / 65ms |
| Kokoro-82M af_heart | 1005ms / 1311ms | 1005ms | 277ms / 319ms |

Winner: **Piper, by 6x (sentence) to 20x (paragraph)**.
Notes:

- Piper's Python API (`PiperVoice.synthesize`) natively yields per-sentence `AudioChunk`s, so incremental sentence-at-a-time synthesis is confirmed working out of the box. 46ms to first audio means TTS adds no perceptible lag to the voice loop.
- Kokoro synthesizes its whole input before yielding when given a paragraph (first-audio == full-synth), so it MUST be driven sentence-at-a-time. Even then, 277ms is 6x Piper. Keep it as an opt-in quality voice.
- Kokoro quirk: on first run it shells out to `uv pip install` to fetch spacy's `en_core_web_sm` model; in a venv not named `.venv` this fails with a confusing "No virtual environment found" unless `VIRTUAL_ENV` is exported.
- Coqui TTS was not benchmarked: both primary candidates hit the streaming requirement and Piper exceeds the latency bar by an order of magnitude, so the "include Coqui only if both underperform" condition was not met.

## VAD: Silero speech-onset detection vs the 200ms interrupt budget

Method: 2.000s of digital silence spliced ahead of real speech, streamed through `VADIterator` (threshold 0.5) in 512-sample (32ms) chunks; onset lag = end of the chunk where VAD first fires minus the true onset position. 10 runs.

| Measurement | Median | Worst |
|---|---|---|
| Onset detection lag (audio time) | 48ms | 48ms |
| Per-chunk inference (compute) | 0.08ms | 26.6ms (first-chunk warm-up) |

Interrupt-path budget on paper:

```
48ms   VAD onset lag (includes the 32ms chunk quantization)
0.1ms  VAD inference
20ms   generous allowance for stopping playback + flushing the TTS queue
-----
~68ms  total, leaving ~132ms headroom under the 200ms target
```

Verdict: the 200ms barge-in target is comfortably achievable on paper, with two thirds of the budget spare for real-world audio-device latency and scheduling jitter.
P3-01 later moved the shipped `barge_in.vad_threshold` default to 0.6 after the reproducible noise sweep in `docs/tuning-protocol.md`.

## Wake word: "Hey Earshot" first-pass model (openWakeWord)

Approach: kept openWakeWord's pretrained feature backbone (melspectrogram + Google speech embedding, the frontend every official model uses) and trained a small MLP head (16x96 window, ~50k params, exported to ONNX and loaded by `openwakeword.Model`).
Training data was fully synthetic, following the official pipeline's strategy at reduced scale, but limited to the macOS `say` generator that is committed in `spikes/train_wakeword.py`:

- Positives: 80 "Hey Earshot" clips from 20 macOS `say` voices at 4 rates.
- Negatives: 400 `say` clips over 20 phrases, including hard negatives ("Hey Marshall", "within earshot", "Hair shirt", "Hey Ella", "Hey Earl"), plus silence and noise.
- Augmentation on everything: noise, low volume, and +-8% pitch/tempo resampling.

Held-out test: 88 clips, none of whose voices appear in training (4 unseen `say` voices at unseen rates for 8 positives and 80 negatives over all 20 phrases).

Scoring uses a patience rule: the wake fires only when 4 consecutive 80ms windows all score above 0.95.
This conservative operating point eliminated false triggers on the held-out set, but missed most unseen-voice positives.
A real "Hey Earshot" holds a high score across several consecutive frames; a phonetic confusion usually spikes on one or two.

Results on the 88-clip held-out set (threshold 0.95, patience 4):

| Metric | Result |
|---|---|
| Positives detected | 3/8 (false-negative rate 62%) |
| False triggers | 0/80 (false-positive rate 0%) |

Failure analysis:

- The 5 misses are all unseen-voice positives whose patience-window scores stay below the 0.95 threshold.
- None of the 80 negatives triggered, including the deliberately adversarial hard negatives ("Hey Marshall", "within earshot", "Hair shirt", "Hey Ella", "Hey Earl").

Verdict: **"Hey Earshot" remains a plausible wake word, but this model is only a feasibility spike artifact.**
The pipeline (multi-voice synthetic generation, streaming-consistent feature extraction, ONNX head loadable by `openwakeword.Model`, patience-scored detection) works end to end and rejects normal conversation reliably, but the committed model does not yet have acceptable unseen-voice recall.
That recall gap is exactly what more positive-sample scale and real-noise tuning should address; nothing here proves the phrase or the architecture is production-ready.
P3-01 later changed the shipped config operating point to sensitivity 0.9 and patience 3 after the sweep detected 2/2 unseen-voice positives with zero false fires across the synthetic noise scenarios.
The trained artifact is committed at `spikes/models/hey_earshot.onnx` (11 KB) with the exact operating point baked into `spikes/train_wakeword.py`.

Hard-won implementation lessons (these cost most of the spike time and matter for #5):

1. Features for training MUST come from the streaming preprocessor (`AudioFeatures.__call__` + `get_features`), not the batch `_get_embeddings` path; the two produce subtly different features (correlation ~0.73 observed) and a head trained on batch features scores ~0 at streaming inference.
2. Raw embedding features (std ~15, range +-60) saturate a default-initialized dense layer into a constant function; per-dimension standardization must be baked into the exported graph.
3. Class imbalance needs the full-ratio positive weight; with a sqrt weight the head converges to "always negative", which already scores 97% accuracy on an imbalanced set.
4. The feature buffer initializes with template values, so a wake model in deployment should only be trusted once ~1.4s of real audio has flowed through; the test harness prepends silence to mimic always-on listening.

## Interrupt path, end to end (informative)

Wake-word inference itself runs on the same 1280-sample (80ms) cadence as openWakeWord's standard models and adds ~0.1ms compute per frame on this machine, so the ambient-listening loop (wake word + VAD in parallel) costs well under 1% CPU per stream.

## What this spike deliberately did not do

- No real-microphone testing (synthetic audio only); P3-01 adds reproducible synthetic-noise tuning and leaves live-room validation steps in `docs/tuning-protocol.md`.
- No GPU/CoreML acceleration; CPU numbers already clear every target.
- No API-mode latency (optional path, not latency-critical).
- The wake-word model is feasibility-grade; if real-world rates disappoint in #5, the escalation path is the official openWakeWord training pipeline at full scale (thousands of generated positives, the ACAV100M precomputed negative features, and their augmentation stack) which this spike's data pipeline already mirrors in miniature.
