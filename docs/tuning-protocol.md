# P3-01 Tuning Protocol and Results

Date: 2026-07-07.
Machine: Apple M4 Pro, macOS, real Piper voice and speaker for latency runs.
Everything here reproduces with `spikes/tuning_sweep.py`:

```sh
python spikes/tuning_sweep.py --workdir /tmp/earshot-tuning --stage all
```

## Scenarios (stage `gen`)

Thirty seconds each, synthesized deterministically (seeded), 16kHz mono:

| Scenario | Content | Stands in for |
|---|---|---|
| S1 quiet | near-silence with faint noise floor | quiet room during agent playback |
| S2 music | detuned harmonic stack with a 2Hz beat envelope | music playing near the mic |
| S3 conversation | real synthesized speech attenuated to background level | other people talking nearby |
| S4 keyboard | impulse clicks at typing cadence with decay | typing next to the mic |

Genuine-speech onset clips (four short interjections after 2.000s of known
silence) measure detection lag against ground truth.
Pass bars: zero false fires in S1 (the acceptance's quiet-room bar); as low
as achievable in S2/S4; S3 is expected to fire (background speech IS
speech to a VAD) and is mitigated by the documented headset assumption and
the `earshot interrupt` escape hatch, not by thresholds.

## VAD threshold sweep (stage `vad`)

False speech-onset fires per 30s scenario, and onset lag on the four
genuine clips:

| threshold | quiet | music | conversation | keyboard | onset lag ms |
|---|---|---|---|---|---|
| 0.3 | 0 | 165 | 64 | 10 | 80, 80, 80, 80 |
| 0.4 | 0 | 106 | 57 | 2 | 160, 160, 80, 80 |
| 0.5 (old default) | 0 | 105 | 46 | 2 | 160, 160, 160, 80 |
| **0.6 (new default)** | **0** | **0** | 38 | **0** | **160, 160, 160, 80** |
| 0.7 | 0 | 0 | 30 | 0 | 160, 160, 160, 160 |
| 0.8 | 0 | 0 | 27 | 0 | 240, 160, 160, 160 |
| 0.9 | 0 | 0 | 13 | 0 | 240, 160, 160, 240 |

**Decision: `barge_in.vad_threshold` default moves 0.5 to 0.6.**
0.6 is a pure win: identical onset lag to 0.5 but zero false barge-ins
from music and keyboard, which at 0.5 fired over a hundred times in 30s.
Background conversation cannot be thresholded away (still 13 fires at 0.9,
at the cost of blowing the latency budget); headset use or modest speaker
volume remains the documented operating assumption.

## Wake-word sweep (stage `wake`)

False fires per 30s scenario and detections of two unseen-voice positives:

| sensitivity | patience | quiet | music | conversation | keyboard | detected |
|---|---|---|---|---|---|---|
| 0.85 | 2 | 2 | 0 | 0 | 0 | 2/2 |
| 0.85 | 3 | 0 | 0 | 0 | 0 | 2/2 |
| **0.90** | **3** | **0** | **0** | **0** | **0** | **2/2** |
| 0.90 | 4 | 0 | 0 | 0 | 0 | 0/2 |
| 0.95 (old default) | 4 (old) | 0 | 0 | 0 | 0 | 0/2 |

**Decision: `wake_word.sensitivity` 0.95 to 0.9, `patience` 4 to 3.**
The old operating point never fired falsely but also never fired at all on
an unseen voice; 0.9/3 keeps a spotless false-fire record across two
minutes of adversarial audio while detecting every positive.
The model itself is not the limiter at this operating point, so no retrain
was needed (the escalation path recorded in docs/latency-spike.md stands).

## Addressing accuracy (stage `addressing`)

Sixteen spoken commands (two unseen voices, three-agent fleet, real tiny.en
STT in the loop): **0 misroutes, 0 clarify-or-misses (16/16 correct)**,
including transcripts the STT genuinely mangled ("Marvin and do the last
change", "check the death").
Matcher thresholds (route 0.80, clarify 0.60) are unchanged; the corpus
suite in tests/test_addressing.py remains the regression gate.

## Interrupt latency re-validation (post-tuning)

Instrumented stop path (cancel + stop_and_flush while audio audibly plays,
real Piper + real speaker, 5 cycles): **median 124ms, worst 140ms**,
within the 200ms bar for the instrumented portion.
End-to-end accounting including detection: the frame-quantized onset lag
measured above adds 80 to 160ms of audio time at the shipped threshold, so
the common case lands near the 200ms target and the worst case is about
300ms; shrinking mic frames below 80ms is the known lever if real use
wants more margin (recorded in docs/tickets/P1-04.md).

## Remaining manual validation (needs a human and a room)

1. Live noisy-room wake test: play music at conversational volume, speak
   "Hey Earshot" from across the room, repeat ten times; expect at least
   nine activations and zero false fires over ten minutes of mixed noise.
2. Live barge-in feel test with a headset: interrupt mid-response ten
   times; the instrumentation logs each latency (grep "barge-in" in the
   daemon log) for the real-room distribution.
3. A multi-hour 16-agent session watching RSS (the watcher buffers are
   size-capped by design; this validates there is no other growth).
