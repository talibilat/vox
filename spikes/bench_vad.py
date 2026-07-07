#!/usr/bin/env python3
"""P0-03: Silero VAD speech-onset detection latency.

Two numbers matter for the 200ms barge-in budget:
1. Compute cost: per-chunk inference time on a 512-sample (32ms) chunk.
2. Detection lag: how much audio time passes between true speech onset and
   the chunk on which VAD first reports speech.

Test signal: 2.000s of digital silence spliced onto a real speech clip, so
the true onset position is known exactly.

Usage: python3 bench_vad.py <speech_wav_16k_mono> [runs]
"""

import statistics
import sys
import time

import numpy as np
import soundfile as sf
from silero_vad import VADIterator, load_silero_vad

SR = 16000
CHUNK = 512  # 32ms, the size silero-vad expects at 16kHz


def main():
    wav_path = sys.argv[1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    speech, sr = sf.read(wav_path, dtype="float32")
    assert sr == SR, f"expected 16kHz, got {sr}"
    silence = np.zeros(SR * 2, dtype="float32")
    signal = np.concatenate([silence, speech])
    true_onset_s = 2.0

    model = load_silero_vad()

    onset_lags, chunk_times = [], []
    for _ in range(runs):
        vad = VADIterator(model, threshold=0.5, sampling_rate=SR)
        detected_at = None
        for i in range(0, len(signal) - CHUNK, CHUNK):
            chunk = signal[i : i + CHUNK]
            t0 = time.perf_counter()
            event = vad(chunk, return_seconds=True)
            chunk_times.append(time.perf_counter() - t0)
            if event and "start" in event and detected_at is None:
                # end of this chunk is when the decision is available
                detected_at = (i + CHUNK) / SR
                break
        vad.reset_states()
        assert detected_at is not None, "VAD never detected speech"
        onset_lags.append(detected_at - true_onset_s)

    print(
        f"silero-vad onset detection ({runs} runs, threshold=0.5, {CHUNK}-sample chunks)"
    )
    print(
        f"  audio-time lag from true onset: median={statistics.median(onset_lags)*1000:.0f}ms "
        f"worst={max(onset_lags)*1000:.0f}ms"
    )
    print(
        f"  per-chunk inference: median={statistics.median(chunk_times)*1000:.2f}ms "
        f"worst={max(chunk_times)*1000:.2f}ms over {len(chunk_times)} chunks"
    )
    budget = 200
    lag = statistics.median(onset_lags) * 1000
    infer = statistics.median(chunk_times) * 1000
    # One chunk of mic buffering already counted inside the lag; add inference
    # cost and a generous 20ms allowance for stopping playback.
    est = lag + infer + 20
    print(
        f"  interrupt-path estimate: {lag:.0f}ms detection + {infer:.2f}ms inference "
        f"+ 20ms audio-stop = {est:.0f}ms"
    )
    print(f"  headroom vs {budget}ms budget: {budget - est:.0f}ms")


if __name__ == "__main__":
    main()
