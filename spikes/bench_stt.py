#!/usr/bin/env python3
"""P0-03: faster-whisper transcription latency benchmark.

Measures end-of-speech to transcript-available: the wall-clock time of
transcribe() on an already-recorded clip, which is exactly the wait a user
experiences after they stop talking.

Usage: python3 bench_stt.py <wav_path> [runs]
"""

import statistics
import sys
import time

from faster_whisper import WhisperModel

MODELS = ["tiny.en", "base.en", "small.en"]


def bench(model_name, wav, runs):
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    times = []
    text = ""
    for _ in range(runs):
        t0 = time.perf_counter()
        segments, _info = model.transcribe(wav, beam_size=1, language="en")
        text = " ".join(s.text.strip() for s in segments)  # generator: consume = actual work
        times.append(time.perf_counter() - t0)
    return times, text


def main():
    wav = sys.argv[1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    for name in MODELS:
        times, text = bench(name, wav, runs)
        print(f"model={name} runs={runs}")
        print(f"  median={statistics.median(times)*1000:.0f}ms worst={max(times)*1000:.0f}ms best={min(times)*1000:.0f}ms")
        print(f"  transcript: {text[:120]}")


if __name__ == "__main__":
    main()
