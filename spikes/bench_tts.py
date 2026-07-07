#!/usr/bin/env python3
"""P0-03: TTS first-audio latency benchmark for Piper and Kokoro.

First-audio latency = time from submitting text to the first audio chunk
being available (the moment playback could start). Benchmarks both a full
paragraph submitted at once and a sentence-at-a-time streaming pattern.

Usage: python3 bench_tts.py <piper_voice.onnx> [runs]
"""

import statistics
import sys
import time

PARAGRAPH = (
    "The refactor is finished and all forty two tests pass. "
    "I moved the token refresh logic into its own module and added six new unit tests. "
    "The database migration from last week still applies cleanly. "
    "Let me know if you want me to push the branch."
)
SENTENCES = [s.strip() + "." for s in PARAGRAPH.split(". ") if s.strip().rstrip(".")]


def report(label, times, extra=""):
    print(
        f"  {label}: median={statistics.median(times)*1000:.0f}ms "
        f"worst={max(times)*1000:.0f}ms best={min(times)*1000:.0f}ms {extra}"
    )


def bench_piper(voice_path, runs):
    from piper import PiperVoice

    voice = PiperVoice.load(voice_path)
    print("piper (en_US-lessac-medium, CPU)")

    first, total = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        t_first = None
        n = 0
        for chunk in voice.synthesize(PARAGRAPH):
            if t_first is None:
                t_first = time.perf_counter() - t0
            n += len(chunk.audio_int16_bytes)
        total.append(time.perf_counter() - t0)
        first.append(t_first)
    report("paragraph first-audio", first)
    report("paragraph full-synth ", total, f"({n} bytes)")

    stream_first = []
    for _ in range(runs):
        t0 = time.perf_counter()
        chunk = next(iter(voice.synthesize(SENTENCES[0])))
        stream_first.append(time.perf_counter() - t0)
    report("first-sentence stream", stream_first)


def bench_kokoro(runs):
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    print("kokoro (Kokoro-82M af_heart, CPU/MPS)")

    first, total = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        t_first = None
        n = 0
        for _gs, _ps, audio in pipeline(PARAGRAPH, voice="af_heart"):
            if t_first is None:
                t_first = time.perf_counter() - t0
            n += len(audio)
        total.append(time.perf_counter() - t0)
        first.append(t_first)
    report("paragraph first-audio", first)
    report("paragraph full-synth ", total, f"({n} samples)")

    stream_first = []
    for _ in range(runs):
        t0 = time.perf_counter()
        next(iter(pipeline(SENTENCES[0], voice="af_heart")))
        stream_first.append(time.perf_counter() - t0)
    report("first-sentence stream", stream_first)


def main():
    voice_path = sys.argv[1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    bench_piper(voice_path, runs)
    bench_kokoro(runs)


if __name__ == "__main__":
    main()
