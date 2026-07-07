#!/usr/bin/env python3
"""P3-01 tuning sweeps: VAD thresholds, wake-word sensitivity, and
addressing accuracy, measured against the reproducible scenario set defined
in docs/tuning-protocol.md.

Stages:
  gen        synthesize the noise/speech scenario audio (macOS `say` + numpy)
  vad        sweep SpeechOnsetDetector thresholds across scenarios
  wake       sweep wake-word sensitivity/patience across scenarios
  addressing measure end-to-end misroutes: say audio -> tiny.en STT -> matcher

Usage: python3 spikes/tuning_sweep.py --workdir /path --stage all
"""

import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SR = 16000
FRAME = 1280


def say_wav(text, out_path, voice="Samantha", rate=180):
    aiff = out_path + ".aiff"
    subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", aiff, text], check=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", aiff, "-ar", str(SR), "-ac", "1", out_path],
        check=True,
    )
    os.remove(aiff)


def read_wav(path):
    import wave

    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def write_wav(path, audio):
    import wave

    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(audio.astype(np.int16).tobytes())


def _scale(audio, peak):
    peak_now = max(1, np.abs(audio).max())
    return (audio.astype(np.float32) * (peak / peak_now)).astype(np.int16)


def _fit_samples(audio, sample_count):
    if len(audio) >= sample_count:
        return audio[:sample_count]
    return np.concatenate([audio, np.zeros(sample_count - len(audio), dtype=np.int16)])


def stage_gen(workdir):
    rng = np.random.default_rng(7)
    os.makedirs(workdir, exist_ok=True)

    # S1 quiet room: near-silence with faint electrical noise.
    write_wav(f"{workdir}/s1_quiet.wav", (rng.standard_normal(SR * 30) * 30).astype(np.int16))

    # S2 music-like: sum of detuned harmonics with a beat envelope.
    t = np.arange(SR * 30) / SR
    music = sum(np.sin(2 * np.pi * f * t + i) for i, f in enumerate([220, 277, 330, 440, 554]))
    beat = 0.6 + 0.4 * np.square(np.sin(2 * np.pi * 2 * t))
    write_wav(f"{workdir}/s2_music.wav", _scale(music * beat, 6000))

    # S3 conversation nearby: real speech, attenuated to background level.
    say_wav(
        "So then I told him the deployment was fine, and he said the logs looked strange, "
        "and honestly we went back and forth about the database for a while, it was a whole thing, "
        "anyway the weather has been lovely lately and the coffee downstairs is much better now.",
        f"{workdir}/s3_speech_full.wav",
        voice="Daniel",
        rate=170,
    )
    speech = read_wav(f"{workdir}/s3_speech_full.wav")
    write_wav(f"{workdir}/s3_conversation.wav", _fit_samples(_scale(speech, 2500), SR * 30))

    # S4 keyboard: impulse clicks at typing cadence.
    keys = np.zeros(SR * 30)
    for start in rng.integers(0, SR * 30 - 400, 260):
        keys[start : start + 400] += rng.standard_normal(400) * np.exp(-np.arange(400) / 60)
    write_wav(f"{workdir}/s4_keyboard.wav", _scale(keys, 9000))

    # Genuine interjections for onset-lag measurement: silence then speech.
    for i, phrase in enumerate(["stop", "wait a moment", "no, hold on", "actually, change that"]):
        say_wav(phrase, f"{workdir}/onset_{i}_raw.wav", voice="Tessa", rate=180)
        clip = read_wav(f"{workdir}/onset_{i}_raw.wav")
        write_wav(
            f"{workdir}/onset_{i}.wav",
            np.concatenate([np.zeros(SR * 2, dtype=np.int16), _scale(clip, 16000)]),
        )
    print("scenarios written")


def frames(audio):
    total = len(audio) - len(audio) % FRAME
    for start in range(0, total, FRAME):
        yield audio[start : start + FRAME]


def stage_vad(workdir):
    from earshot.barge.vad import SpeechOnsetDetector

    scenarios = ["s1_quiet", "s2_music", "s3_conversation", "s4_keyboard"]
    print("VAD sweep: false fires per 30s scenario, onset lag on genuine speech")
    print(
        f"{'thr':>5} | "
        + " | ".join(f"{s[3:]:>13}" for s in scenarios)
        + " | onset lag ms (4 clips)"
    )
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        fires = []
        for scenario in scenarios:
            detector = SpeechOnsetDetector(threshold=threshold)
            count = 0
            for frame in frames(read_wav(f"{workdir}/{scenario}.wav")):
                if detector.onset(frame):
                    count += 1
                    detector.reset()  # count distinct fires
            fires.append(count)
        lags = []
        for i in range(4):
            detector = SpeechOnsetDetector(threshold=threshold)
            fired_at = None
            for index, frame in enumerate(frames(read_wav(f"{workdir}/onset_{i}.wav"))):
                if detector.onset(frame):
                    fired_at = index
                    break
            true_frame = SR * 2 // FRAME
            lags.append("miss" if fired_at is None else (fired_at - true_frame + 1) * 80)
        print(f"{threshold:>5} | " + " | ".join(f"{f:>13}" for f in fires) + f" | {lags}")


def stage_wake(workdir):
    from earshot.wakeword import WakeWordDetector

    model = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "spikes",
        "models",
        "hey_earshot.onnx",
    )
    # Positives: unseen voice saying the wake phrase at two rates.
    for i, rate in enumerate((160, 200)):
        say_wav("Hey Earshot.", f"{workdir}/wake_pos_{i}_raw.wav", voice="Tessa", rate=rate)
        clip = read_wav(f"{workdir}/wake_pos_{i}_raw.wav")
        write_wav(
            f"{workdir}/wake_pos_{i}.wav",
            np.concatenate([np.zeros(SR * 2, dtype=np.int16), _scale(clip, 16000)]),
        )
    scenarios = ["s1_quiet", "s2_music", "s3_conversation", "s4_keyboard"]
    print("wake sweep: false fires per 30s scenario / positives detected (of 2)")
    print(
        f"{'sens':>5} {'pat':>4} | " + " | ".join(f"{s[3:]:>13}" for s in scenarios) + " | detected"
    )
    for sensitivity in (0.85, 0.9, 0.95):
        for patience in (2, 3, 4):
            fires = []
            for scenario in scenarios:
                detector = WakeWordDetector(model, sensitivity=sensitivity, patience=patience)
                count = 0
                for frame in frames(read_wav(f"{workdir}/{scenario}.wav")):
                    if detector.detected(frame):
                        count += 1
                fires.append(count)
            detected = 0
            for i in range(2):
                detector = WakeWordDetector(model, sensitivity=sensitivity, patience=patience)
                if any(
                    detector.detected(f) for f in frames(read_wav(f"{workdir}/wake_pos_{i}.wav"))
                ):
                    detected += 1
            print(
                f"{sensitivity:>5} {patience:>4} | "
                + " | ".join(f"{f:>13}" for f in fires)
                + f" | {detected}/2"
            )


ADDRESS_COMMANDS = [
    ("marvin", "run the tests"),
    ("marvin", "undo the last change"),
    ("olivia", "check the diff"),
    ("olivia", "keep going"),
    ("sebastian", "write the docs"),
    ("sebastian", "try a different approach"),
    (None, "make it faster"),
    (None, "that looks right, continue"),
]


def stage_addressing(workdir):
    from earshot.conductor.addressing import ROUTE_THRESHOLD, extract_address
    from earshot.stt.local_whisper import LocalWhisperBackend

    names = ["marvin", "olivia", "sebastian"]
    stt = LocalWhisperBackend(model="tiny.en")
    voices = ["Tessa", "Reed (English (US))"]
    total = 0
    misroutes = 0
    clarifies = 0
    for voice in voices:
        for target, command in ADDRESS_COMMANDS:
            utterance = f"{target}, {command}" if target else command
            path = f"{workdir}/addr_{voice.split()[0]}_{total}.wav"
            say_wav(utterance, path, voice=voice, rate=185)
            transcript = stt.transcribe(read_wav(path), SR)
            address = extract_address(transcript, names)
            routed = address.name if address.confidence >= ROUTE_THRESHOLD else None
            ambiguous = address.name is not None and address.confidence < ROUTE_THRESHOLD
            outcome = "ok"
            if routed != target:
                if ambiguous or routed is None and target is not None:
                    clarifies += 1
                    outcome = "clarify/miss"
                else:
                    misroutes += 1
                    outcome = "MISROUTE"
            total += 1
            print(f"  [{outcome:>12}] said={utterance!r} heard={transcript!r} -> {routed!r}")
    print(
        f"addressing over {total} spoken commands (2 voices, real tiny.en STT): "
        f"{misroutes} misroutes ({100 * misroutes / total:.0f}%), "
        f"{clarifies} clarify-or-miss ({100 * clarifies / total:.0f}%)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument(
        "--stage", choices=["gen", "vad", "wake", "addressing", "all"], default="all"
    )
    args = parser.parse_args()
    if args.stage in ("gen", "all"):
        stage_gen(args.workdir)
    if args.stage in ("vad", "all"):
        stage_vad(args.workdir)
    if args.stage in ("wake", "all"):
        stage_wake(args.workdir)
    if args.stage in ("addressing", "all"):
        stage_addressing(args.workdir)


if __name__ == "__main__":
    main()
