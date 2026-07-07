# Earshot Demo Script

The headline flow to record (terminal capture plus room audio, one take,
roughly 90 seconds). Needs a human voice; everything below is set up so the
recording session is just following the script.

## Setup (before recording)

1. Config with two agents in separate scratch repos and the wake model:

```yaml
wake_word:
  model_path: /path/to/hey_earshot.onnx
agents:
  marvin:
    harness: opencode
    workdir: ~/demo/backend
  olivia:
    harness: claude-code
    workdir: ~/demo/frontend
```

2. Headset on (the barge-in assumption), `earshot start`, wait for
   "voice loop listening" in the log.
3. Screen recording on the terminal tailing the log
   (`tail -f ~/.local/state/earshot/earshot.log`), audio recording on.

## The take

1. **Wake plus instruct**: "Hey Earshot" (pause for the wake log line)
   "marvin, list the files in this project and describe what it does."
2. **Spoken response**: let marvin's answer play for a few seconds.
3. **Barge-in**: talk over it mid-sentence: "actually, just name the three
   biggest files." Playback stops within a beat (the log prints the
   measured stop latency) and the follow-up runs without a wake word.
4. **Second agent by name**: "olivia, what testing framework does your
   project use?" (olivia answers; marvin keeps working in silence.)
5. **Status roll-call**: "agent status" ("marvin has finished; olivia is
   still working." or similar).
6. **Read-back on request**: "marvin, what's your response" to hear
   marvin's buffered answer again.
7. `earshot stop` on camera: every agent process exits with the daemon.

## After recording

- Trim, export, and link it from the README's demo line.
- Include one frame of the log showing a "barge-in ... stopped in NNms" line;
  it is the product's best single receipt.
