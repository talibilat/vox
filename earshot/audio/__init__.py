"""Audio input/output primitives shared by the voice pipelines."""

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80ms at 16kHz, the cadence openWakeWord models expect
