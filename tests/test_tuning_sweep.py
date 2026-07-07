"""Tests for reproducible tuning sweep fixtures."""

import wave

import numpy as np
import pytest

from spikes import tuning_sweep


@pytest.mark.parametrize("speech_samples", [tuning_sweep.SR * 5, tuning_sweep.SR * 31])
def test_stage_gen_writes_s3_conversation_as_30_seconds(tmp_path, monkeypatch, speech_samples):
    def fake_say_wav(*args, **kwargs):
        pass

    def fake_read_wav(path):
        return np.arange(speech_samples, dtype=np.int16)

    monkeypatch.setattr(tuning_sweep, "say_wav", fake_say_wav)
    monkeypatch.setattr(tuning_sweep, "read_wav", fake_read_wav)

    tuning_sweep.stage_gen(str(tmp_path))

    with wave.open(str(tmp_path / "s3_conversation.wav")) as wav:
        assert wav.getnframes() == tuning_sweep.SR * 30
