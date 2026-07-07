import sys
from types import ModuleType

import numpy as np


def install_fake_openwakeword(monkeypatch, *, model_cls=None, vad_cls=None, download_models=None):
    package = ModuleType("openwakeword")
    package.__path__ = []
    model_module = ModuleType("openwakeword.model")
    vad_module = ModuleType("openwakeword.vad")
    utils_module = ModuleType("openwakeword.utils")
    if model_cls is not None:
        model_module.Model = model_cls
    if vad_cls is not None:
        vad_module.VAD = vad_cls
    if download_models is not None:
        utils_module.download_models = download_models

    monkeypatch.setitem(sys.modules, "openwakeword", package)
    monkeypatch.setitem(sys.modules, "openwakeword.model", model_module)
    monkeypatch.setitem(sys.modules, "openwakeword.vad", vad_module)
    monkeypatch.setitem(sys.modules, "openwakeword.utils", utils_module)


def test_wakeword_detector_downloads_missing_openwakeword_resources(monkeypatch):
    model_attempts = []
    downloads = []

    class Model:
        def __init__(self, **_kwargs):
            model_attempts.append("attempt")
            if len(model_attempts) == 1:
                raise RuntimeError("NoSuchFile: melspectrogram.onnx")
            self.models = {"hey_earshot": object()}

        def reset(self):
            pass

        def predict(self, _frame):
            return {"hey_earshot": 0.0}

    install_fake_openwakeword(
        monkeypatch,
        model_cls=Model,
        download_models=lambda: downloads.append("download"),
    )

    from earshot.wakeword.detector import WakeWordDetector

    detector = WakeWordDetector("spikes/models/hey_earshot.onnx")

    assert detector.detected(np.zeros(1280, dtype=np.int16)) is False
    assert downloads == ["download"]
    assert len(model_attempts) == 2


def test_end_of_speech_detector_downloads_missing_openwakeword_resources(monkeypatch):
    vad_attempts = []
    downloads = []

    class VAD:
        def __init__(self):
            vad_attempts.append("attempt")
            if len(vad_attempts) == 1:
                raise RuntimeError("NoSuchFile: silero_vad.onnx")

        def reset_states(self):
            pass

        def predict(self, _frame, *, frame_size):
            return 0.0

    install_fake_openwakeword(
        monkeypatch,
        vad_cls=VAD,
        download_models=lambda: downloads.append("download"),
    )

    from earshot.audio.endpointing import EndOfSpeechDetector

    detector = EndOfSpeechDetector(max_utterance_ms=200)

    assert detector.finished(np.zeros(1280, dtype=np.int16)) is False
    assert downloads == ["download"]
    assert len(vad_attempts) == 2
