"""Resource bootstrap for openWakeWord's lazily downloaded ONNX files."""

from __future__ import annotations


def is_missing_openwakeword_resource(error: Exception) -> bool:
    message = f"{type(error).__name__}: {error}"
    return any(
        marker in message
        for marker in (
            "NoSuchFile",
            "melspectrogram.onnx",
            "embedding_model.onnx",
            "silero_vad.onnx",
        )
    )


def download_openwakeword_resources() -> None:
    from openwakeword.utils import download_models

    download_models()
