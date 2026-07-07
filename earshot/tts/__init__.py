"""Text-to-speech backends behind a common interface."""

from earshot.config import Config
from earshot.tts.base import TtsBackend


def create_backend(config: Config) -> TtsBackend:
    """Instantiate the TTS backend selected in config."""
    if config.tts.backend == "local":
        from earshot.tts.local_piper import PiperBackend

        return PiperBackend(voice=config.tts.local.voice, speed=config.tts.local.speed)
    # config validation guarantees backend is "local" or "api"
    raise NotImplementedError("the API TTS backend is issue #10; only 'local' works today")


__all__ = ["TtsBackend", "create_backend"]
