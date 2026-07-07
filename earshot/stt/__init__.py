"""Speech-to-text backends behind a common interface."""

from earshot.config import Config
from earshot.stt.base import SttBackend


def create_backend(config: Config) -> SttBackend:
    """Instantiate the STT backend selected in config."""
    if config.stt.backend == "local":
        from earshot.stt.local_whisper import LocalWhisperBackend

        return LocalWhisperBackend(
            model=config.stt.local.model,
            device=config.stt.local.device,
            compute_type=config.stt.local.compute_type,
        )
    # config validation guarantees backend is "local" or "api"
    raise NotImplementedError("the API STT backend is issue #10; only 'local' works today")


__all__ = ["SttBackend", "create_backend"]
