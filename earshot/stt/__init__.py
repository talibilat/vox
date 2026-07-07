"""Speech-to-text backends behind a common interface."""

from earshot.config import Config
from earshot.stt.base import SttBackend


def _local_factory(config: Config):
    def make() -> SttBackend:
        from earshot.stt.local_whisper import LocalWhisperBackend

        return LocalWhisperBackend(
            model=config.stt.local.model,
            device=config.stt.local.device,
            compute_type=config.stt.local.compute_type,
        )

    return make


def create_backend(config: Config) -> SttBackend:
    """Instantiate the STT backend selected in config."""
    if config.stt.backend == "local":
        return _local_factory(config)()
    # config validation guarantees backend is "local" or "api"
    from earshot.stt.api_openai import ApiSttBackend

    return ApiSttBackend(
        base_url=config.stt.api.base_url,
        api_key_env=config.stt.api.api_key_env,
        model=config.stt.api.model,
        fallback_factory=_local_factory(config) if config.stt.api.fallback_to_local else None,
    )


__all__ = ["SttBackend", "create_backend"]
