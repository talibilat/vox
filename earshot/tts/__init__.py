"""Text-to-speech backends behind a common interface."""

from earshot.config import Config
from earshot.tts.base import TtsBackend


def _local_factory(config: Config):
    def make() -> TtsBackend:
        if config.tts.local.engine != "piper":
            raise NotImplementedError(
                f"local TTS engine {config.tts.local.engine!r} is not implemented; "
                "only 'piper' works today"
            )
        from earshot.tts.local_piper import PiperBackend

        return PiperBackend(voice=config.tts.local.voice, speed=config.tts.local.speed)

    return make


def create_backend(config: Config) -> TtsBackend:
    """Instantiate the TTS backend selected in config."""
    if config.tts.backend == "local":
        return _local_factory(config)()
    # config validation guarantees backend is "local" or "api"
    from earshot.tts.api_openai import ApiTtsBackend

    return ApiTtsBackend(
        base_url=config.tts.api.base_url,
        api_key_env=config.tts.api.api_key_env,
        model=config.tts.api.model,
        voice=config.tts.api.voice,
        fallback_factory=_local_factory(config) if config.tts.api.fallback_to_local else None,
    )


__all__ = ["TtsBackend", "create_backend"]
