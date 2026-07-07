"""Configuration schema, loading, and validation for Earshot.

The full schema is defined here, including fields consumed by the Phase 1
opencode voice loop and fields that later phases consume (multi-agent
conductor settings, barge-in settings for the interrupt path). Every field is
a dataclass attribute with a default; a config file only needs to override what
it changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path("~/.config/earshot/config.yaml").expanduser()
DEFAULT_STATE_DIR = Path("~/.local/state/earshot").expanduser()

HARNESSES = ("opencode", "claude-code", "codex")
BACKENDS = ("local", "api")
TTS_ENGINES = ("piper", "kokoro")
CODE_BLOCK_MODES = ("summarize", "skip", "read")
DEFAULT_OPENCODE_MODEL = "opencode/deepseek-v4-flash-free"


class ConfigError(Exception):
    """A config file problem, with the offending key path in the message."""


@dataclass
class WakeWordConfig:
    phrase: str = "hey earshot"
    model_path: str | None = None
    sensitivity: float = 0.95
    patience: int = 4  # consecutive windows above sensitivity to fire


@dataclass
class SttLocalConfig:
    model: str = "base.en"
    device: str = "cpu"
    compute_type: str = "int8"


@dataclass
class SttApiConfig:
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "whisper-1"


@dataclass
class SttConfig:
    backend: str = "local"
    local: SttLocalConfig = field(default_factory=SttLocalConfig)
    api: SttApiConfig = field(default_factory=SttApiConfig)


@dataclass
class TtsLocalConfig:
    engine: str = "piper"
    voice: str = "en_US-lessac-medium"
    speed: float = 1.0


@dataclass
class TtsApiConfig:
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "tts-1"
    voice: str = "alloy"


@dataclass
class TtsConfig:
    backend: str = "local"
    local: TtsLocalConfig = field(default_factory=TtsLocalConfig)
    api: TtsApiConfig = field(default_factory=TtsApiConfig)


@dataclass
class AgentConfig:
    harness: str = "opencode"
    command: str | None = None  # explicit opencode-compatible serve command override
    workdir: str = "~"
    # "provider/model-id" (or the harness's own model naming); None uses the
    # harness-specific default chosen by its adapter.
    model: str | None = None
    tmux_pane: str | None = None  # only for a harness on the tmux fallback path


@dataclass
class BargeInConfig:
    vad_threshold: float = 0.5
    interrupt_hotkey: str | None = None  # push-to-interrupt escape hatch


@dataclass
class DaemonConfig:
    log_file: str = str(DEFAULT_STATE_DIR / "earshot.log")
    pid_file: str = str(DEFAULT_STATE_DIR / "earshot.pid")


@dataclass
class Config:
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    code_blocks: str = "summarize"
    agents: dict[str, AgentConfig] = field(default_factory=lambda: {"main": AgentConfig()})
    barge_in: BargeInConfig = field(default_factory=BargeInConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)


_SECTION_TYPES = {
    "wake_word": WakeWordConfig,
    "stt": SttConfig,
    "tts": TtsConfig,
    "barge_in": BargeInConfig,
    "daemon": DaemonConfig,
}
_NESTED_TYPES = {
    ("stt", "local"): SttLocalConfig,
    ("stt", "api"): SttApiConfig,
    ("tts", "local"): TtsLocalConfig,
    ("tts", "api"): TtsApiConfig,
}


def _fail(path: str, message: str) -> None:
    raise ConfigError(f"{path}: {message}")


def _require_mapping(value: object, path: str) -> dict:
    if not isinstance(value, dict):
        _fail(path, f"expected a mapping, got {type(value).__name__}")
    return value


def _set_fields(
    obj: object, data: dict, path: str, nested: dict[str, object] | None = None
) -> None:
    """Assign mapping keys onto a dataclass, rejecting unknown keys."""
    valid = set(obj.__dataclass_fields__)  # type: ignore[attr-defined]
    for key, value in data.items():
        if key not in valid:
            _fail(f"{path}.{key}", f"unknown key (valid keys: {', '.join(sorted(valid))})")
        if nested and key in nested:
            subpath = f"{path}.{key}"
            _set_fields(getattr(obj, key), _require_mapping(value, subpath), subpath)
        else:
            setattr(obj, key, value)


def _from_dict(data: dict) -> Config:
    config = Config()
    top_valid = set(config.__dataclass_fields__)
    for key, value in data.items():
        if key not in top_valid:
            _fail(key, f"unknown key (valid keys: {', '.join(sorted(top_valid))})")
        if key in _SECTION_TYPES:
            section = getattr(config, key)
            nested = {name: typ for (sec, name), typ in _NESTED_TYPES.items() if sec == key}
            _set_fields(section, _require_mapping(value, key), key, nested)
        elif key == "agents":
            agents_data = _require_mapping(value, "agents")
            config.agents = {}
            for name, agent_value in agents_data.items():
                agent = AgentConfig()
                agent_path = f"agents.{name}"
                _set_fields(agent, _require_mapping(agent_value, agent_path), agent_path)
                config.agents[name] = agent
        else:
            setattr(config, key, value)
    return config


def _check_enum(value: object, options: tuple[str, ...], path: str) -> None:
    if value not in options:
        _fail(path, f"expected one of {', '.join(repr(o) for o in options)}, got {value!r}")


def _check_range(value: object, low: float, high: float, path: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        _fail(path, f"expected a number, got {type(value).__name__}")
    if not low <= value <= high:
        _fail(path, f"expected a value between {low} and {high}, got {value}")


def _check_str(value: object, path: str, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        _fail(path, f"expected a non-empty string, got {value!r}")


def _check_model_pin(value: object, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        _fail(path, f"expected a non-empty string, got {value!r}")
    provider, separator, model_id = value.partition("/")
    if separator != "/" or not provider.strip() or not model_id.strip() or "/" in model_id:
        _fail(path, f"expected provider/model-id, got {value!r}")


def validate(config: Config) -> Config:
    """Validate every field; raise ConfigError with the key path on failure."""
    _check_str(config.wake_word.phrase, "wake_word.phrase")
    _check_str(config.wake_word.model_path, "wake_word.model_path", optional=True)
    _check_range(config.wake_word.sensitivity, 0.0, 1.0, "wake_word.sensitivity")
    if (
        not isinstance(config.wake_word.patience, int)
        or isinstance(config.wake_word.patience, bool)
        or config.wake_word.patience < 1
    ):
        _fail("wake_word.patience", f"expected an integer >= 1, got {config.wake_word.patience!r}")

    _check_enum(config.stt.backend, BACKENDS, "stt.backend")
    _check_str(config.stt.local.model, "stt.local.model")
    _check_str(config.stt.local.device, "stt.local.device")
    _check_str(config.stt.local.compute_type, "stt.local.compute_type")
    _check_str(config.stt.api.base_url, "stt.api.base_url")
    _check_str(config.stt.api.api_key_env, "stt.api.api_key_env")
    _check_str(config.stt.api.model, "stt.api.model")

    _check_enum(config.tts.backend, BACKENDS, "tts.backend")
    _check_enum(config.tts.local.engine, TTS_ENGINES, "tts.local.engine")
    _check_str(config.tts.local.voice, "tts.local.voice")
    if (
        not isinstance(config.tts.local.speed, int | float)
        or isinstance(config.tts.local.speed, bool)
        or config.tts.local.speed <= 0
    ):
        _fail("tts.local.speed", f"expected a number > 0, got {config.tts.local.speed!r}")
    _check_str(config.tts.api.base_url, "tts.api.base_url")
    _check_str(config.tts.api.api_key_env, "tts.api.api_key_env")
    _check_str(config.tts.api.model, "tts.api.model")
    _check_str(config.tts.api.voice, "tts.api.voice")

    _check_enum(config.code_blocks, CODE_BLOCK_MODES, "code_blocks")

    if not config.agents:
        _fail("agents", "at least one agent must be configured")
    for name, agent in config.agents.items():
        _check_str(name, "agents (agent name)")
        _check_enum(agent.harness, HARNESSES, f"agents.{name}.harness")
        _check_str(agent.command, f"agents.{name}.command", optional=True)
        _check_str(agent.workdir, f"agents.{name}.workdir")
        if agent.harness == "opencode" and agent.model is not None:
            _check_model_pin(agent.model, f"agents.{name}.model")
        else:
            _check_str(agent.model, f"agents.{name}.model", optional=True)
        _check_str(agent.tmux_pane, f"agents.{name}.tmux_pane", optional=True)

    _check_range(config.barge_in.vad_threshold, 0.0, 1.0, "barge_in.vad_threshold")
    _check_str(config.barge_in.interrupt_hotkey, "barge_in.interrupt_hotkey", optional=True)

    _check_str(config.daemon.log_file, "daemon.log_file")
    _check_str(config.daemon.pid_file, "daemon.pid_file")
    return config


def load(path: str | os.PathLike | None = None) -> Config:
    """Load and validate config; create a commented default file on first run.

    With no explicit path, reads DEFAULT_CONFIG_PATH and generates it (plus
    parent directories) if missing. With an explicit path, the file must
    exist.
    """
    explicit = path is not None
    config_path = Path(path).expanduser() if explicit else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        if explicit:
            raise ConfigError(f"config file not found: {config_path}")
        write_default(config_path)
    raw = config_path.read_text()
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise ConfigError(f"{config_path} is not valid YAML: {e}") from e
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{config_path}: top level must be a mapping")
    return validate(_from_dict(data))


DEFAULT_CONFIG_TEMPLATE = """\
# Earshot configuration. Every key is optional; missing keys use these defaults.

wake_word:
  phrase: hey earshot
  model_path: null          # path to a trained openWakeWord .onnx model
  sensitivity: 0.95         # detection threshold, 0..1
  patience: 4               # consecutive windows above threshold to fire

stt:
  backend: local            # local works today; api is reserved for #10
  local:
    model: base.en          # faster-whisper model size
    device: cpu
    compute_type: int8
  api:
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    model: whisper-1

tts:
  backend: local            # local works today; api is reserved for #10
  local:
    engine: piper           # piper works today; kokoro is reserved
    voice: en_US-lessac-medium
    speed: 1.0
  api:
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    model: tts-1
    voice: alloy

code_blocks: summarize      # fenced code blocks: summarize | skip | read

# Spoken agent name -> how to reach that agent.
agents:
  main:
    harness: opencode       # opencode works today; claude-code/codex are reserved
    command: null           # optional opencode-compatible serve command; --port is appended
    workdir: "~"
    model: null             # null = the harness's default (opencode pins {model})
    tmux_pane: null         # only for a harness on the tmux fallback path

barge_in:
  vad_threshold: 0.5        # 0..1, Silero VAD speech probability
  interrupt_hotkey: null    # optional label; bind `earshot interrupt` outside Earshot

daemon:
  log_file: {log_file}
  pid_file: {pid_file}
"""


def write_default(config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        DEFAULT_CONFIG_TEMPLATE.format(
            log_file="~/.local/state/earshot/earshot.log",
            pid_file="~/.local/state/earshot/earshot.pid",
            model=DEFAULT_OPENCODE_MODEL,
        )
    )
