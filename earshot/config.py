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


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of silently
    keeping the last one (two agents with the same spoken name would
    otherwise collapse into one without a trace)."""


def _strict_mapping(loader: _StrictLoader, node, deep=False):
    seen = set()
    for key_node, _value in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ConfigError(
                f"duplicate key {key!r} in config (line {key_node.start_mark.line + 1})"
            )
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _strict_mapping)


@dataclass
class WakeWordConfig:
    phrase: str = "hey earshot"
    model_path: str | None = None
    # Tuned in the P3-01 sweep (docs/tuning-protocol.md): 0.9/3 detected
    # every positive with zero false fires across all noise scenarios, while
    # 0.95 missed unseen voices entirely.
    sensitivity: float = 0.9
    patience: int = 3  # consecutive windows above sensitivity to fire


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
    fallback_to_local: bool = False  # degrade to the local backend on API failure


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
    fallback_to_local: bool = False  # degrade to the local backend on API failure


@dataclass
class TtsConfig:
    backend: str = "local"
    local: TtsLocalConfig = field(default_factory=TtsLocalConfig)
    api: TtsApiConfig = field(default_factory=TtsApiConfig)


@dataclass
class AgentConfig:
    harness: str = "opencode"
    command: str | None = None  # explicit harness command override
    workdir: str = "~"
    # "provider/model-id" (or the harness's own model naming); None uses the
    # harness-specific default chosen by its adapter.
    model: str | None = None
    restart_on_death: bool = True  # fleet supervisor restarts this agent if it dies
    tmux_pane: str | None = None  # non-null session name selects tmux fallback transport


@dataclass
class BargeInConfig:
    # Tuned in the P3-01 sweep: 0.6 has the same speech-onset lag as 0.5
    # but zero false barge-ins from music and keyboard noise.
    vad_threshold: float = 0.6
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


def _check_min_int(value: object, minimum: int, path: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        _fail(path, f"expected an integer >= {minimum}, got {value!r}")


def _check_positive_number(value: object, path: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
        _fail(path, f"expected a number > 0, got {value!r}")


def _check_str(value: object, path: str, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        _fail(path, f"expected a non-empty string, got {value!r}")


def _check_bool(value: object, path: str) -> None:
    if not isinstance(value, bool):
        _fail(path, f"expected true or false, got {value!r}")


def _check_model_pin(value: object, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        _fail(path, f"expected a non-empty string, got {value!r}")
    provider, separator, model_id = value.partition("/")
    if separator != "/" or not provider.strip() or not model_id.strip() or "/" in model_id:
        _fail(path, f"expected provider/model-id, got {value!r}")


def _warn_phonetically_risky_names(names: list[str]) -> None:
    """Warn (never fail) on spoken names likely to confuse STT: the plan's
    mitigation is phonetically distinct, multi-syllable names."""
    import difflib
    import logging
    import re

    logger = logging.getLogger("earshot.config")
    for name in names:
        if len(re.findall(r"[aeiouy]+", name.lower())) <= 1:
            logger.warning(
                "agent name %r is short for speech recognition; "
                "multi-syllable names transcribe more reliably",
                name,
            )
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= 0.75:
                logger.warning(
                    "agent names %r and %r sound alike and may be confused "
                    "by speech recognition; pick more distinct names",
                    a,
                    b,
                )


def validate(config: Config) -> Config:
    """Validate every field; raise ConfigError with the key path on failure."""
    _check_str(config.wake_word.phrase, "wake_word.phrase")
    _check_str(config.wake_word.model_path, "wake_word.model_path", optional=True)
    _check_range(config.wake_word.sensitivity, 0.0, 1.0, "wake_word.sensitivity")
    _check_min_int(config.wake_word.patience, 1, "wake_word.patience")

    _check_enum(config.stt.backend, BACKENDS, "stt.backend")
    _check_str(config.stt.local.model, "stt.local.model")
    _check_str(config.stt.local.device, "stt.local.device")
    _check_str(config.stt.local.compute_type, "stt.local.compute_type")
    _check_str(config.stt.api.base_url, "stt.api.base_url")
    _check_str(config.stt.api.api_key_env, "stt.api.api_key_env")
    _check_str(config.stt.api.model, "stt.api.model")
    _check_bool(config.stt.api.fallback_to_local, "stt.api.fallback_to_local")

    _check_enum(config.tts.backend, BACKENDS, "tts.backend")
    _check_enum(config.tts.local.engine, TTS_ENGINES, "tts.local.engine")
    _check_str(config.tts.local.voice, "tts.local.voice")
    _check_positive_number(config.tts.local.speed, "tts.local.speed")
    _check_str(config.tts.api.base_url, "tts.api.base_url")
    _check_str(config.tts.api.api_key_env, "tts.api.api_key_env")
    _check_str(config.tts.api.model, "tts.api.model")
    _check_str(config.tts.api.voice, "tts.api.voice")
    _check_bool(config.tts.api.fallback_to_local, "tts.api.fallback_to_local")

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
        _check_bool(agent.restart_on_death, f"agents.{name}.restart_on_death")
        _check_str(agent.tmux_pane, f"agents.{name}.tmux_pane", optional=True)
    if len(config.agents) > 1:
        # Spoken names only reach STT with two or more agents (a single
        # agent gets verbatim pass-through routing), so a lone default
        # agent does not nag about its own name.
        _warn_phonetically_risky_names(list(config.agents))

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
        data = yaml.load(raw, Loader=_StrictLoader)
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
  sensitivity: 0.9          # detection threshold, 0..1 (tuned; see docs/tuning-protocol.md)
  patience: 3               # consecutive windows above threshold to fire

stt:
  backend: local            # local or api
  local:
    model: base.en          # faster-whisper model size
    device: cpu
    compute_type: int8
  api:
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY   # name of the env var holding the key; never a literal key
    model: whisper-1
    fallback_to_local: false      # degrade to the local backend on API failure

tts:
  backend: local            # local or api
  local:
    engine: piper           # piper works today; kokoro is reserved
    voice: en_US-lessac-medium
    speed: 1.0
  api:
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY   # name of the env var holding the key; never a literal key
    model: tts-1
    voice: alloy
    fallback_to_local: false      # degrade to the local backend on API failure

code_blocks: summarize      # fenced code blocks: summarize | skip | read

# Spoken agent name -> how to reach that agent.
agents:
  main:
    harness: opencode       # opencode | claude-code | codex
    command: null           # claude --print, codex app-server; tmux fallback launches it
    workdir: "~"
    model: null             # null = the harness's default (opencode pins {model})
    restart_on_death: true  # fleet supervisor restarts this agent if it dies
    tmux_pane: null         # non-null session name selects the tmux fallback transport

barge_in:
  vad_threshold: 0.6        # 0..1 speech probability (tuned; docs/tuning-protocol.md)
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
