"""Config loading and validation tests."""

import pytest
import yaml

from earshot import config as config_module
from earshot.config import Config, ConfigError, load, validate, write_default


def write(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_defaults_are_valid():
    validate(Config())


def test_empty_file_loads_defaults(tmp_path):
    config = load(write(tmp_path, ""))
    assert config.wake_word.phrase == "hey earshot"
    assert config.stt.backend == "local"
    assert config.tts.local.engine == "piper"
    assert config.code_blocks == "summarize"
    assert list(config.agents) == ["main"]


def test_generated_default_file_loads(tmp_path):
    path = tmp_path / "generated.yaml"
    write_default(path)
    config = load(path)
    assert config.wake_word.sensitivity == 0.95
    assert config.daemon.log_file.startswith("~")


def test_missing_default_config_is_created(tmp_path, monkeypatch):
    default = tmp_path / "nested" / "config.yaml"
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", default)
    config = load(None)
    assert default.exists()
    assert config.wake_word.phrase == "hey earshot"


def test_explicit_missing_path_errors(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load(tmp_path / "nope.yaml")


def test_overrides_apply(tmp_path):
    config = load(
        write(
            tmp_path,
            """
            wake_word: {phrase: hey vox, sensitivity: 0.8}
            stt: {backend: api, api: {model: whisper-large}}
            tts: {local: {engine: kokoro, speed: 1.2}}
            code_blocks: read
            agents:
              six: {harness: codex, workdir: /tmp}
              seven: {harness: claude-code, tmux_pane: "%3"}
            """,
        )
    )
    assert config.wake_word.phrase == "hey vox"
    assert config.stt.backend == "api"
    assert config.stt.api.model == "whisper-large"
    assert config.stt.api.api_key_env == "OPENAI_API_KEY"  # untouched default
    assert config.tts.local.engine == "kokoro"
    assert config.tts.local.speed == 1.2
    assert config.code_blocks == "read"
    assert config.agents["six"].harness == "codex"
    assert config.agents["seven"].tmux_pane == "%3"
    assert config.agents["seven"].workdir == "~"


@pytest.mark.parametrize(
    ("text", "path_in_error"),
    [
        ("wake_word: {phrase: ''}", "wake_word.phrase"),
        ("wake_word: {sensitivity: 1.5}", "wake_word.sensitivity"),
        ("wake_word: {patience: 0}", "wake_word.patience"),
        ("stt: {backend: remote}", "stt.backend"),
        ("tts: {local: {engine: espeak}}", "tts.local.engine"),
        ("tts: {local: {speed: 0}}", "tts.local.speed"),
        ("code_blocks: mumble", "code_blocks"),
        ("agents: {}", "agents"),
        ("agents: {main: {harness: cursor}}", "agents.main.harness"),
        ("barge_in: {vad_threshold: -0.1}", "barge_in.vad_threshold"),
    ],
)
def test_invalid_values_name_the_key(tmp_path, text, path_in_error):
    with pytest.raises(ConfigError, match=path_in_error.replace(".", r"\.")):
        load(write(tmp_path, text))


def test_unknown_top_level_key(tmp_path):
    with pytest.raises(ConfigError, match="wake_words: unknown key"):
        load(write(tmp_path, "wake_words: {phrase: hi}"))


def test_unknown_nested_key_lists_valid_keys(tmp_path):
    with pytest.raises(ConfigError, match=r"stt\.modle: unknown key.*backend"):
        load(write(tmp_path, "stt: {modle: base.en}"))


def test_unknown_agent_key(tmp_path):
    with pytest.raises(ConfigError, match=r"agents\.main\.pane: unknown key"):
        load(write(tmp_path, "agents: {main: {pane: '%1'}}"))


def test_invalid_yaml_reports_file(tmp_path):
    with pytest.raises(ConfigError, match="not valid YAML"):
        load(write(tmp_path, "wake_word: [unclosed"))


def test_non_mapping_top_level(tmp_path):
    with pytest.raises(ConfigError, match="top level must be a mapping"):
        load(write(tmp_path, "- just\n- a list\n"))


def test_non_mapping_section(tmp_path):
    with pytest.raises(ConfigError, match="stt: expected a mapping"):
        load(write(tmp_path, "stt: fast"))


def test_example_config_matches_schema():
    """The committed example must always load cleanly."""
    from pathlib import Path

    example = Path(__file__).resolve().parent.parent / "config.example.yaml"
    data = yaml.safe_load(example.read_text())
    config = validate(config_module._from_dict(data))
    assert config.wake_word.phrase == "hey earshot"
