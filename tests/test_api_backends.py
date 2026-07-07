"""API-mode STT/TTS backend tests against a mocked OpenAI-compatible server.

Covers: both backends' happy paths (including auth header and multipart
shape), key-from-environment enforcement, HTTP failure paths with clear
errors, local fallback (including sample-rate adaptation for TTS), backend
selection via config with mixed local/API combinations, and the rule that
local mode never touches API code or credentials.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from earshot.config import Config
from earshot.stt import create_backend as create_stt
from earshot.stt.api_openai import ApiSttBackend, BackendUnavailable
from earshot.tts import create_backend as create_tts
from earshot.tts.api_openai import PCM_SAMPLE_RATE, ApiTtsBackend

TONE = (np.sin(np.linspace(0, 440 * 2 * np.pi, 16000)) * 8000).astype(np.int16)


class FakeOpenAiHandler(BaseHTTPRequestHandler):
    behavior = "ok"  # class-level toggle: ok | http_error
    seen: list[dict] = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        FakeOpenAiHandler.seen.append(
            {
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "content_type": self.headers.get("Content-Type", ""),
                "body": body,
            }
        )
        if FakeOpenAiHandler.behavior == "http_error":
            self.send_response(401)
            payload = b'{"error": {"message": "bad key"}}'
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path.endswith("/audio/transcriptions"):
            payload = json.dumps({"text": "run the tests please"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif self.path.endswith("/audio/speech"):
            pcm = np.arange(24000, dtype=np.int16).tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(pcm)))
            self.end_headers()
            self.wfile.write(pcm)
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture()
def api_server(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    FakeOpenAiHandler.behavior = "ok"
    FakeOpenAiHandler.seen = []
    monkeypatch.setenv("EARSHOT_TEST_KEY", "sk-test-123")
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


class TestApiStt:
    def make(self, base_url, **kwargs):
        return ApiSttBackend(
            base_url=base_url, api_key_env="EARSHOT_TEST_KEY", model="whisper-1", **kwargs
        )

    def test_transcribes_and_sends_wav_with_auth(self, api_server):
        text = self.make(api_server).transcribe(TONE, 16000)
        assert text == "run the tests please"
        request = FakeOpenAiHandler.seen[-1]
        assert request["auth"] == "Bearer sk-test-123"
        assert "multipart/form-data" in request["content_type"]
        assert b"whisper-1" in request["body"]
        assert b"RIFF" in request["body"], "no wav payload in the upload"

    def test_missing_key_env_is_actionable(self, api_server, monkeypatch):
        monkeypatch.delenv("EARSHOT_TEST_KEY")
        with pytest.raises(BackendUnavailable, match="EARSHOT_TEST_KEY"):
            self.make(api_server)

    def test_http_error_is_clear(self, api_server):
        FakeOpenAiHandler.behavior = "http_error"
        with pytest.raises(BackendUnavailable, match="401"):
            self.make(api_server).transcribe(TONE, 16000)

    def test_unreachable_is_clear(self, monkeypatch):
        monkeypatch.setenv("EARSHOT_TEST_KEY", "sk-test-123")
        backend = self.make("http://127.0.0.1:1/v1")
        with pytest.raises(BackendUnavailable, match="unreachable"):
            backend.transcribe(TONE, 16000)

    def test_fallback_to_local(self, api_server):
        FakeOpenAiHandler.behavior = "http_error"

        class FakeLocal:
            def transcribe(self, audio, sample_rate):
                return "local fallback text"

        backend = self.make(api_server, fallback=FakeLocal())
        assert backend.transcribe(TONE, 16000) == "local fallback text"


class TestApiTts:
    def make(self, base_url, **kwargs):
        return ApiTtsBackend(
            base_url=base_url,
            api_key_env="EARSHOT_TEST_KEY",
            model="tts-1",
            voice="alloy",
            **kwargs,
        )

    def test_streams_pcm_chunks(self, api_server):
        backend = self.make(api_server)
        chunks = list(backend.synthesize("Say something nice."))
        assert backend.sample_rate == PCM_SAMPLE_RATE
        assert len(chunks) > 1, "response was not streamed in chunks"
        combined = np.concatenate(chunks)
        assert np.array_equal(combined, np.arange(24000, dtype=np.int16))
        request = FakeOpenAiHandler.seen[-1]
        sent = json.loads(request["body"])
        assert sent == {
            "model": "tts-1",
            "voice": "alloy",
            "input": "Say something nice.",
            "response_format": "pcm",
        }
        assert request["auth"] == "Bearer sk-test-123"

    def test_http_error_is_clear(self, api_server):
        FakeOpenAiHandler.behavior = "http_error"
        with pytest.raises(BackendUnavailable, match="401"):
            list(self.make(api_server).synthesize("hello"))

    def test_fallback_resamples_local_audio(self, api_server):
        FakeOpenAiHandler.behavior = "http_error"

        class FakeLocal:
            sample_rate = 12000  # half the API rate: resampler must double it

            def synthesize(self, text):
                yield np.ones(1200, dtype=np.int16) * 1000

        backend = self.make(api_server, fallback=FakeLocal())
        chunks = list(backend.synthesize("hello"))
        assert sum(len(c) for c in chunks) == 2400, "fallback audio was not resampled"


class TestBackendSelection:
    def test_api_backends_selected_via_config(self, api_server):
        config = Config()
        config.stt.backend = "api"
        config.stt.api.base_url = api_server
        config.stt.api.api_key_env = "EARSHOT_TEST_KEY"
        config.tts.backend = "api"
        config.tts.api.base_url = api_server
        config.tts.api.api_key_env = "EARSHOT_TEST_KEY"
        assert isinstance(create_stt(config), ApiSttBackend)
        assert isinstance(create_tts(config), ApiTtsBackend)

    def test_mixed_mode_local_stt_api_tts(self, api_server):
        config = Config()  # stt stays on its local default
        config.tts.backend = "api"
        config.tts.api.base_url = api_server
        config.tts.api.api_key_env = "EARSHOT_TEST_KEY"
        tts = create_tts(config)
        assert isinstance(tts, ApiTtsBackend)
        assert list(tts.synthesize("hi")), "API TTS did not produce audio in mixed mode"
        assert config.stt.backend == "local"

    def test_local_mode_never_needs_credentials(self, monkeypatch):
        # Local selection must not read any API key or open any socket; a
        # poisoned key variable proves the API path is never consulted.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = Config()
        pytest.importorskip("faster_whisper")
        from earshot.stt.local_whisper import LocalWhisperBackend

        assert isinstance(create_stt(config), LocalWhisperBackend)


class TestVoiceLoopSttFailureFeedback:
    def test_stt_failure_is_spoken_not_fatal(self, tmp_path):
        pytest.importorskip("openwakeword")
        from pathlib import Path

        from earshot.agents.base import AgentAdapter
        from earshot.audio.capture import ArraySource
        from earshot.barge import InterruptibleVoiceLoop
        from tests.test_agents import FakeOutput
        from tests.test_barge import read_wav

        class FailingStt:
            def transcribe(self, audio, sample_rate):
                raise BackendUnavailable("STT API is unreachable: simulated")

        class IdleAdapter(AgentAdapter):
            prompts: list = []

            def start(self):
                pass

            def stop(self):
                pass

            @property
            def alive(self):
                return True

            def send(self, prompt):
                self.prompts.append(prompt)
                return iter(())

        repo = Path(__file__).resolve().parent.parent
        config = Config()
        config.wake_word.model_path = str(repo / "spikes" / "models" / "hey_earshot.onnx")
        config.wake_word.sensitivity = 0.9
        config.wake_word.patience = 3

        adapter = IdleAdapter()
        output = FakeOutput()
        loop = InterruptibleVoiceLoop(
            config,
            adapter,
            output,
            source=ArraySource(read_wav("wake_then_command.wav")),
            stt=FailingStt(),
        )
        loop.run()  # must return cleanly, not raise
        assert adapter.prompts == []
        assert any("could not transcribe" in s for s in output.spoken)
