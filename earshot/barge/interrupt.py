"""The interruptible voice loop: the barge-in state machine.

Supersedes the plain InputPipeline wiring in the daemon. One thread owns
the microphone and walks these states:

    LISTENING   wake-word watch; the wake phrase starts a recording.
    RECORDING   buffer mic frames until end-of-speech, then transcribe and
                dispatch the turn to the agent on a worker thread.
    RESPONDING  the agent is speaking. VAD watches the mic; sustained user
                speech (or the push-to-interrupt escape hatch) stops
                playback within the 200ms budget and jumps straight back to
                RECORDING with the onset audio pre-rolled in, so the
                interrupting utterance itself becomes the next command,
                no wake word needed. When playback finishes naturally, the
                loop returns to LISTENING.

Every interrupt logs onset-to-silence latency; `latencies_ms` keeps the
distribution so the 200ms target is verified by data, not vibes.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import numpy as np

from earshot.agents import AgentAdapter
from earshot.audio import SAMPLE_RATE
from earshot.audio.endpointing import EndOfSpeechDetector
from earshot.barge.vad import SpeechOnsetDetector
from earshot.config import Config
from earshot.loop import ConversationLoop
from earshot.output import OutputPipeline
from earshot.stt import SttBackend, create_backend
from earshot.wakeword import WakeWordDetector

logger = logging.getLogger("earshot.barge")

LISTENING, RECORDING, RESPONDING = "listening", "recording", "responding"


class InterruptibleVoiceLoop:
    def __init__(
        self,
        config: Config,
        handler,
        output: OutputPipeline,
        source=None,
        stt: SttBackend | None = None,
        restart: Callable[[], bool] | None = None,
    ):
        """`handler` consumes transcripts: anything with handle_transcript()
        and say() (a per-agent ConversationLoop, or the multi-agent Router).
        An AgentAdapter is also accepted and wrapped for convenience."""
        if not config.wake_word.model_path:
            raise ValueError("wake_word.model_path is not set; the voice loop needs a model")
        if source is None:
            from earshot.audio.capture import MicSource

            source = MicSource()
        self._source = source
        self._output = output
        if isinstance(handler, AgentAdapter):
            handler = ConversationLoop(handler, output, restart=restart)
        self._conversation = handler
        self._wake = WakeWordDetector(
            model_path=config.wake_word.model_path,
            sensitivity=config.wake_word.sensitivity,
            patience=config.wake_word.patience,
        )
        self._end_of_speech = EndOfSpeechDetector(threshold=config.barge_in.vad_threshold)
        self._onset = SpeechOnsetDetector(threshold=config.barge_in.vad_threshold)
        self._stt = stt if stt is not None else create_backend(config)
        self._stop = threading.Event()
        self._interrupt_requested = threading.Event()
        self._responder: threading.Thread | None = None
        self.latencies_ms: list[float] = []
        self.interrupts = 0

    def stop(self) -> None:
        self._stop.set()
        self._interrupt_requested.set()  # unblock a responding wait quickly
        stop_source = getattr(self._source, "stop", None)
        if callable(stop_source):
            stop_source()

    def request_interrupt(self) -> None:
        """The push-to-interrupt escape hatch (`earshot interrupt`)."""
        self._interrupt_requested.set()

    def run(self) -> None:
        """Blocking loop; call stop() from another thread to exit."""
        state = LISTENING
        buffer: list[np.ndarray] = []
        self._wake.reset()
        for frame in self._source.frames():
            if self._stop.is_set():
                break
            if state == LISTENING:
                state, buffer = self._continue_listening(frame, buffer)
            elif state == RECORDING:
                state, buffer = self._continue_recording(frame, buffer)
            elif state == RESPONDING:
                state, buffer = self._continue_responding(frame, buffer)
        self._flush_recording_if_needed(state, buffer)
        self._join_responder(timeout=30)

    # --- transitions -----------------------------------------------------

    def _enter_listening(self) -> str:
        self._wake.reset()
        return LISTENING

    def _enter_recording(self, pre_roll: list[np.ndarray]) -> tuple[str, list[np.ndarray]]:
        self._end_of_speech.reset()
        buffer = list(pre_roll)
        for frame in pre_roll:
            self._end_of_speech.finished(frame)  # prime with the onset audio
        return RECORDING, buffer

    def _enter_responding(self) -> str:
        self._onset.reset()
        self._interrupt_requested.clear()
        return RESPONDING

    def _continue_listening(
        self, frame: np.ndarray, buffer: list[np.ndarray]
    ) -> tuple[str, list[np.ndarray]]:
        if self._responding():
            # A response is still playing out (e.g. an error message after a
            # failed turn); treat it as RESPONDING.
            return self._enter_responding(), buffer
        if self._wake.detected(frame):
            logger.info("wake word detected, recording")
            return self._enter_recording([])
        return LISTENING, buffer

    def _continue_recording(
        self, frame: np.ndarray, buffer: list[np.ndarray]
    ) -> tuple[str, list[np.ndarray]]:
        buffer.append(frame)
        if self._end_of_speech.finished(frame):
            self._dispatch(np.concatenate(buffer))
            return self._enter_responding(), buffer
        return RECORDING, buffer

    def _continue_responding(
        self, frame: np.ndarray, buffer: list[np.ndarray]
    ) -> tuple[str, list[np.ndarray]]:
        interrupted = self._interrupt_requested.is_set() or self._onset.onset(frame)
        if interrupted:
            return self._enter_recording(self._interrupt_playback())
        if not self._responding():
            logger.info("response finished, back to listening")
            return self._enter_listening(), buffer
        return RESPONDING, buffer

    def _flush_recording_if_needed(self, state: str, buffer: list[np.ndarray]) -> None:
        # Source exhausted (tests/replay): flush a recording in progress.
        if state == RECORDING and buffer and not self._stop.is_set():
            self._dispatch(np.concatenate(buffer))

    def _responding(self) -> bool:
        worker_busy = self._responder is not None and self._responder.is_alive()
        return worker_busy or not self._output.wait_until_idle(timeout=0)

    def _dispatch(self, audio: np.ndarray) -> None:
        try:
            text = self._stt.transcribe(audio, SAMPLE_RATE)
        except Exception:
            # An unreachable API backend (or any STT failure) must produce
            # spoken feedback, not a dead daemon.
            logger.exception("transcription failed")
            self._start_responder(
                self._conversation.say,
                "I could not transcribe that. Check the logs.",
            )
            return
        if not text:
            logger.info("nothing transcribed, back to listening")
            return
        self._start_responder(self._conversation.handle_transcript, text)

    def _start_responder(self, target: Callable[[str], object], text: str) -> None:
        self._join_responder(timeout=30)
        self._responder = threading.Thread(
            target=target,
            args=(text,),
            daemon=True,
            name="responder",
        )
        self._responder.start()

    def _interrupt_playback(self) -> list[np.ndarray]:
        triggered_by = "hotkey" if self._interrupt_requested.is_set() else "voice"
        started = time.perf_counter()
        self._output.cancel_current()
        self._output.stop_and_flush()
        latency_ms = (time.perf_counter() - started) * 1000
        self.interrupts += 1
        self.latencies_ms.append(latency_ms)
        logger.info(
            "barge-in (%s): audio stopped in %.0fms (interrupt #%d)",
            triggered_by,
            latency_ms,
            self.interrupts,
        )
        return self._onset.pre_roll() if triggered_by == "voice" else []

    def _join_responder(self, timeout: float) -> None:
        if self._responder is not None and self._responder.is_alive():
            self._responder.join(timeout=timeout)
