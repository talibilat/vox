"""The single-agent conversation loop: transcript in, spoken response out.

This is the MVP moment: InputPipeline's on_transcript hook feeds spoken
instructions here; the adapter streams the agent's markdown response; the
OutputPipeline speaks it as it arrives. Follow-up transcripts continue the
same agent session, so multi-turn persistence is the adapter's session's.

Failure policy: every realistic failure produces spoken feedback instead of
silence. If the agent process died, one automatic restart is attempted; a
restart loses the session, which is also said out loud.
"""

from __future__ import annotations

import logging

from earshot.agents import AgentAdapter, AgentError
from earshot.output import OutputPipeline

logger = logging.getLogger("earshot.loop")


class ConversationLoop:
    def __init__(self, adapter: AgentAdapter, output: OutputPipeline):
        self._adapter = adapter
        self._output = output

    def handle_transcript(self, text: str) -> None:
        """Run one voice turn. Called from the input pipeline's thread."""
        logger.info("you said: %s", text)
        try:
            self._speak_response(text)
        except AgentError as error:
            logger.warning("agent turn failed: %s", error)
            self._recover(text, error)

    def _speak_response(self, text: str) -> None:
        self._output.speak_stream(self._adapter.send(text))
        self._output.wait_until_idle()

    def _recover(self, text: str, error: AgentError) -> None:
        if self._adapter.alive:
            self._say("The agent is not responding. Please try again.")
            return
        self._say("The agent stopped. Restarting it now.")
        try:
            self._adapter.stop()
            self._adapter.start()
        except AgentError as restart_error:
            logger.error("agent restart failed: %s", restart_error)
            self._say("I could not restart the agent. Check the logs.")
            return
        self._say("The agent is back, in a fresh session. Repeating your request.")
        try:
            self._speak_response(text)
        except AgentError as retry_error:
            logger.error("retry after restart failed: %s", retry_error)
            self._say("The agent failed again. Giving up on this request.")

    def _say(self, sentence: str) -> None:
        """Speak a status sentence, never raising into the input thread."""
        try:
            self._output.speak(sentence)
            self._output.wait_until_idle()
        except Exception:
            logger.exception("could not speak status message")
