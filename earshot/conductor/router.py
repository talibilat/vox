"""Utterance routing: who was addressed, and what should happen.

The Router is the multi-agent transcript handler (the single-agent case is
just a one-name fleet, so the daemon always uses it). Classification order:

1. fleet command  reserved phrases like "agent status" are NEVER sent to an
                  agent as prompt text; #13 owns what status actually says.
2. read request   "<name>, what's your response" style utterances go to the
                  output layer seam (#13 implements it; the default politely
                  declines).
3. agent command  a confidently matched leading name routes the command and
                  makes that agent active.
4. clarification  an ambiguous name match asks aloud and holds the command;
                  a spoken "yes" routes it, anything else replaces it.
5. follow-up      no name at all goes to the active agent.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

from earshot.conductor.addressing import ROUTE_THRESHOLD, extract_address
from earshot.conductor.lifecycle import Fleet
from earshot.loop import ConversationLoop
from earshot.output import OutputPipeline

logger = logging.getLogger("earshot.router")

FLEET_PHRASES = (
    "agent status",
    "agents status",
    "status report",
    "fleet status",
    "status update",
)
READ_PATTERNS = (
    re.compile(r"\bwhat('?s| is| was)? (your|the) (response|answer|output)\b", re.I),
    re.compile(r"\bwhat did you (say|answer|reply)\b", re.I),
    re.compile(r"\bread (that|it|your (response|answer|output))( again)?\b", re.I),
)
CONFIRMATIONS = ("yes", "yes please", "yeah", "yep", "correct", "right", "do it")


class Router:
    def __init__(
        self,
        fleet: Fleet,
        output: OutputPipeline,
        read_response: Callable[[str], str] | None = None,
        fleet_status: Callable[[], str] | None = None,
        dispatch: Callable[[str, str], None] | None = None,
    ):
        self._fleet = fleet
        self._output = output
        self._read_response = read_response
        self._fleet_status = fleet_status
        # With an external dispatcher (the watcher pool), turns run on the
        # watchers' threads and the fleet supervisor owns ALL restarts, so
        # the active-agent exemption only applies in ConversationLoop mode.
        self._dispatch = dispatch
        self._exempt_active = dispatch is None
        self._loops: dict[str, ConversationLoop] = {}
        names = fleet.registry.names()
        self._active = names[0] if names else None
        self._pending: tuple[str, str] | None = None  # (candidate name, held command)
        if self._active and self._exempt_active:
            fleet.set_active(self._active)

    @property
    def active_agent(self) -> str | None:
        return self._active

    def handle_transcript(self, text: str) -> None:
        """Route one utterance. Called from the voice loop's thread."""
        utterance = text.strip()
        if not utterance:
            return
        if self._is_fleet_command(utterance):
            self.say(self._status_line())
            return
        if self._pending is not None and self._resolve_pending(utterance):
            return
        names = self._fleet.registry.names()
        if len(names) == 1:
            self._route(names[0], utterance)
            return
        address = extract_address(utterance, names)
        if address.name and address.confidence >= ROUTE_THRESHOLD:
            if not address.command:
                self._switch_active(address.name)
                self.say(f"{address.name} is listening.")
                return
            if self._is_read_request(address.command):
                self._speak_response_of(address.name)
                return
            self._route(address.name, address.command)
        elif address.name:  # ambiguous: clarify aloud, never silently misroute
            self._pending = (address.name, address.command)
            logger.info("ambiguous address (%.2f) for %r; asking", address.confidence, address.name)
            self.say(f"Did you mean {address.name}?")
        else:
            if self._is_read_request(utterance):
                self._speak_response_of(self._active)
                return
            if self._active is None:
                self.say("No agents are configured.")
                return
            self._route(self._active, utterance)

    def say(self, sentence: str) -> None:
        """Speak a status sentence; never raises into the voice loop."""
        try:
            self._output.speak(sentence)
            self._output.wait_until_idle()
        except Exception:
            logger.exception("could not speak router message")

    # --- internals -----------------------------------------------------

    def _resolve_pending(self, utterance: str) -> bool:
        name, command = self._pending
        self._pending = None
        if utterance.lower().strip(" .,!") in CONFIRMATIONS:
            if command:
                self._route(name, command)
            else:
                self._switch_active(name)
                self.say(f"{name} is listening.")
            return True
        return False  # not a confirmation: fall through and route normally

    def _route(self, name: str, command: str) -> None:
        try:
            record = self._fleet.get(name)
        except KeyError:
            self.say(f"I do not know an agent called {name}.")
            return
        if record.status == "dead":
            self.say(f"{name} is not running.")
            return
        self._switch_active(name)
        logger.info("routing to %s: %s", name, command)
        if self._dispatch is not None:
            self._dispatch(name, command)
            return
        loop = self._loops.get(name)
        if loop is None:
            # Turn-triggered recovery goes through the fleet so the registry
            # status stays consistent with reality.
            loop = ConversationLoop(
                record.adapter, self._output, restart=lambda n=name: self._fleet.restart(n)
            )
            self._loops[name] = loop
        loop.handle_transcript(command)

    def _switch_active(self, name: str) -> None:
        if name != self._active:
            logger.info("active agent: %s -> %s", self._active, name)
        self._active = name
        if self._exempt_active:
            self._fleet.set_active(name)

    def _speak_response_of(self, name: str | None) -> None:
        if name is None:
            self.say("No agents are configured.")
            return
        if self._read_response is None:
            self.say("I cannot read responses back yet.")
            return
        self.say(self._read_response(name))

    def _status_line(self) -> str:
        if self._fleet_status is not None:
            return self._fleet_status()
        statuses = self._fleet.statuses()
        if not statuses:
            return "No agents are configured."
        parts = [f"{name} is {status}" for name, status in statuses.items()]
        return ", ".join(parts) + "."

    @staticmethod
    def _is_fleet_command(utterance: str) -> bool:
        cleaned = re.sub(r"[^a-z ]", "", utterance.lower()).strip()
        return any(
            cleaned == phrase or cleaned.startswith(phrase + " ") or cleaned.endswith(" " + phrase)
            for phrase in FLEET_PHRASES
        )

    @staticmethod
    def _is_read_request(text: str) -> bool:
        return any(pattern.search(text) for pattern in READ_PATTERNS)
