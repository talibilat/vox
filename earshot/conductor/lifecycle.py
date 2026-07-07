"""Fleet lifecycle: the Conductor spawns, supervises, and stops every agent.

Per the plan's locked decision 5, nothing is ever started by hand: on
daemon start the fleet spawns all configured agents through their adapters
(staggered so sixteen spawns do not stampede the machine), a supervisor
thread watches process liveness, and shutdown terminates everything.

Restart ownership is split to avoid double-restarts: the supervisor
restarts a dead agent only when it is NOT the active conversation agent
(the ConversationLoop already owns turn-triggered recovery for that one,
with spoken feedback).
"""

from __future__ import annotations

import logging
import threading
import time

from earshot import agents as agents_module
from earshot.conductor.registry import AgentRecord, Registry
from earshot.config import Config

logger = logging.getLogger("earshot.conductor")

STAGGER_SECONDS = 0.5  # delay between spawns during fleet startup
SUPERVISION_INTERVAL = 2.0


class Fleet:
    def __init__(self, config: Config, stagger_seconds: float = STAGGER_SECONDS):
        self._config = config
        self._stagger = stagger_seconds
        self.registry = Registry()
        self._stop = threading.Event()
        self._supervisor: threading.Thread | None = None
        self._active_name: str | None = None
        for name, agent_config in config.agents.items():
            adapter = agents_module.create_adapter(name, agent_config)
            self.registry.add(AgentRecord(name=name, config=agent_config, adapter=adapter))

    # --- startup / shutdown ---------------------------------------------

    def start_all(self) -> None:
        """Spawn every configured agent, staggered. Individual failures mark
        that agent dead and do not abort the rest of the fleet."""
        started = time.perf_counter()
        for index, record in enumerate(self.registry.records()):
            if index:
                time.sleep(self._stagger)
            self._start_record(record)
        up = [r.name for r in self.registry.records() if r.status != "dead"]
        dead = [r.name for r in self.registry.records() if r.status == "dead"]
        logger.info(
            "fleet up in %.1fs: %d/%d agents (%s)%s",
            time.perf_counter() - started,
            len(up),
            len(self.registry.records()),
            ", ".join(up),
            f"; dead: {', '.join(dead)}" if dead else "",
        )

    def stop_all(self) -> None:
        self.stop_supervision()
        for record in self.registry.records():
            try:
                record.adapter.stop()
            except Exception:
                logger.exception("stopping agent %s failed", record.name)
            record.mark("dead")

    # --- lookups ----------------------------------------------------------

    def get(self, name: str) -> AgentRecord:
        return self.registry.get(name)

    def statuses(self) -> dict[str, str]:
        return self.registry.statuses()

    def restart(self, name: str) -> bool:
        return self._start_record(self.registry.get(name), restart=True)

    # --- supervision -------------------------------------------------------

    def set_active(self, name: str | None) -> None:
        """The active conversation agent is exempt from supervisor restarts
        (its recovery belongs to the conversation loop). The router calls
        this as addressing switches."""
        self._active_name = name

    def start_supervision(self, active_name: str | None = None) -> None:
        """Watch process liveness; restart dead agents per their config,
        except the active conversation agent (owned by the loop's recovery)."""
        self._active_name = active_name
        self._stop.clear()
        self._supervisor = threading.Thread(
            target=self._supervise, daemon=True, name="fleet-supervisor"
        )
        self._supervisor.start()

    def stop_supervision(self) -> None:
        self._stop.set()
        if self._supervisor is not None:
            self._supervisor.join(timeout=SUPERVISION_INTERVAL * 3)
            self._supervisor = None

    def _supervise(self) -> None:
        while not self._stop.wait(SUPERVISION_INTERVAL):
            for record in self.registry.records():
                if record.status == "dead" or record.adapter.alive:
                    continue
                logger.warning("agent %s died (was %s)", record.name, record.status)
                record.mark("dead")
                if record.config.restart_on_death and record.name != self._active_name:
                    self._start_record(record, restart=True)

    def _start_record(self, record: AgentRecord, restart: bool = False) -> bool:
        record.mark("starting")
        try:
            if restart:
                record.adapter.stop()
            record.adapter.start()
        except Exception:
            logger.exception(
                "%s agent %s failed", "restarting" if restart else "starting", record.name
            )
            record.mark("dead")
            return False
        record.mark("idle")
        if restart:
            logger.info("agent %s restarted in a fresh session", record.name)
        return True
