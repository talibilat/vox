"""The agent registry: one runtime record per configured agent.

This is the internal API the rest of Phase 2 builds on: voice addressing
(#12) resolves spoken names against it, and output watching (#13) attaches
to the records it holds. Lifecycle status is process-level until watchers
drive turn state: starting/ready/idle/dead for lifecycle, busy while a turn
runs, and finished when an unread response is buffered.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from earshot.agents import AgentAdapter
from earshot.config import AgentConfig

STATUSES = ("starting", "ready", "busy", "idle", "finished", "dead")


@dataclass
class AgentRecord:
    name: str
    config: AgentConfig
    adapter: AgentAdapter
    status: str = "starting"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def mark(self, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown agent status {status!r}")
        with self._lock:
            self.status = status


class Registry:
    """Name-keyed lookup over the fleet's agent records."""

    def __init__(self) -> None:
        self._records: dict[str, AgentRecord] = {}

    def add(self, record: AgentRecord) -> None:
        if record.name in self._records:
            raise ValueError(f"agent {record.name!r} is already registered")
        self._records[record.name] = record

    def get(self, name: str) -> AgentRecord:
        try:
            return self._records[name]
        except KeyError:
            raise KeyError(f"no agent named {name!r} is configured") from None

    def names(self) -> list[str]:
        return list(self._records)

    def records(self) -> list[AgentRecord]:
        return list(self._records.values())

    def statuses(self) -> dict[str, str]:
        return {name: record.status for name, record in self._records.items()}
