"""The spoken fleet status: an accurate, natural-sounding roll-call.

"marvin and olivia have finished; sebastian is still working." Groups are
ordered by how actionable they are (finished first, then busy, idle, dead)
and empty groups are simply not mentioned.
"""

from __future__ import annotations


def _name_list(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def spoken_status(statuses: dict[str, str]) -> str:
    if not statuses:
        return "No agents are configured."
    groups: dict[str, list[str]] = {}
    for name, status in statuses.items():
        groups.setdefault(status, []).append(name)

    parts: list[str] = []
    if finished := groups.get("finished"):
        verb = "has" if len(finished) == 1 else "have"
        parts.append(f"{_name_list(finished)} {verb} finished")
    busy = groups.get("busy", []) + groups.get("starting", [])
    if busy:
        verb = "is" if len(busy) == 1 else "are"
        parts.append(f"{_name_list(busy)} {verb} still working")
    if idle := groups.get("idle", []) + groups.get("ready", []):
        verb = "is" if len(idle) == 1 else "are"
        parts.append(f"{_name_list(idle)} {verb} idle")
    if dead := groups.get("dead"):
        verb = "is" if len(dead) == 1 else "are"
        parts.append(f"{_name_list(dead)} {verb} not running")
    return "; ".join(parts) + "."
