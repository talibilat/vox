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


def _status_part(names: list[str], singular_verb: str, plural_verb: str, phrase: str) -> str:
    verb = singular_verb if len(names) == 1 else plural_verb
    return f"{_name_list(names)} {verb} {phrase}"


def spoken_status(statuses: dict[str, str]) -> str:
    if not statuses:
        return "No agents are configured."
    groups: dict[str, list[str]] = {}
    for name, status in statuses.items():
        groups.setdefault(status, []).append(name)

    parts: list[str] = []
    if finished := groups.get("finished"):
        parts.append(_status_part(finished, "has", "have", "finished"))
    busy = groups.get("busy", []) + groups.get("starting", [])
    if busy:
        parts.append(_status_part(busy, "is", "are", "still working"))
    if idle := groups.get("idle", []) + groups.get("ready", []):
        parts.append(_status_part(idle, "is", "are", "idle"))
    if dead := groups.get("dead"):
        parts.append(_status_part(dead, "is", "are", "not running"))
    return "; ".join(parts) + "."
