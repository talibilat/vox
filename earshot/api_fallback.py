from __future__ import annotations

from collections.abc import Callable
from logging import Logger
from typing import TypeVar

T = TypeVar("T")


def get_or_create_fallback(
    fallback: T | None,
    fallback_factory: Callable[[], T] | None,
    logger: Logger,
    label: str,
) -> tuple[T | None, Callable[[], T] | None]:
    """Return the existing fallback or lazily create one.

    If creation fails, log the failure and clear the factory so future calls do
    not retry an unusable local fallback for every API request.
    """
    if fallback is not None or fallback_factory is None:
        return fallback, fallback_factory
    try:
        return fallback_factory(), fallback_factory
    except Exception:
        logger.exception("could not create the local %s fallback", label)
        return None, None
