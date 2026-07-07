"""Spoken-name extraction from noisy transcripts.

The convention is leading-name addressing: "<name>, <command>". Names are
matched fuzzily against the small closed set of configured agent names
(the plan's mitigation for STT noise), using normalized similarity plus a
consonant-skeleton comparison that survives vowel-level mishearings
("marvin" vs "marvun").

Thresholds are corpus-driven (tests/test_addressing.py): a match at or
above ROUTE_THRESHOLD routes immediately; between CLARIFY_THRESHOLD and
ROUTE_THRESHOLD the caller should ask aloud instead of silently
misrouting; below that the utterance is treated as unaddressed.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

ROUTE_THRESHOLD = 0.80
CLARIFY_THRESHOLD = 0.60


@dataclass
class Address:
    name: str | None  # the matched agent name (best candidate if ambiguous)
    command: str  # the utterance with any matched name prefix removed
    confidence: float  # 0..1 similarity of the name match


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _consonants(text: str) -> str:
    return re.sub(r"[aeiouy ]", "", _normalize(text))


def _similarity(candidate: str, name: str) -> float:
    text_score = difflib.SequenceMatcher(None, _normalize(candidate), _normalize(name)).ratio()
    skeleton_score = difflib.SequenceMatcher(
        None, _consonants(candidate), _consonants(name)
    ).ratio()
    return 0.6 * text_score + 0.4 * skeleton_score


def extract_address(utterance: str, names: list[str]) -> Address:
    """Split an utterance into (addressed agent, command, confidence).

    Tries the leading 1..N words (N sized by each name's own word count)
    against every configured name and keeps the best score.
    """
    words = utterance.split()
    if not words or not names:
        return Address(None, utterance.strip(), 0.0)

    best_name = None
    best_score = 0.0
    best_consumed = 0
    for name in names:
        name_words = len(name.split())
        for consumed in (name_words, name_words + 1):
            candidate = " ".join(words[:consumed])
            if not candidate:
                continue
            score = _similarity(candidate, name)
            if score > best_score:
                best_name, best_score, best_consumed = name, score, consumed

    if best_score < CLARIFY_THRESHOLD:
        return Address(None, utterance.strip(), best_score)
    command = " ".join(words[best_consumed:]).lstrip(",.;:").strip()
    return Address(best_name, command, best_score)
