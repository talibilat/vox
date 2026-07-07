"""Incremental sentence chunking for streaming TTS.

Text arrives token by token from the agent; the chunker yields each sentence
the moment it is complete so synthesis starts within a beat of the response
beginning rather than after it ends.
"""

from __future__ import annotations

import re

# A sentence ends at . ! or ? followed by whitespace or the current buffer end.
# A closing quote/paren may sit between the punctuation and the whitespace.
_BOUNDARY = re.compile(r'([.!?][)"\']?)(\s+|$)')

# Common abbreviations that end with a period but do not end a sentence.
_NON_TERMINAL = re.compile(r"(?:\b(?:e\.g|i\.e|etc|vs|Mr|Mrs|Ms|Dr|St|No)\.|\b[A-Za-z]\.|\d\.)$")


class SentenceChunker:
    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        """Add streamed text; return any sentences completed by it."""
        self._buffer += text
        sentences: list[str] = []
        while True:
            match = self._find_boundary(self._buffer)
            if match is None:
                break
            end = match.end(1)
            sentences.append(self._buffer[:end].strip())
            self._buffer = self._buffer[match.end() :]
        return sentences

    def flush(self) -> list[str]:
        """The stream is over; return whatever remains as a final chunk."""
        rest = self._buffer.strip()
        self._buffer = ""
        return [rest] if rest else []

    def _find_boundary(self, text: str) -> re.Match | None:
        for match in _BOUNDARY.finditer(text):
            head = text[: match.end(1)].rstrip()
            if _NON_TERMINAL.search(head):
                continue
            return match
        return None
