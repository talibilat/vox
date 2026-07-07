"""The speech output pipeline: streamed markdown -> speakable text ->
sentence chunks -> TTS -> playback.

First audio must start on the first complete sentence of a response, so
paragraph text streams through the sentence chunker as it arrives and each
sentence is converted and synthesized on its own. Only constructs that
cannot be interpreted until complete are buffered: fenced code blocks
(the summary needs the whole block) and tables (a row is meaningless until
the header separator has been seen).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from earshot.config import Config
from earshot.speakable import SentenceChunker, to_speakable
from earshot.tts import TtsBackend, create_backend

logger = logging.getLogger("earshot.output")

_FENCE = re.compile(r"^\s*(```|~~~)")
_TABLE_LINE = re.compile(r"^\s*\|")


def _inline_balanced(text: str) -> bool:
    """True when the text has no dangling inline markup, so converting it in
    isolation cannot voice a stray marker."""
    if text.count("`") % 2:
        return False
    without_strong = text.replace("**", "").replace("__", "")
    if without_strong.count("*") % 2 or without_strong.count("_") % 2:
        return False
    return text.count("[") == text.count("]")


class _StreamConverter:
    """Turns a markdown character stream into speakable sentences.

    Paragraph and list text flows sentence-by-sentence; fences and tables
    buffer until their closing line, then convert as one block.
    """

    def __init__(self, code_blocks: str):
        self._code_blocks = code_blocks
        self._partial = ""  # current incomplete line
        self._fed = 0  # how much of the partial line already reached the chunker
        self._fence: list[str] | None = None  # open fence lines
        self._table: list[str] | None = None  # open table lines
        self._chunker = SentenceChunker()
        self._held = ""  # sentences held back until inline markup balances

    def feed(self, text: str) -> list[str]:
        out: list[str] = []
        self._partial += text
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            out.extend(self._line(line))
        # Stream the incomplete line too, so a sentence never waits for a
        # newline. A line that might still become a fence or table row is
        # held until it completes; feeding it early could voice its markers.
        if (
            self._fence is None
            and self._table is None
            and len(self._partial) > self._fed
            and not self._maybe_structural(self._partial)
        ):
            out.extend(self._stream_text(self._partial[self._fed :]))
            self._fed = len(self._partial)
        return out

    def finish(self) -> list[str]:
        out: list[str] = []
        if self._partial:
            out.extend(self._line(self._partial))
            self._partial = ""
        if self._fence is not None:  # unterminated fence: treat as closed
            out.extend(self._close_fence())
        if self._table is not None:
            out.extend(self._close_table())
        remainder = " ".join([self._held, *self._chunker.flush()]).strip()
        self._held = ""
        if remainder:
            out.extend(self._convert(remainder))
        return out

    @staticmethod
    def _maybe_structural(partial: str) -> bool:
        head = partial.lstrip()
        return not head or head[0] in "`~|"

    def _line(self, line: str) -> list[str]:
        unfed = line[self._fed :]
        self._fed = 0
        if self._fence is not None:
            self._fence.append(line)
            if _FENCE.match(line):
                return self._close_fence()
            return []
        if _FENCE.match(line):
            out = self._end_text_run()
            self._fence = [line]
            return out
        if _TABLE_LINE.match(line):
            if self._table is None:
                out = self._end_text_run()
                self._table = [line]
                return out
            self._table.append(line)
            return []
        if self._table is not None:
            out = self._close_table()
            out.extend(self._line(line))
            return out
        if not line.strip():
            return self._end_text_run()
        return self._stream_text(unfed + "\n")

    def _stream_text(self, text: str) -> list[str]:
        out: list[str] = []
        for sentence in self._chunker.feed(text):
            self._held = f"{self._held} {sentence}".strip() if self._held else sentence
            if _inline_balanced(self._held):
                out.extend(self._convert(self._held))
                self._held = ""
        return out

    def _end_text_run(self) -> list[str]:
        """A blank line or structural switch ends the open text run."""
        out: list[str] = []
        remainder = " ".join([self._held, *self._chunker.flush()]).strip()
        self._held = ""
        if remainder:
            out.extend(self._convert(remainder))
        return out

    def _close_fence(self) -> list[str]:
        block = "\n".join(self._fence or [])
        self._fence = None
        return self._convert(block)

    def _close_table(self) -> list[str]:
        block = "\n".join(self._table or [])
        self._table = None
        return self._convert(block)

    def _convert(self, markdown: str) -> list[str]:
        speakable = to_speakable(markdown, code_blocks=self._code_blocks)
        return [speakable] if speakable else []


class OutputPipeline:
    def __init__(self, config: Config, player=None, tts: TtsBackend | None = None):
        self._code_blocks = config.code_blocks
        self._tts = tts if tts is not None else create_backend(config)
        if player is None:
            from earshot.audio.playback import Player, SounddeviceSink

            player = Player(SounddeviceSink(self._tts.sample_rate))
        self._player = player

    @property
    def player(self):
        """The playback layer; barge-in (#7) calls player.stop_and_flush()."""
        return self._player

    def speak_stream(self, markdown_stream: Iterable[str]) -> None:
        """Speak a streamed markdown response as it arrives."""
        converter = _StreamConverter(self._code_blocks)
        for text in markdown_stream:
            for sentence in converter.feed(text):
                self._synthesize(sentence)
        for sentence in converter.finish():
            self._synthesize(sentence)

    def speak(self, markdown: str) -> None:
        self.speak_stream([markdown])

    def stop_and_flush(self) -> None:
        self._player.stop_and_flush()

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        return self._player.wait_until_idle(timeout=timeout)

    def _synthesize(self, sentence: str) -> None:
        logger.debug("speaking: %s", sentence)
        self._player.enqueue(self._tts.synthesize(sentence))
