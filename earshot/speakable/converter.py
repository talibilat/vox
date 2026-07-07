"""Markdown to natural speakable text.

The hard rule: no raw markdown syntax may ever be voiced. "hash hash bold
star star" is the canonical failure. Conversion walks markdown-it-py's token
stream and emits plain sentences; code blocks follow the configured mode
(summarize / skip / read, default summarize).
"""

from __future__ import annotations

from markdown_it import MarkdownIt
from markdown_it.token import Token

_parser = MarkdownIt("commonmark", {"breaks": False}).enable("table")


def to_speakable(markdown: str, code_blocks: str = "summarize") -> str:
    """Convert markdown to plain speakable text.

    code_blocks: "summarize" -> one descriptive sentence per block;
    "skip" -> blocks are omitted entirely; "read" -> code text is read out.
    """
    tokens = _parser.parse(markdown)
    parts = _render_tokens(tokens, code_blocks)
    text = " ".join(part for part in parts if part.strip())
    return " ".join(text.split())


def _sentence(text: str) -> str:
    """Give a fragment terminal punctuation so TTS pauses naturally."""
    text = text.strip()
    if text and text[-1] not in ".!?:;":
        text += "."
    return text


def _summarize_code(content: str, info: str) -> str:
    lines = [line for line in content.splitlines() if line.strip()]
    language = info.strip().split()[0] if info.strip() else ""
    label = f"{language} code block" if language else "code block"
    count = len(lines)
    line_word = "line" if count == 1 else "lines"
    return f"A {count} {line_word} {label}."


def _render_inline(token: Token) -> str:
    """Flatten an inline token: keep text, drop markup, voice link text only."""
    out: list[str] = []
    for child in token.children or []:
        if child.type == "text":
            out.append(child.content)
        elif child.type == "code_inline":
            out.append(child.content)
        elif child.type in ("softbreak", "hardbreak"):
            out.append(" ")
        elif child.type == "image":
            alt = child.content or "an image"
            out.append(alt)
        # link_open/link_close, strong/em markers: drop, their text children
        # already flow through as plain "text" tokens.
    return "".join(out)


def _render_tokens(tokens: list[Token], code_blocks: str) -> list[str]:
    parts: list[str] = []
    ordered_counters: list[int] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.type in ("fence", "code_block"):
            if code_blocks == "read":
                parts.append(_sentence(" ".join(token.content.split())))
            elif code_blocks == "summarize":
                parts.append(_summarize_code(token.content, getattr(token, "info", "")))
            # skip: nothing at all
        elif token.type == "inline":
            text = _render_inline(token)
            if text.strip():
                parts.append(text.strip())
        elif token.type == "heading_close":
            # A heading is a sentence of its own; the inline before this
            # close already emitted its text.
            if parts:
                parts[-1] = _sentence(parts[-1])
        elif token.type == "paragraph_close":
            if parts:
                parts[-1] = _sentence(parts[-1])
        elif token.type == "ordered_list_open":
            ordered_counters.append(int(token.attrGet("start") or 1))
        elif token.type == "ordered_list_close":
            ordered_counters.pop()
        elif token.type == "list_item_open" and ordered_counters:
            parts.append(f"{ordered_counters[-1]}.")
            ordered_counters[-1] += 1
        elif token.type == "table_close":
            pass
        elif token.type == "tr_close":
            if parts:
                parts[-1] = _sentence(parts[-1])
        elif token.type in ("th_close", "td_close"):
            # Separate cells with commas so rows read as natural lists.
            if parts and parts[-1] and parts[-1][-1] not in ".!?,":
                parts[-1] = parts[-1] + ","
        i += 1
    # Merge cell fragments: the loop above appends per-inline; join handled
    # by caller, so strip trailing commas at sentence ends.
    return parts
