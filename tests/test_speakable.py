"""Markdown-to-speakable converter and sentence chunker tests.

The converter's hard rule: no raw markdown syntax survives into speech.
"""

import pytest

from earshot.speakable import SentenceChunker, to_speakable

FORBIDDEN = ["#", "*", "`", "](", "|", "_", "~~"]


def assert_speakable(text: str):
    for symbol in FORBIDDEN:
        assert symbol not in text, f"markdown artifact {symbol!r} voiced in {text!r}"


class TestConverter:
    def test_headings(self):
        out = to_speakable("## Results\n\nAll tests pass.")
        assert_speakable(out)
        assert "Results" in out
        assert "All tests pass." in out

    def test_bold_italic(self):
        out = to_speakable("This is **very** important and _quite_ subtle.")
        assert_speakable(out)
        assert "very important" in out
        assert "quite subtle" in out

    def test_bullets(self):
        out = to_speakable("- first thing\n- second thing\n- third thing")
        assert_speakable(out)
        assert "first thing" in out
        assert "second thing" in out

    def test_numbered_list_keeps_numbers(self):
        out = to_speakable("1. clone the repo\n2. run the tests")
        assert_speakable(out)
        assert "1. clone the repo" in out
        assert "2. run the tests" in out

    def test_links_speak_text_not_url(self):
        out = to_speakable("See [the docs](https://example.com/deep/path) for more.")
        assert_speakable(out)
        assert "the docs" in out
        assert "example.com" not in out
        assert "https" not in out

    def test_inline_code_reads_content(self):
        out = to_speakable("Run `pytest -q` before pushing.")
        assert_speakable(out)
        assert "pytest -q" in out

    def test_table_linearized(self):
        md = "| Name | Score |\n|---|---|\n| Piper | 46ms |\n| Kokoro | 277ms |"
        out = to_speakable(md)
        assert_speakable(out)
        assert "Piper" in out
        assert "46ms" in out

    def test_code_block_summarize_default(self):
        md = "Here you go:\n\n```python\ndef f():\n    return 1\n\nprint(f())\n```\n\nDone."
        out = to_speakable(md)
        assert_speakable(out)
        assert "3 lines python code block" in out
        assert "def f" not in out
        assert "Done." in out

    def test_code_block_skip(self):
        md = "Before.\n\n```\nsecret()\n```\n\nAfter."
        out = to_speakable(md, code_blocks="skip")
        assert_speakable(out)
        assert "secret" not in out
        assert "Before." in out
        assert "After." in out

    def test_code_block_read(self):
        out = to_speakable("```\nls -la\n```", code_blocks="read")
        assert "ls -la" in out

    def test_single_line_code_block_grammar(self):
        out = to_speakable("```python\nx = 1\n```")
        assert "1 line python code block" in out

    def test_kitchen_sink_never_voices_markdown(self):
        md = (
            "# Big **Header**\n\n"
            "Some *emphasis*, a [link](http://x.io), and `code`.\n\n"
            "- bullet **one**\n- bullet `two`\n\n"
            "1. step\n\n"
            "> a quote\n\n"
            "| a | b |\n|---|---|\n| 1 | 2 |\n\n"
            "```js\nconsole.log('hi')\n```\n"
        )
        for mode in ("summarize", "skip", "read"):
            out = to_speakable(md, code_blocks=mode)
            if mode != "read":
                assert_speakable(out)
            else:
                for symbol in ["#", "**", "](", "|"]:
                    assert symbol not in out

    def test_empty_input(self):
        assert to_speakable("") == ""
        assert to_speakable("\n\n") == ""


class TestChunker:
    def test_sentence_completes_mid_stream(self):
        chunker = SentenceChunker()
        assert chunker.feed("The tests ") == []
        assert chunker.feed("pass. Now the") == ["The tests pass."]
        assert chunker.flush() == ["Now the"]

    def test_sentence_completes_at_end_of_stream_chunk(self):
        chunker = SentenceChunker()
        assert chunker.feed("Done.") == ["Done."]
        assert chunker.flush() == []

    def test_multiple_sentences_in_one_feed(self):
        chunker = SentenceChunker()
        out = chunker.feed("One. Two! Three? Four")
        assert out == ["One.", "Two!", "Three?"]
        assert chunker.flush() == ["Four"]

    def test_abbreviations_do_not_split(self):
        chunker = SentenceChunker()
        out = chunker.feed("Use faster-whisper, e.g. the base model. Done. ")
        assert out == ["Use faster-whisper, e.g. the base model.", "Done."]

    def test_decimals_do_not_split(self):
        chunker = SentenceChunker()
        out = chunker.feed("Latency is 46.5 ms on average. Good. ")
        assert out == ["Latency is 46.5 ms on average.", "Good."]

    def test_closing_quote_stays_with_sentence(self):
        chunker = SentenceChunker()
        out = chunker.feed('He said "done." Then left. ')
        assert out[0] == 'He said "done."'

    def test_flush_empty(self):
        assert SentenceChunker().flush() == []

    @pytest.mark.parametrize("token_size", [1, 3, 7])
    def test_token_size_invariance(self, token_size):
        text = "First sentence here. Second one follows! Third ends. "
        chunker = SentenceChunker()
        out = []
        for i in range(0, len(text), token_size):
            out.extend(chunker.feed(text[i : i + token_size]))
        out.extend(chunker.flush())
        assert out == ["First sentence here.", "Second one follows!", "Third ends."]
