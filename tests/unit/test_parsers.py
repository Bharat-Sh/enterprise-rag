"""Text parsers, the token estimator, and backoff jitter."""

from __future__ import annotations

import pytest

from rag.adapters.parsers import HtmlParser, MarkdownParser, PlainTextParser
from rag.adapters.tokenize import HeuristicTokenCounter
from rag.db.repositories.job import backoff_delay


class TestPlainText:
    def test_text_survives_intact(self) -> None:
        parsed = PlainTextParser().parse(b"First line.\n\nSecond paragraph.")

        assert parsed.text == "First line.\n\nSecond paragraph."

    def test_windows_line_endings_are_normalised(self) -> None:
        # Otherwise `\r` lands inside chunk text and inside embeddings.
        parsed = PlainTextParser().parse(b"one\r\ntwo\r\n\r\nthree")

        assert "\r" not in parsed.text
        assert parsed.text == "one\ntwo\n\nthree"

    def test_paragraph_breaks_survive(self) -> None:
        # The chunker splits on them first; collapsing them would push every
        # boundary somewhere worse.
        parsed = PlainTextParser().parse(b"a\n\n\n\n\n\nb")

        assert parsed.text == "a\n\nb"

    def test_undecodable_bytes_do_not_fail_the_document(self) -> None:
        parsed = PlainTextParser().parse(b"good text " + b"\xff\xfe" + b" more")

        assert "good text" in parsed.text
        assert "more" in parsed.text

    def test_empty_input_is_reported_as_empty(self) -> None:
        assert PlainTextParser().parse(b"   \n\n  ").is_empty


class TestMarkdown:
    def test_markup_is_preserved(self) -> None:
        """Deliberately not rendered away.

        `## Heading` and `- item` are structure a reader understands and that
        survives embedding; stripping them costs a dependency and loses the
        heading cues that make a chunk interpretable alone.
        """
        source = b"# Title\n\nSome *emphasis* and a [link](http://example.com).\n\n- one\n- two"

        parsed = MarkdownParser().parse(source)

        assert "# Title" in parsed.text
        assert "*emphasis*" in parsed.text
        assert "- one" in parsed.text


class TestHtml:
    def test_tags_are_stripped_and_text_kept(self) -> None:
        parsed = HtmlParser().parse(b"<html><body><p>Hello <b>world</b>.</p></body></html>")

        assert "Hello" in parsed.text
        assert "world" in parsed.text
        assert "<p>" not in parsed.text

    def test_script_and_style_content_is_dropped(self) -> None:
        # Otherwise minified JavaScript is embedded and retrieved as if it were
        # prose, which pollutes every result for that document.
        source = (
            b"<html><head><style>.a{color:red}</style></head>"
            b"<body><script>var secret = 1;</script><p>Real content.</p></body></html>"
        )

        parsed = HtmlParser().parse(source)

        assert "Real content." in parsed.text
        assert "secret" not in parsed.text
        assert "color:red" not in parsed.text

    def test_block_elements_become_breaks(self) -> None:
        # Without this, "one" and "two" run together into a token that is in no
        # vocabulary and matches no query.
        parsed = HtmlParser().parse(b"<p>one</p><p>two</p>")

        assert "onetwo" not in parsed.text
        assert "one" in parsed.text
        assert "two" in parsed.text

    def test_entities_are_decoded(self) -> None:
        parsed = HtmlParser().parse(b"<p>caf&eacute; &amp; cr&egrave;me</p>")

        assert "café" in parsed.text
        assert "&" in parsed.text

    def test_malformed_markup_does_not_raise(self) -> None:
        parsed = HtmlParser().parse(b"<p>unclosed <b>bold <div>nested</p></span>")

        assert "unclosed" in parsed.text

    def test_a_stray_closing_tag_does_not_suppress_the_rest(self) -> None:
        # The suppression depth is clamped at zero; without that, this markup
        # would drive it negative and swallow every following character.
        parsed = HtmlParser().parse(b"</script><p>still visible</p>")

        assert "still visible" in parsed.text


class TestTokenEstimator:
    def test_empty_text_is_zero(self) -> None:
        assert HeuristicTokenCounter().count("") == 0

    def test_any_non_empty_text_is_at_least_one(self) -> None:
        # The chunker uses this to decide whether progress was made; a span
        # counting as zero would let it loop.
        assert HeuristicTokenCounter().count("a") >= 1

    def test_english_prose_is_roughly_a_quarter_of_its_length(self) -> None:
        text = "The quick brown fox jumps over the lazy dog. " * 10

        estimate = HeuristicTokenCounter().count(text)

        assert len(text) / 6 < estimate < len(text) / 2

    def test_cjk_is_counted_far_denser_than_latin(self) -> None:
        """Not a nicety.

        At four characters per token, a CJK document would be underestimated
        four-fold and produce chunks that silently overflow the model's window.
        """
        counter = HeuristicTokenCounter()

        assert counter.count("日本語のテキストです") > counter.count("a" * 10)

    def test_the_count_grows_with_the_text(self) -> None:
        counter = HeuristicTokenCounter()

        assert counter.count("word " * 100) > counter.count("word " * 10)

    def test_a_zero_ratio_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            HeuristicTokenCounter(chars_per_token=0)


class TestBackoffJitter:
    def test_the_delay_grows_with_attempts(self) -> None:
        # Compared at a pinned jitter sample, so the growth is the only variable.
        early = backoff_delay(1, jitter=lambda: 0.5)
        later = backoff_delay(4, jitter=lambda: 0.5)

        assert later > early

    def test_the_delay_is_capped(self) -> None:
        # A poisoned job must still retry within an operator's attention span.
        assert backoff_delay(50, jitter=lambda: 1.0).total_seconds() <= 600

    def test_jitter_spreads_the_delay_but_keeps_a_floor(self) -> None:
        """The whole point: break up a synchronised retry herd.

        Full jitter across the entire window would occasionally schedule a
        retry almost immediately, which is the opposite of backing off. Half the
        window keeps a guaranteed floor.
        """
        lowest = backoff_delay(3, jitter=lambda: 0.0).total_seconds()
        highest = backoff_delay(3, jitter=lambda: 1.0).total_seconds()

        assert lowest < highest
        assert lowest >= highest / 2

    def test_successive_calls_differ(self) -> None:
        # Without jitter every job that failed on a shared cause retries at the
        # same instant and knocks the recovering dependency over again.
        delays = {backoff_delay(3).total_seconds() for _ in range(50)}

        assert len(delays) > 1
