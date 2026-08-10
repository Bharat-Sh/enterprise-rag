"""Parsers for the text formats: plain text, Markdown, HTML.

All three are stdlib-only. HTML uses `html.parser`, which is forgiving of
malformed markup and — the reason it is chosen over an XML parser — has no
entity-expansion or external-entity behaviour to disable, so the billion-laughs
and XXE classes simply do not arise.

Decoding is UTF-8 with `errors="replace"`. The sniffer has already established
that the content decodes cleanly in its first 4 KiB; `replace` covers a file
that turns bad later on, and substituting a replacement character for a bad byte
is better than failing an otherwise good hundred-page document.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from rag.domain.ingestion import ParsedDocument

__all__ = ["HtmlParser", "MarkdownParser", "PlainTextParser"]

#: Elements whose *content* is code or styling, never prose. Their text would
#: otherwise be embedded and retrieved as if it were part of the document.
_NON_CONTENT_TAGS = frozenset({"script", "style", "noscript", "template", "svg", "head"})

#: Elements that imply a break in the text flow, so words either side are not
#: run together into a single nonsense token.
_BLOCK_TAGS = frozenset(
    {
        "p", "div", "br", "hr", "section", "article", "header", "footer", "aside",
        "h1", "h2", "h3", "h4", "h5", "h6",
        "ul", "ol", "li", "dl", "dt", "dd",
        "table", "thead", "tbody", "tr", "td", "th",
        "blockquote", "pre", "figure", "figcaption", "main", "nav",
    }
)  # fmt: skip

_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _tidy(text: str) -> str:
    """Normalise whitespace without destroying paragraph structure.

    Paragraph breaks survive because the chunker splits on them first — collapse
    them and every chunk boundary falls somewhere worse.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_SPACE.sub("\n", text)
    return _EXCESS_BLANK_LINES.sub("\n\n", text).strip()


class PlainTextParser:
    """Satisfies `rag.domain.ports.DocumentParser`."""

    def parse(self, data: bytes) -> ParsedDocument:
        return ParsedDocument(text=_tidy(_decode(data)))


class MarkdownParser:
    """Markdown, kept as-is.

    Deliberately not rendered to HTML and stripped back to text. Markdown's
    markup *is* structure a reader understands — `## Heading`, `- item` — and
    it survives embedding fine. Rendering it away costs a dependency and loses
    the heading cues that make a chunk interpretable on its own.
    """

    def parse(self, data: bytes) -> ParsedDocument:
        return ParsedDocument(text=_tidy(_decode(data)))


class _TextExtractor(HTMLParser):
    """Collects visible text, dropping scripts, styles, and markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._suppress_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _NON_CONTENT_TAGS:
            self._suppress_depth += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _NON_CONTENT_TAGS:
            # Clamped at zero: malformed markup with a stray closing tag would
            # otherwise drive this negative and suppress the rest of the file.
            self._suppress_depth = max(0, self._suppress_depth - 1)
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._suppress_depth == 0:
            self._parts.append(data)

    @property
    def text(self) -> str:
        return "".join(self._parts)


class HtmlParser:
    """Satisfies `rag.domain.ports.DocumentParser`."""

    def parse(self, data: bytes) -> ParsedDocument:
        extractor = _TextExtractor()
        extractor.feed(_decode(data))
        extractor.close()
        return ParsedDocument(text=_tidy(extractor.text))
