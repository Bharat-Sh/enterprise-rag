"""Which parser handles which content type.

A plain dict rather than a registry class with registration hooks. There are
five content types and they are all known at import time; a plugin mechanism
would be machinery for a table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.adapters.parsers.text import HtmlParser, MarkdownParser, PlainTextParser
from rag.domain.ingestion import ContentType

if TYPE_CHECKING:
    from rag.domain.ports import DocumentParser

__all__ = ["PARSERS", "parser_for"]

#: Content types with a parser *today*. PDF and DOCX are recognised by the
#: sniffer and deliberately absent here until M3b, so an upload of one is
#: refused at the boundary with an honest reason rather than being handed to
#: the text parser and turned into mojibake.
PARSERS: dict[ContentType, DocumentParser] = {
    ContentType.TEXT: PlainTextParser(),
    ContentType.MARKDOWN: MarkdownParser(),
    ContentType.HTML: HtmlParser(),
}


def parser_for(content_type: ContentType) -> DocumentParser | None:
    """The parser for `content_type`, or `None` if none exists yet."""
    return PARSERS.get(content_type)
