"""Which parser handles which content type.

A factory rather than a module-level dict, because the binary parsers need the
configured limits — page caps, decompression budgets — and reading configuration
at import time is how a test ends up unable to set them.

Still a plain dict inside. There are five content types and all are known at
import; a plugin mechanism with registration hooks would be machinery for a
table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.adapters.parsers.docx import DocxParser
from rag.adapters.parsers.pdf import PdfParser
from rag.adapters.parsers.text import HtmlParser, MarkdownParser, PlainTextParser
from rag.domain.ingestion import ContentType

if TYPE_CHECKING:
    from rag.core.config import IngestionSettings
    from rag.domain.ports import DocumentParser

__all__ = ["ParserRegistry", "build_registry"]


class ParserRegistry:
    """Maps a content type to the parser that handles it.

    `supported` is what the upload endpoint checks, so a format with no parser
    is refused at the boundary with an honest reason rather than being accepted,
    queued, and dead-lettered by a worker minutes later.
    """

    def __init__(self, parsers: dict[ContentType, DocumentParser]) -> None:
        self._parsers = parsers

    def get(self, content_type: ContentType) -> DocumentParser | None:
        return self._parsers.get(content_type)

    @property
    def supported(self) -> frozenset[ContentType]:
        return frozenset(self._parsers)


def build_registry(settings: IngestionSettings) -> ParserRegistry:
    """Every parser this build can run."""
    return ParserRegistry(
        {
            ContentType.TEXT: PlainTextParser(),
            ContentType.MARKDOWN: MarkdownParser(),
            ContentType.HTML: HtmlParser(),
            ContentType.PDF: PdfParser(settings),
            ContentType.DOCX: DocxParser(settings),
        }
    )
