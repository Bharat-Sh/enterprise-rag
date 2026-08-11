"""Document parsers, one per content type.

The text formats need nothing but the standard library. PDF and DOCX need real
libraries and, more importantly, real limits — they are the first code in this
system to process genuinely hostile binary input. See docs/adr/0010.
"""

from __future__ import annotations

from rag.adapters.parsers.docx import DocxParser
from rag.adapters.parsers.pdf import PdfParser
from rag.adapters.parsers.registry import ParserRegistry, build_registry
from rag.adapters.parsers.text import HtmlParser, MarkdownParser, PlainTextParser

__all__ = [
    "DocxParser",
    "HtmlParser",
    "MarkdownParser",
    "ParserRegistry",
    "PdfParser",
    "PlainTextParser",
    "build_registry",
]
