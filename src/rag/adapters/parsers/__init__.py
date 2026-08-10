"""Document parsers, one per content type.

M3a covers the text formats, which need nothing but the standard library. PDF
and DOCX arrive in M3b along with the dependencies they require — and with the
hardening that parsing genuinely untrusted binary formats demands (XXE,
decompression bombs, element caps), which is a body of work in its own right
rather than a footnote to this one.
"""

from __future__ import annotations

from rag.adapters.parsers.registry import PARSERS, parser_for
from rag.adapters.parsers.text import HtmlParser, MarkdownParser, PlainTextParser

__all__ = ["PARSERS", "HtmlParser", "MarkdownParser", "PlainTextParser", "parser_for"]
