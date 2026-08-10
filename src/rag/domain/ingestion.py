"""Ingestion vocabulary: content types and what a parser returns.

Pure. The parsers themselves are adapters — this is only the shape of what they
hand back, so `rag.services` can orchestrate a pipeline without importing a PDF
library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = ["ContentType", "ParsedDocument"]


class ContentType(StrEnum):
    """A format we can ingest.

    Deliberately a closed set. "Accept anything and try" means an unrecognised
    file reaches a parser chosen by guesswork, and the failure surfaces deep in
    a worker rather than at the boundary where the user can act on it.
    """

    TEXT = "text/plain"
    MARKDOWN = "text/markdown"
    HTML = "text/html"
    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    @property
    def is_supported_in_m3a(self) -> bool:
        """Whether a parser exists yet.

        PDF and DOCX are recognised by the sniffer from M3a but parsed only from
        M3b. Recognising them early is what lets an upload be rejected with
        "not supported yet" instead of being silently mis-parsed as text — a
        binary run through a text decoder produces mojibake chunks that look
        like successful ingestion.
        """
        return self in {ContentType.TEXT, ContentType.MARKDOWN, ContentType.HTML}


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Extracted text plus whatever the parser learned on the way.

    One flat string rather than a tree of blocks. Chunking needs character
    offsets into a single text, and every structural refinement we might want
    later (headings, tables, page boundaries) can be carried in `metadata`
    without changing this shape or the column layout it feeds.
    """

    text: str
    #: `None` for formats with no concept of a page.
    page_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """No extractable text.

        Usually a scanned PDF — a real document to a human, and nothing at all
        to retrieval. Worth failing loudly: silently indexing zero chunks
        produces a document that is `READY` and unfindable.
        """
        return not self.text.strip()
