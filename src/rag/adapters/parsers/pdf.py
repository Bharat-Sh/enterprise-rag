"""PDF text extraction (docs/adr/0010).

`pypdf` rather than the faster alternatives, and the reason is licensing rather
than engineering: **PyMuPDF is AGPL**, which would reach this entire codebase,
and this repository is MIT and intended to go public. `pdfplumber` gives better
layout reconstruction but is built on `pdfminer.six`, which is markedly slower
and brings a larger surface for the same job. `pypdf` is pure Python, MIT, and
extracts a text layer, which is all retrieval needs.

What this does not do
---------------------
**No OCR.** A scanned PDF has no text layer, and `ParsedDocument.is_empty` is
true for it — the pipeline then fails the document loudly rather than marking it
`READY` with zero chunks and leaving it permanently unfindable. That is the
right behaviour for now; OCR is a separate service with a GPU budget, not a
footnote to a parser.

**No layout reconstruction.** Multi-column PDFs extract in whatever order the
content stream stores, which is sometimes wrong. Retrieval is fairly tolerant of
this — chunks are still topically coherent — and fixing it properly means layout
analysis, which is `pdfplumber` at best and a model at worst.

Hostile input
-------------
A PDF is a container format with an object graph, and a malicious one can be
deeply nested, self-referential, or simply enormous. The page cap bounds the
work; the parse timeout in the pipeline bounds everything else. Encrypted PDFs
are refused rather than attempted, because the alternative is `pypdf` prompting
for a password in a background worker.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pypdf
from pypdf.errors import PdfReadError

from rag.core.logging import get_logger
from rag.domain.errors import InvalidInputError
from rag.domain.ingestion import ParsedDocument

if TYPE_CHECKING:
    from rag.core.config import IngestionSettings

__all__ = ["PdfParser"]

_log = get_logger(__name__)


class PdfParser:
    """Satisfies `rag.domain.ports.DocumentParser`."""

    def __init__(self, settings: IngestionSettings) -> None:
        self._max_pages = settings.max_pdf_pages

    def parse(self, data: bytes) -> ParsedDocument:
        try:
            reader = pypdf.PdfReader(io.BytesIO(data), strict=False)
        except (PdfReadError, ValueError, OSError) as exc:
            # `strict=False` already tolerates a great deal of malformation, so
            # reaching here means the file is not usefully a PDF. A domain error
            # so the worker dead-letters it instead of retrying five times.
            raise InvalidInputError(
                "The file could not be read as a PDF.",
                details={"reason": type(exc).__name__},
            ) from exc

        if reader.is_encrypted and not self._try_empty_password(reader):
            raise InvalidInputError(
                "The PDF is password-protected and cannot be indexed.",
                details={"encrypted": True},
            )

        pages = reader.pages
        total = len(pages)
        limit = min(total, self._max_pages)
        if total > limit:
            # Truncation is recorded rather than silent: a user searching for
            # something on page 2001 deserves an explanation, and `metadata`
            # is where a future API surface would read it from.
            _log.warning("pdf.truncated", pages=total, limit=limit)

        extracted: list[str] = []
        for index in range(limit):
            try:
                extracted.append(pages[index].extract_text() or "")
            # Broad on purpose: pypdf raises a variety of things on damaged
            # content streams, and one bad page is not a bad document.
            except Exception as exc:
                # Losing one page of a long report is far better than losing the
                # report, so the failure is recorded and extraction continues.
                _log.warning("pdf.page_failed", page=index, reason=type(exc).__name__)
                extracted.append("")

        # Pages are joined with a blank line so the chunker's first separator —
        # the paragraph break — lands on a page boundary in preference to
        # anywhere inside one.
        text = "\n\n".join(part.strip() for part in extracted if part.strip())

        return ParsedDocument(
            text=text,
            page_count=total,
            metadata={"pages_extracted": limit, "truncated": total > limit},
        )

    @staticmethod
    def _try_empty_password(reader: pypdf.PdfReader) -> bool:
        """Some PDFs are "encrypted" with an empty owner password.

        These are ubiquitous — it is how a tool marks a document read-only — and
        they are readable without any secret at all. Refusing them would reject
        a large slice of ordinary business documents.
        """
        try:
            return bool(reader.decrypt(""))
        except (PdfReadError, NotImplementedError, ValueError):
            # NotImplementedError covers algorithms pypdf does not support.
            return False
