"""DOCX text extraction (docs/adr/0010).

A `.docx` is a ZIP archive of XML. That single fact is the whole security story,
because it means an uploaded document arrives carrying two of the oldest
attacks in the book — a decompression bomb in the container, and entity
expansion in the payload — and both are enabled by default in the obvious
implementation.

Why not `python-docx`
---------------------
It would be the obvious choice, and it was rejected for a specific reason: it
hands the XML to `lxml` with a default parser, which **expands internal entities**
— that is billion-laughs — and it does no decompression accounting at all. So
using it would mean writing the zip hardening anyway, and then either patching
its parser or trusting it. Since the hardening is the substantial part and text
extraction from `w:t` elements is not, the library buys a dependency (plus lxml,
a C extension) and takes the security decisions out of view.

`zipfile` and `defusedxml` are both standard, and the result is that every limit
is visible in this file.

What is deliberately given up
-----------------------------
`python-docx` models styles, headers, footers, footnotes, and comments. This
extracts body text — every `w:t` in `word/document.xml`, including inside
tables, which is where most factual content in a business document lives.
Headers and footers are usually a page number and a confidentiality banner
repeated on every page, which is noise that would be embedded into every chunk.
Losing them is closer to a feature than a regression; footnotes are a genuine
loss and are noted rather than pretended away.
"""

from __future__ import annotations

import io
import zipfile
from typing import TYPE_CHECKING
from xml.etree.ElementTree import Element

from defusedxml.common import DefusedXmlException
from defusedxml.ElementTree import fromstring as safe_fromstring

from rag.core.logging import get_logger
from rag.domain.errors import InvalidInputError
from rag.domain.ingestion import ParsedDocument

if TYPE_CHECKING:
    from rag.core.config import IngestionSettings

__all__ = ["DOCUMENT_PART", "DocxParser"]

_log = get_logger(__name__)

#: The part every DOCX must contain. Its presence is what distinguishes a DOCX
#: from any other zip — the magic number identifies the container only, so
#: without this check an XLSX or a JAR would reach this parser and produce
#: nonsense rather than a clear refusal.
DOCUMENT_PART = "word/document.xml"

#: WordprocessingML namespace.
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_TEXT = f"{_W}t"
_PARAGRAPH = f"{_W}p"
_TAB = f"{_W}tab"
_BREAK = f"{_W}br"

_READ_CHUNK = 64 * 1024


class DocxParser:
    """Satisfies `rag.domain.ports.DocumentParser`."""

    def __init__(self, settings: IngestionSettings) -> None:
        self._max_extracted = settings.max_extracted_bytes
        self._max_ratio = settings.max_compression_ratio
        self._max_entries = settings.max_archive_entries

    def parse(self, data: bytes) -> ParsedDocument:
        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise InvalidInputError(
                "The file is not a readable DOCX archive.",
                details={"reason": type(exc).__name__},
            ) from exc

        with archive:
            self._assert_archive_is_sane(archive)
            xml = self._read_bounded(archive, DOCUMENT_PART)

        try:
            root = safe_fromstring(xml)
        except DefusedXmlException as exc:
            # Entity expansion, external entities, or a DTD. Refused before any
            # expansion happens, which is the entire point of `defusedxml`.
            _log.warning("docx.xml_refused", reason=type(exc).__name__)
            raise InvalidInputError(
                "The document contains XML constructs that are not permitted.",
                details={"reason": type(exc).__name__},
            ) from exc
        # Broad: malformed XML surfaces as any of several exception types,
        # and none of them should reach the worker as an unexpected crash.
        except Exception as exc:
            raise InvalidInputError(
                "The document body could not be parsed.",
                details={"reason": type(exc).__name__},
            ) from exc

        paragraphs = _paragraphs(root)
        return ParsedDocument(
            text="\n\n".join(paragraphs),
            metadata={"paragraphs": len(paragraphs)},
        )

    # -- container hardening -----------------------------------------------

    def _assert_archive_is_sane(self, archive: zipfile.ZipFile) -> None:
        """Refuse bombs before reading a single byte of content.

        Everything here is checked against the *central directory*, which the
        archive declares up front — so a bomb is rejected on its own stated
        numbers, without decompressing anything. The bounded read afterwards is
        what catches an archive that lies about them.
        """
        entries = archive.infolist()

        if len(entries) > self._max_entries:
            raise InvalidInputError(
                "The document archive contains too many parts.",
                details={"entries": len(entries), "limit": self._max_entries},
            )

        if not any(entry.filename == DOCUMENT_PART for entry in entries):
            # A zip, but not a DOCX. The sniffer can only identify the
            # container; this is where the payload is confirmed.
            raise InvalidInputError(
                "The archive is not a Word document.",
                details={"expected_part": DOCUMENT_PART},
            )

        declared_total = sum(entry.file_size for entry in entries)
        if declared_total > self._max_extracted:
            raise InvalidInputError(
                "The document expands to more data than is permitted.",
                details={"declared_bytes": declared_total, "limit": self._max_extracted},
            )

        for entry in entries:
            # A ratio needs a denominator; tiny entries are exempt because a
            # 12-byte stored file trivially exceeds any ratio and means nothing.
            if entry.compress_size < 1024:
                continue
            ratio = entry.file_size / entry.compress_size
            if ratio > self._max_ratio:
                raise InvalidInputError(
                    "The document contains a suspiciously compressible part.",
                    details={
                        "part": entry.filename,
                        "ratio": round(ratio),
                        "limit": self._max_ratio,
                    },
                )

    def _read_bounded(self, archive: zipfile.ZipFile, name: str) -> bytes:
        """Read one part, stopping if it exceeds the budget.

        The central directory is metadata an attacker writes, so `file_size` can
        simply be a lie. This reads incrementally and gives up at the limit —
        the check that holds when the declared numbers do not.
        """
        collected = bytearray()
        with archive.open(name) as part:
            while block := part.read(_READ_CHUNK):
                collected.extend(block)
                if len(collected) > self._max_extracted:
                    raise InvalidInputError(
                        "The document expands to more data than is permitted.",
                        details={"part": name, "limit": self._max_extracted},
                    )
        return bytes(collected)


# -- text extraction --------------------------------------------------------


def _paragraphs(root: Element) -> list[str]:
    """Body text, one entry per non-empty `w:p`.

    Paragraphs rather than one flat string because the chunker splits on blank
    lines first, so preserving them puts chunk boundaries on the breaks a reader
    would recognise.

    `iter` walks the whole tree, which matters: table cells contain their own
    `w:p` elements, and a shallow scan would silently drop every table in the
    document — usually the densest factual content it has.
    """
    paragraphs: list[str] = []
    for paragraph in root.iter(_PARAGRAPH):
        text = _text_of(paragraph)
        if text:
            paragraphs.append(text)
    return paragraphs


def _text_of(paragraph: Element) -> str:
    """Concatenate the runs inside one paragraph.

    Word splits a sentence across many `w:t` runs whenever formatting changes
    mid-sentence, so joining without a separator is correct — inserting one
    would break words apart wherever somebody bolded a single term.
    """
    parts: list[str] = []
    for node in paragraph.iter():
        if node.tag == _TEXT:
            parts.append(node.text or "")
        elif node.tag == _TAB:
            parts.append("\t")
        elif node.tag == _BREAK:
            parts.append("\n")
    return "".join(parts).strip()
