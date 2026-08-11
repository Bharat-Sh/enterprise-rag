"""Determine a file's type from its bytes, not from what the caller claimed.

`Content-Type` and the filename are both attacker-controlled. "This `.txt` is
actually a zip" is the oldest trick there is, and the consequence here is not
theoretical: a parser chosen from a lie is a parser handed input it never
expected, which is exactly where memory-exhaustion and decompression bombs live.

So the rule is: **content decides**. The declared type and the extension are
used only to break a tie the bytes cannot — distinguishing Markdown from plain
text, both of which are just text.

Deliberately hand-rolled rather than `python-magic` (a libmagic binding, so a
system dependency and a wheel problem on Windows) or `puremagic` (a large
signature database for a system that accepts five formats). Five formats is a
`startswith` each; a dependency here would be more code to audit, not less.
"""

from __future__ import annotations

from rag.domain.ingestion import ContentType

__all__ = ["MARKDOWN_EXTENSIONS", "sniff"]

#: Enough bytes for every signature below, and cheap to hold.
SNIFF_BYTES = 4096

_PDF_MAGIC = b"%PDF-"
#: DOCX is a zip. So is XLSX, PPTX, JAR, and any number of other things — the
#: signature identifies the container, never the payload, which is why the
#: DOCX parser (M3b) must verify `word/document.xml` exists rather than
#: trusting this.
_ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")

MARKDOWN_EXTENSIONS = frozenset({".md", ".markdown", ".mdown", ".mkd"})
_HTML_EXTENSIONS = frozenset({".html", ".htm", ".xhtml"})

#: Matched case-insensitively against the start of the decoded text.
_HTML_MARKERS = ("<!doctype html", "<html", "<head", "<body")

#: Control characters that never appear in text we would want to index.
#: Tab, newline, carriage return and form feed are excluded — they are text.
_TEXT_SAFE_CONTROLS = {0x09, 0x0A, 0x0D, 0x0C}


def _looks_binary(data: bytes) -> bool:
    """Whether the sample contains bytes no plain-text document would.

    A NUL is decisive: no text encoding we accept produces one mid-document, and
    it is the single most reliable binary marker. Beyond that, a high proportion
    of other control characters means the same thing with less certainty.
    """
    if b"\x00" in data:
        return True

    controls = sum(1 for byte in data if byte < 0x20 and byte not in _TEXT_SAFE_CONTROLS)
    # 5% is generous. Real text sits near zero; binary mistaken for text sits
    # far above. The threshold only has to separate two well-separated things.
    return bool(data) and controls / len(data) > 0.05


def _decode(data: bytes) -> str | None:
    """UTF-8 or nothing.

    Legacy encodings are guessable but not reliably, and a wrong guess produces
    plausible mojibake that survives all the way into retrieval results. A
    refusal the user can act on beats silent corruption.
    """
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def sniff(
    data: bytes,
    *,
    filename: str | None = None,
    declared: str | None = None,
) -> ContentType | None:
    """Identify `data`, or `None` if it is nothing we accept.

    `filename` and `declared` are hints of last resort, consulted only once the
    bytes have established that the content is text. They can never promote
    binary content to a text type or override a magic-number match.
    """
    sample = data[:SNIFF_BYTES]

    if sample.startswith(_PDF_MAGIC):
        return ContentType.PDF
    if sample.startswith(_ZIP_MAGIC):
        # Every zip-based format we might add lands here; only DOCX is claimed,
        # and the parser confirms it. See the note on `_ZIP_MAGIC`.
        return ContentType.DOCX

    if _looks_binary(sample):
        return None

    text = _decode(sample)
    if text is None:
        return None

    stripped = text.lstrip().lower()
    if any(stripped.startswith(marker) for marker in _HTML_MARKERS):
        return ContentType.HTML

    # From here the content is definitely text, so the caller's hints are safe
    # to consult: the worst they can do is pick the wrong *text* parser.
    suffix = _suffix(filename)
    if suffix in _HTML_EXTENSIONS or _declared_is(declared, ContentType.HTML):
        return ContentType.HTML
    if suffix in MARKDOWN_EXTENSIONS or _declared_is(declared, ContentType.MARKDOWN):
        return ContentType.MARKDOWN

    return ContentType.TEXT


def _suffix(filename: str | None) -> str:
    if not filename:
        return ""
    _, dot, extension = filename.rpartition(".")
    return f".{extension.lower()}" if dot else ""


def _declared_is(declared: str | None, content_type: ContentType) -> bool:
    if not declared:
        return False
    # Strip any `; charset=...` parameter before comparing.
    return declared.split(";", 1)[0].strip().lower() == content_type.value
