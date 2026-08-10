"""Content sniffing: the bytes decide, the caller's claims do not.

The lying-caller cases are the point. Everything else is a smoke test.
"""

from __future__ import annotations

import pytest

from rag.domain.ingestion import ContentType
from rag.domain.sniff import sniff

PDF = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\ntrailing content"
ZIP = b"PK\x03\x04\x14\x00\x08\x00\x08\x00 and then some"
PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\x00"


class TestMagicNumbersWin:
    def test_a_pdf_is_detected_from_its_header(self) -> None:
        assert sniff(PDF) is ContentType.PDF

    def test_a_zip_is_reported_as_docx(self) -> None:
        # The signature identifies the *container*. The DOCX parser (M3b) is
        # responsible for confirming `word/document.xml` is actually inside.
        assert sniff(ZIP) is ContentType.DOCX

    def test_a_pdf_named_txt_and_declared_text_is_still_a_pdf(self) -> None:
        detected = sniff(PDF, filename="notes.txt", declared="text/plain")

        assert detected is ContentType.PDF

    def test_a_zip_named_pdf_is_not_a_pdf(self) -> None:
        """The oldest trick there is.

        A parser chosen from a lie is a parser handed input it never expected,
        which is where the decompression-bomb and memory-exhaustion classes
        live.
        """
        detected = sniff(ZIP, filename="invoice.pdf", declared="application/pdf")

        assert detected is ContentType.DOCX


class TestBinaryIsRefused:
    def test_a_png_is_not_ingestible(self) -> None:
        assert sniff(PNG) is None

    def test_nul_bytes_mean_binary(self) -> None:
        assert sniff(b"looks like text\x00but is not") is None

    def test_undecodable_bytes_are_refused(self) -> None:
        # Latin-1 text. Guessable, but a wrong guess produces plausible mojibake
        # that survives into retrieval results; a refusal is more honest.
        assert sniff("caf\xe9 r\xe9sum\xe9".encode("latin-1")) is None

    def test_a_binary_file_cannot_be_promoted_by_a_hint(self) -> None:
        assert sniff(PNG, filename="readme.md", declared="text/markdown") is None

    def test_control_character_soup_is_refused(self) -> None:
        assert sniff(bytes(range(1, 32)) * 20) is None


class TestTextFormats:
    def test_plain_text(self) -> None:
        assert sniff(b"Just some ordinary prose.") is ContentType.TEXT

    @pytest.mark.parametrize(
        "body",
        [b"<!DOCTYPE html><html><body>hi</body></html>", b"  \n<html>", b"<HEAD>", b"<Body>"],
    )
    def test_html_is_detected_from_content(self, body: bytes) -> None:
        assert sniff(body) is ContentType.HTML

    def test_html_by_extension_when_the_content_is_ambiguous(self) -> None:
        # No markers at the start, but it is text, so a hint is safe: the worst
        # outcome is picking the wrong *text* parser.
        assert sniff(b"a fragment <p>with markup</p>", filename="page.html") is ContentType.HTML

    @pytest.mark.parametrize("filename", ["notes.md", "NOTES.MARKDOWN", "a.mkd"])
    def test_markdown_by_extension(self, filename: str) -> None:
        assert sniff(b"# Heading\n\nBody.", filename=filename) is ContentType.MARKDOWN

    def test_markdown_by_declared_type(self) -> None:
        assert sniff(b"# Heading", declared="text/markdown") is ContentType.MARKDOWN

    def test_a_charset_parameter_does_not_defeat_the_match(self) -> None:
        detected = sniff(b"# Heading", declared="text/markdown; charset=utf-8")

        assert detected is ContentType.MARKDOWN

    def test_unknown_extensions_fall_back_to_plain_text(self) -> None:
        assert sniff(b"key = value", filename="config.ini") is ContentType.TEXT

    def test_a_filename_with_no_extension_is_fine(self) -> None:
        assert sniff(b"contents", filename="LICENSE") is ContentType.TEXT


class TestEdges:
    def test_empty_input_is_text(self) -> None:
        # An empty upload is rejected later, by the service, with a message
        # about being empty rather than about its type.
        assert sniff(b"") is ContentType.TEXT

    def test_binary_within_the_sniff_window_is_caught(self) -> None:
        assert sniff(b"plain text " * 100 + b"\x00" * 10_000) is None

    def test_only_the_head_is_examined(self) -> None:
        """A deliberate limit, worth stating rather than discovering.

        Sniffing all of a 50 MB upload would be pointless work, so only the
        first `SNIFF_BYTES` are examined — which means a file that is text for
        4 KiB and binary afterwards is accepted as text. The parser then decodes
        it with `errors="replace"`, so the outcome is replacement characters in
        a few chunks rather than anything unsafe.
        """
        head = b"plain text " * 400  # comfortably past the 4 KiB window
        assert len(head) > 4096

        assert sniff(head + b"\x00" * 10_000) is ContentType.TEXT
