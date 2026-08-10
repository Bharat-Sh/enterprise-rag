"""PDF and DOCX parsing, and the hostile input they have to survive.

This is the first code in the system that processes genuinely attacker-supplied
binary formats, so the interesting half of this file is the malicious half. A
parser that reads well-formed documents is easy; the tests below are the ones
that would have caught a real vulnerability.

Fixtures are generated, not committed. A binary blob in the repository is opaque
in review, impossible to diff, and tempting to source from somewhere with
unclear provenance.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from rag.adapters.parsers import DocxParser, PdfParser, build_registry
from rag.core.config import IngestionSettings
from rag.domain.errors import InvalidInputError
from rag.domain.ingestion import ContentType
from tests.support import docx_document_xml, make_docx, make_pdf

DEFAULTS = IngestionSettings()


@pytest.fixture
def pdf_parser() -> PdfParser:
    return PdfParser(DEFAULTS)


@pytest.fixture
def docx_parser() -> DocxParser:
    return DocxParser(DEFAULTS)


class TestPdf:
    def test_text_is_extracted(self, pdf_parser: PdfParser) -> None:
        parsed = pdf_parser.parse(make_pdf(["Annual leave is twenty-five days."]))

        assert "Annual leave is twenty-five days." in parsed.text
        assert parsed.page_count == 1

    def test_every_page_is_read(self, pdf_parser: PdfParser) -> None:
        parsed = pdf_parser.parse(make_pdf([f"Page {index} content." for index in range(5)]))

        assert parsed.page_count == 5
        for index in range(5):
            assert f"Page {index} content." in parsed.text

    def test_pages_are_separated_by_a_blank_line(self, pdf_parser: PdfParser) -> None:
        # So the chunker's first separator — the paragraph break — prefers a
        # page boundary over anywhere inside a page.
        parsed = pdf_parser.parse(make_pdf(["First page.", "Second page."]))

        assert "\n\n" in parsed.text

    def test_the_page_cap_truncates_and_says_so(self) -> None:
        parser = PdfParser(IngestionSettings(max_pdf_pages=2))

        parsed = parser.parse(make_pdf([f"Page {index}." for index in range(6)]))

        # The true page count is still reported, so the truncation is visible
        # rather than looking like a six-page document that lost its content.
        assert parsed.page_count == 6
        assert parsed.metadata["pages_extracted"] == 2
        assert parsed.metadata["truncated"] is True
        assert "Page 5." not in parsed.text

    def test_a_file_that_is_not_a_pdf_is_refused(self, pdf_parser: PdfParser) -> None:
        # A domain error, so the worker dead-letters it rather than retrying an
        # unparseable file five times.
        with pytest.raises(InvalidInputError, match="could not be read"):
            pdf_parser.parse(b"%PDF-1.4\nthis is not really a pdf at all")

    def test_an_empty_pdf_reports_itself_empty(self, pdf_parser: PdfParser) -> None:
        """The scanned-document case.

        No text layer means no chunks. The pipeline turns `is_empty` into a
        loud failure, because the alternative is a `READY` document with zero
        chunks that is permanently unfindable while appearing to have worked.
        """
        parsed = pdf_parser.parse(make_pdf([]))

        assert parsed.is_empty

    def test_truncated_bytes_do_not_crash_the_worker(self, pdf_parser: PdfParser) -> None:
        whole = make_pdf(["Some content here."])

        with pytest.raises(InvalidInputError):
            pdf_parser.parse(whole[: len(whole) // 3])


class TestDocx:
    def test_paragraphs_are_extracted(self, docx_parser: DocxParser) -> None:
        parsed = docx_parser.parse(make_docx(["First paragraph.", "Second paragraph."]))

        assert parsed.text == "First paragraph.\n\nSecond paragraph."

    def test_empty_paragraphs_are_dropped(self, docx_parser: DocxParser) -> None:
        parsed = docx_parser.parse(make_docx(["Real content.", "", "   ", "More content."]))

        assert parsed.text == "Real content.\n\nMore content."

    def test_runs_within_a_paragraph_are_joined_without_separators(
        self, docx_parser: DocxParser
    ) -> None:
        """Word splits a sentence at every formatting change.

        Inserting a separator between runs would break words apart wherever
        somebody bolded a single term mid-sentence.
        """
        xml = (
            '<?xml version="1.0"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p>"
            "<w:r><w:t>The quick </w:t></w:r>"
            "<w:r><w:t>brown</w:t></w:r>"
            "<w:r><w:t> fox.</w:t></w:r>"
            "</w:p></w:body></w:document>"
        )

        parsed = docx_parser.parse(make_docx(document_xml=xml))

        assert parsed.text == "The quick brown fox."

    def test_table_cell_text_is_extracted(self, docx_parser: DocxParser) -> None:
        """Tables carry the densest factual content in most business documents.

        Their paragraphs are nested inside `w:tbl`, so a shallow scan of the
        body would silently drop every one of them.
        """
        xml = (
            '<?xml version="1.0"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:tbl><w:tr>"
            "<w:tc><w:p><w:r><w:t>Region</w:t></w:r></w:p></w:tc>"
            "<w:tc><w:p><w:r><w:t>Revenue</w:t></w:r></w:p></w:tc>"
            "</w:tr></w:tbl></w:body></w:document>"
        )

        parsed = docx_parser.parse(make_docx(document_xml=xml))

        assert "Region" in parsed.text
        assert "Revenue" in parsed.text

    def test_a_zip_that_is_not_a_docx_is_refused(self, docx_parser: DocxParser) -> None:
        # The sniffer identifies the *container*; this is where the payload is
        # confirmed. An XLSX or a JAR reaches this parser with the same magic.
        with pytest.raises(InvalidInputError, match="not a Word document"):
            docx_parser.parse(make_docx(omit_document=True))

    def test_a_file_that_is_not_a_zip_is_refused(self, docx_parser: DocxParser) -> None:
        with pytest.raises(InvalidInputError, match="not a readable DOCX"):
            docx_parser.parse(b"PK\x03\x04 followed by nonsense")

    def test_malformed_xml_is_refused(self, docx_parser: DocxParser) -> None:
        with pytest.raises(InvalidInputError, match="could not be parsed"):
            docx_parser.parse(make_docx(document_xml="<w:document><unclosed>"))

    def test_an_empty_document_reports_itself_empty(self, docx_parser: DocxParser) -> None:
        assert docx_parser.parse(make_docx([])).is_empty


class TestDocxHostileInput:
    """The attacks a DOCX carries by virtue of being a zip full of XML."""

    def test_billion_laughs_is_refused(self, docx_parser: DocxParser) -> None:
        """Entity expansion — the classic XML denial of service.

        Ten nested entities expanding ten-fold each is 10^9 characters from a
        few hundred bytes. The stdlib parser expands entities by default, which
        is exactly why `defusedxml` is used instead: this is refused before any
        expansion happens, not caught by a memory limit afterwards.
        """
        payload = (
            '<?xml version="1.0"?>'
            "<!DOCTYPE lolz ["
            '<!ENTITY lol "lol">'
            '<!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">'
            '<!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">'
            '<!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">'
            '<!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">'
            "]>"
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>&lol4;</w:t></w:r></w:p></w:body></w:document>"
        )

        with pytest.raises(InvalidInputError) as raised:
            docx_parser.parse(make_docx(document_xml=payload))

        # Pinned to the *defusedxml* rejection, not merely "some error". The
        # stdlib parser also fails on an undefined entity, so a looser assertion
        # would still pass with `defusedxml` removed — and would then be
        # protecting nothing while looking like it did.
        assert raised.value.details["reason"] == "EntitiesForbidden"

    def test_an_external_entity_is_refused(self, docx_parser: DocxParser) -> None:
        """XXE — the one that reads files off the server.

        A parser that resolves this returns the contents of `/etc/passwd` as
        document text, which is then chunked, embedded, and made searchable.
        """
        payload = (
            '<?xml version="1.0"?>'
            '<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>&xxe;</w:t></w:r></w:p></w:body></w:document>"
        )

        with pytest.raises(InvalidInputError) as raised:
            docx_parser.parse(make_docx(document_xml=payload))

        assert raised.value.details["reason"] == "EntitiesForbidden"

    def test_a_decompression_bomb_is_refused_without_decompressing_it(
        self, docx_parser: DocxParser
    ) -> None:
        """A few kilobytes on the wire, gigabytes in memory.

        The upload size limit does nothing about this — the *compressed* file is
        small. It is caught on the archive's own declared numbers, before a byte
        is decompressed.
        """
        bomb = make_docx(
            ["ordinary"], extra_parts={"word/media/bomb.bin": b"\x00" * (200 * 1024 * 1024)}
        )

        with pytest.raises(
            InvalidInputError, match=r"expands to more data|suspiciously compressible"
        ):
            docx_parser.parse(bomb)

    def test_an_archive_lying_about_its_sizes_is_still_caught(self) -> None:
        """The central directory is metadata the attacker writes.

        So `file_size` can simply be a lie, and the declared-size check alone
        would pass. The bounded read is what holds when the numbers do not.
        """
        parser = DocxParser(IngestionSettings(max_extracted_bytes=64 * 1024))

        # Honest header, oversized body: the declared total is small because
        # there is only one part, but reading it exceeds the budget.
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
            archive.writestr("word/document.xml", docx_document_xml(["x" * 200_000]))

        with pytest.raises(InvalidInputError, match="expands to more data"):
            parser.parse(buffer.getvalue())

    def test_too_many_parts_is_refused(self) -> None:
        # An archive of a million tiny files exhausts time and memory without
        # ever tripping a size limit.
        parser = DocxParser(IngestionSettings(max_archive_entries=10))
        extra = {f"word/media/{index}.bin": b"x" for index in range(50)}

        with pytest.raises(InvalidInputError, match="too many parts"):
            parser.parse(make_docx(["content"], extra_parts=extra))

    def test_an_ordinary_document_is_not_caught_by_the_limits(
        self, docx_parser: DocxParser
    ) -> None:
        """The guards must not reject real documents.

        A limit that fires on legitimate input gets raised until it fires on
        nothing, which is how hardening becomes decoration.
        """
        realistic = make_docx(
            [f"Paragraph {index}. " + "Some ordinary prose. " * 20 for index in range(200)],
            extra_parts={"word/styles.xml": b"<styles/>" * 100},
        )

        parsed = docx_parser.parse(realistic)

        assert "Paragraph 199." in parsed.text


class TestRegistry:
    def test_every_recognised_content_type_now_has_a_parser(self) -> None:
        registry = build_registry(DEFAULTS)

        assert registry.supported == set(ContentType)

    def test_the_registry_returns_the_right_parser(self) -> None:
        registry = build_registry(DEFAULTS)

        assert isinstance(registry.get(ContentType.PDF), PdfParser)
        assert isinstance(registry.get(ContentType.DOCX), DocxParser)
