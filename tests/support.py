"""Shared helpers for building test credentials and configurations.

Keys are generated per call rather than checked in. A fixture private key in a
repository is a private key that eventually gets trusted somewhere real, and
every secret scanner would be right to flag it.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Sequence
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa

from rag.core.config import SigningAlgorithm

#: A database password that passes the production guard, so a test can exercise
#: some *other* production rule without tripping the default-credential check.
SAFE_PASSWORD = "not-the-development-default"


def generate_private_pem(algorithm: SigningAlgorithm = SigningAlgorithm.EDDSA) -> str:
    """A fresh PKCS#8 PEM private key. Ed25519 generation is microseconds."""
    key: ed25519.Ed25519PrivateKey | rsa.RSAPrivateKey = (
        ed25519.Ed25519PrivateKey.generate()
        if algorithm is SigningAlgorithm.EDDSA
        else rsa.generate_private_key(public_exponent=65537, key_size=2048)
    )
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")


def public_pem(private_pem: str) -> str:
    """The matching public key, for exercising retired-key verification."""
    private = serialization.load_pem_private_key(private_pem.encode("ascii"), password=None)
    return (
        private.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("ascii")
    )


def make_pdf(pages: Sequence[str]) -> bytes:
    """A minimal, valid PDF containing one text block per page.

    Generated rather than committed as a binary fixture. A checked-in PDF is
    opaque in review, impossible to diff, and tempting to copy from somewhere
    with unclear provenance; this is thirty lines that any reader can verify
    produces exactly what the test claims.

    Byte offsets in the cross-reference table are computed as the file is built,
    because a wrong `xref` is precisely the kind of malformation that would make
    a test pass or fail for reasons unrelated to what it is testing.
    """
    objects: list[bytes] = []

    page_object_ids = [4 + index * 2 for index in range(len(pages))]
    kids = " ".join(f"{object_id} 0 R" for object_id in page_object_ids)

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for index, text in enumerate(pages):
        content = f"BT /F1 24 Tf 72 720 Td ({_escape_pdf_text(text)}) Tj ET".encode()
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> "
                f"/Contents {page_object_ids[index] + 1} 0 R >>"
            ).encode()
        )
        objects.append(
            b"<< /Length "
            + str(len(content)).encode()
            + b" >>\nstream\n"
            + content
            + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode()

    return bytes(out)


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


#: The minimum set of parts a real DOCX carries. Only `word/document.xml` is
#: read, but a fixture missing the rest would not resemble what Word produces.
_DOCX_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" ContentType="application/vnd'
    '.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    "</Types>"
)

_DOCX_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006'
    '/relationships/officeDocument" Target="word/document.xml"/>'
    "</Relationships>"
)


def docx_document_xml(paragraphs: Sequence[str]) -> str:
    """WordprocessingML for a body of plain paragraphs."""
    body = "".join(f"<w:p><w:r><w:t>{_escape_xml(text)}</w:t></w:r></w:p>" for text in paragraphs)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )


def _escape_xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def make_docx(
    paragraphs: Sequence[str] | None = None,
    *,
    document_xml: str | None = None,
    extra_parts: dict[str, bytes] | None = None,
    omit_document: bool = False,
) -> bytes:
    """Build a DOCX in memory.

    The keyword arguments exist so a security test can produce a *malformed*
    one — a missing body part, an entity-expansion payload, a decompression
    bomb — without hand-assembling a zip in every test.
    """
    body = document_xml if document_xml is not None else docx_document_xml(paragraphs or [])

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _DOCX_CONTENT_TYPES)
        archive.writestr("_rels/.rels", _DOCX_RELS)
        if not omit_document:
            archive.writestr("word/document.xml", body)
        for name, content in (extra_parts or {}).items():
            archive.writestr(name, content)
    return buffer.getvalue()


def production_overrides(**extra: Any) -> dict[str, Any]:
    """Settings that satisfy every production guard.

    Centralised so that adding a new production requirement means updating one
    helper rather than hunting through the suite — and so a test that *should*
    fail the guard fails for the reason it names, not an unrelated one.
    """
    overrides: dict[str, Any] = {
        "database": {"password": SAFE_PASSWORD},
        "auth": {
            "private_key_pem": generate_private_pem(),
            "issuer": "https://rag.example.test",
        },
    }
    overrides.update(extra)
    return overrides


class StubEmbeddingProvider:
    """An in-process `EmbeddingProvider` backed by the model service's stub.

    Satisfies `rag.domain.ports.EmbeddingProvider` without a socket, so the
    ingestion tests — which are about parsing, chunking and the state machine —
    do not require a separately started process to reach `READY`.

    **Not a mock.** It delegates to `model_service.backend.StubBackend`, the
    same real implementation of the inference contract that CI serves over HTTP
    (docs/adr/0011): deterministic vectors derived from a hash, and a sparse
    vector built from real token frequencies. What it skips is the network and
    `HttpModelClient` — both of which are covered thoroughly elsewhere, by the
    M4 unit tests over `MockTransport` and by an integration module that runs
    the client against a live service over a real socket.

    It reports `embedding_model="stub"`, which is stamped onto every chunk row
    it produces, so a test asserting on that column can tell where the vectors
    came from.
    """

    def __init__(self) -> None:
        from model_service.backend import StubBackend

        self._backend = StubBackend()

    async def embed(self, texts: Sequence[str], *, mode: Any = None) -> list[Any]:
        from rag.domain.embedding import Embedding, SparseVector

        raw = self._backend.embed(list(texts), mode=str(mode or "passage"))
        return [
            Embedding(
                dense=item.dense,
                sparse=SparseVector(indices=item.sparse_indices, values=item.sparse_values),
            )
            for item in raw
        ]

    async def info(self) -> Any:
        from rag.domain.embedding import ModelInfo

        described = self._backend.info()
        return ModelInfo(
            embedding_model=described.embedding_model,
            embedding_version=described.embedding_version,
            dimensions=described.dimensions,
            max_sequence_tokens=8192,
            tokenizer_hash=described.tokenizer_hash,
            reranker_model=described.reranker_model,
        )

    async def aclose(self) -> None:
        """Nothing to close. Present so it is drop-in for `HttpModelClient`."""
