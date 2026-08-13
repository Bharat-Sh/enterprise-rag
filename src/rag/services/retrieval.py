"""Search: embed the query, filter in the index, hydrate from Postgres.

    query ──► model service ──► Qdrant (pre-filtered) ──► Postgres (re-checked)
                 embed              tenant + ACL              text + title

Two access checks, on purpose
-----------------------------
The vector store already pre-filtered by tenant and ACL — that is
non-negotiable #4 and it is the check that matters, because it runs *inside* the
query and therefore bounds what is read at all. `ChunkRepository.get_many` then
applies the same `AccessFilter` again in SQL, under row-level security.

That is not redundancy for its own sake. Qdrant is a **derived** index
(ADR-0001) and derived state drifts: an ACL change that failed to reproject, a
purge that half-completed, a restore from a stale snapshot. The second check is
what makes drift a recall problem instead of a disclosure. It costs one query
that has to happen anyway, since the index holds no text.

Why the index holds no text
---------------------------
See `rag.domain.retrieval`. The short version: hydration is not overhead we are
tolerating, it is the mechanism that makes the second check possible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.core.logging import get_logger
from rag.domain.embedding import EmbedMode
from rag.domain.errors import InvalidInputError
from rag.domain.retrieval import ScoredChunk

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from rag.domain.access import AccessFilter
    from rag.domain.ports import EmbeddingProvider, UnitOfWork, VectorStore

__all__ = ["MAX_TOP_K", "RetrievalService"]

_log = get_logger(__name__)

#: Ceiling on `top_k`. A caller asking for 10,000 results would have them
#: embedded, filtered, hydrated and serialised — an amplification of one cheap
#: request into a large one, which is a denial-of-service shape rather than a
#: usage pattern. Ranked results past the first few dozen are noise anyway.
MAX_TOP_K = 100


class RetrievalService:
    """Answers a search query for one authenticated caller."""

    def __init__(
        self,
        uow: UnitOfWork,
        *,
        embeddings: EmbeddingProvider,
        vectors: VectorStore,
    ) -> None:
        self._uow = uow
        self._embeddings = embeddings
        self._vectors = vectors

    async def search(
        self,
        query: str,
        access: AccessFilter,
        *,
        top_k: int = 10,
        collection_id: UUID | None = None,
        document_ids: Sequence[UUID] | None = None,
    ) -> list[ScoredChunk]:
        """Find the chunks most similar to `query` that this caller may read.

        `access` is required and positional. It comes from the verified
        credential and never from the request body — non-negotiable #5, and the
        reason there is no code path here that can search without one.
        """
        cleaned = query.strip()
        if not cleaned:
            # Embedding an empty string yields a vector that is technically
            # valid and semantically meaningless; it would return the corpus in
            # an arbitrary order and look like a working search.
            raise InvalidInputError("The search query is empty.")
        if top_k < 1 or top_k > MAX_TOP_K:
            raise InvalidInputError(
                f"top_k must be between 1 and {MAX_TOP_K}.",
                details={"top_k": top_k, "limit": MAX_TOP_K},
            )

        # `mode=QUERY` is discarded by BGE-M3, which uses no asymmetric prefix,
        # and is passed anyway so switching to a provider that needs it does not
        # silently degrade relevance here. See `rag.domain.embedding.EmbedMode`.
        [embedding] = await self._embeddings.embed([cleaned], mode=EmbedMode.QUERY)

        hits = await self._vectors.search(
            embedding,
            access,
            limit=top_k,
            collection_id=collection_id,
            document_ids=document_ids,
        )
        if not hits:
            return []

        # Second check. Same filter, different store — see the module docstring.
        chunks = await self._uow.chunks.get_many([hit.chunk_id for hit in hits], access)
        by_id = {chunk.id: chunk for chunk in chunks}

        titles = await self._document_titles({chunk.document_id for chunk in chunks}, access)

        results: list[ScoredChunk] = []
        for hit in hits:
            chunk = by_id.get(hit.chunk_id)
            if chunk is None:
                # The index knew about a chunk Postgres will not return: either
                # drift, or an ACL that changed since indexing. Dropped
                # silently from the caller's perspective — telling them a
                # result was withheld is itself a disclosure — but logged,
                # because a rising count here is the signal that reconciliation
                # is overdue.
                _log.info(
                    "retrieval.hydration_miss",
                    chunk_id=str(hit.chunk_id),
                    reason="absent_or_forbidden",
                )
                continue
            results.append(
                ScoredChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    document_title=titles.get(chunk.document_id, ""),
                    ordinal=chunk.ordinal,
                    text=chunk.text,
                    score=hit.score,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    metadata=dict(chunk.metadata),
                )
            )

        _log.info(
            "retrieval.search",
            hits=len(hits),
            returned=len(results),
            dropped=len(hits) - len(results),
            top_k=top_k,
        )
        return results

    async def _document_titles(
        self, document_ids: set[UUID], access: AccessFilter
    ) -> dict[UUID, str]:
        """Titles for the documents behind the results.

        Read through the ACL-aware `get`, so a title cannot be disclosed for a
        document whose chunk somehow survived the checks above. Cheap: search
        results cluster into a handful of documents, and this loop is bounded by
        `top_k` in the worst case.
        """
        titles: dict[UUID, str] = {}
        for document_id in document_ids:
            document = await self._uow.documents.get(document_id, access)
            if document is not None:
                titles[document_id] = document.title
        return titles
