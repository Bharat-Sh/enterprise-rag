"""Search — the read half of the platform.

`POST`, not `GET`, for a read operation. Deliberate: a query is user text of
arbitrary length, and putting it in a URL means it lands in access logs, proxy
logs and browser history — a search query is often as sensitive as the document
it finds. The body also carries structured narrowing (`document_ids`) that would
be awkward as repeated query parameters. The cost is that this is not cacheable
by intermediaries, which is fine, because a per-caller ACL-filtered result set
must never be cached by anything that cannot evaluate the ACL.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status

from rag.api.deps import RetrievalServiceDep, TenantRateLimit, require
from rag.api.schemas.search import SearchRequest, SearchResponse, SearchResult
from rag.domain.access import AuthenticatedPrincipal
from rag.domain.authz import Permission

router = APIRouter(prefix="/search", tags=["search"], dependencies=[TenantRateLimit])

SearcherDep = Annotated[AuthenticatedPrincipal, Depends(require(Permission.SEARCH_QUERY))]


@router.post(
    "",
    response_model=SearchResponse,
    summary="Search indexed documents",
    description=(
        "Embeds the query and returns the most similar chunks the caller is "
        "permitted to read. Access is enforced as a pre-filter inside the vector "
        "query and re-checked against the database, so results never include "
        "another tenant's content and never include documents the caller's "
        "principals do not match."
    ),
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "The embedding model or the vector index is unavailable."
        }
    },
)
async def search(
    body: SearchRequest,
    principal: SearcherDep,
    retrieval: RetrievalServiceDep,
) -> SearchResponse:
    """Run a search for the authenticated caller.

    `principal.access` is the only source of tenant and ACL scope. It is built
    from the verified credential in `get_principal`, with an API key's role
    ceiling already applied — which is what stops a key deliberately scoped to
    viewer from matching admin-granted document ACLs.
    """
    chunks = await retrieval.search(
        body.query,
        principal.access,
        top_k=body.top_k,
        collection_id=body.collection_id,
        document_ids=body.document_ids,
    )
    results = [
        SearchResult(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_title=chunk.document_title,
            ordinal=chunk.ordinal,
            text=chunk.text,
            score=chunk.score,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
        )
        for chunk in chunks
    ]
    return SearchResponse(query=body.query, results=results, count=len(results))


__all__ = ["router"]
