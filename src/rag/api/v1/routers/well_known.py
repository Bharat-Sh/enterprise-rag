"""`/.well-known/jwks.json` — the public key set.

Mounted at the root rather than under `/api/v1`, like the health probes. The
path is fixed by RFC 8615 and by every JOSE client's default behaviour; moving
it when the API version changes would break verifiers that are not ours.

Unauthenticated by design. Publishing public keys is the entire point of signing
tokens asymmetrically: a gateway, a sidecar, or a future worker can verify a
token without holding anything that could mint one. A test asserts the response
carries no private material.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response

from rag.api.deps import TokenServiceDep

router = APIRouter(tags=["well-known"])


@router.get(
    "/.well-known/jwks.json",
    summary="JSON Web Key Set",
    description=(
        "The public keys this service signs access tokens with. During a key "
        "rotation it contains both the new signing key and the retired one, so "
        "tokens minted before the changeover keep verifying until they expire."
    ),
)
async def jwks(tokens: TokenServiceDep, response: Response) -> dict[str, Any]:
    # Cacheable, but not for long: a rotation must reach verifiers quickly, and
    # a key set cached for a day is a key set that rejects valid tokens for a
    # day. Five minutes is the usual compromise.
    response.headers["Cache-Control"] = "public, max-age=300"
    return tokens.public_jwks()


__all__ = ["router"]
