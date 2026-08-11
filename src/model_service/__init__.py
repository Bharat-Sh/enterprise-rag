"""The GPU model service: BGE-M3 embeddings and cross-encoder reranking.

A **separate package from `rag`, and a separate process**, for the reasons set
out in docs/adr/0004: model weights want exactly one process per GPU while HTTP
handling wants many, and the API image must not carry 5-7 GB of CUDA to reach a
model it never loads.

The dependency arrow points one way. `model_service` may import `rag.core` — so
that all three processes emit the same structured log shape — and nothing else
of ours. `rag` must never import `model_service`; that is checked by an
import-linter contract, because a single convenience import would put torch back
in the API's dependency closure and undo the whole arrangement.

Install with the `gpu` extra:

    uv sync --extra gpu
    uv run python -m model_service

On a machine with no GPU, `MODEL_SERVICE_BACKEND=stub` serves the same HTTP
contract with a deterministic fake, which is how the routes are tested in CI.
"""

from __future__ import annotations

__all__: list[str] = []
