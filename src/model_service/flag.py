"""The real backend: BGE-M3 and bge-reranker-v2-m3 via FlagEmbedding.

**This is the one module in M4 that CI does not execute.** There is no GPU
runner, so everything above it is tested against `StubBackend` instead and this
file is covered only by an integration test that skips when the service is
unreachable. That is a deliberate trade (docs/adr/0011) and the reason the
surface here is kept as thin as it is: load models, call them, convert types.
Every decision that could be made in tested code has been pushed up into
`app.py`, `batching.py`, or `tokenizer.py`.

Why FlagEmbedding rather than transformers directly
---------------------------------------------------
BGE-M3's sparse output is not a pooling choice, it is a separate `sparse_linear`
head with its own weights, plus token-weight max-pooling and special-token
exclusion. Reimplementing that is about forty lines and easy to get subtly
wrong — and wrong here does not crash, it produces lexical weights that are
merely *worse*, which nothing detects until the M10 golden set says relevance
dropped and nobody knows when.

`sentence-transformers` was rejected outright: it has no sparse output at all,
which forfeits the hybrid retrieval that motivated choosing BGE-M3 (docs/adr/0004)
and would leave us maintaining two retrieval paths — and therefore two
implementations of "which chunks may this caller see".
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any

from model_service.backend import BackendInfo, RawEmbedding, normalise_l2
from model_service.tokenizer import count_tokens, fingerprint, load_tokenizer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from model_service.settings import Settings

__all__ = ["FlagBackend"]

#: `tokenizer.json` sits inside the model snapshot, so the vocabulary used to
#: measure input is by construction the one the weights were trained with.
TOKENIZER_FILENAME = "tokenizer.json"


def _construct(factory: Any, path: str, *, use_fp16: bool, device: str) -> Any:
    """Instantiate a FlagEmbedding model across its device-kwarg rename.

    FlagEmbedding 1.2 took `device`, 1.3 takes `devices`. Pinning a version
    instead would be the usual answer, but the GPU environment is built
    separately from the API's lockfile and is the one place a floating version
    is likely — so this reads the signature rather than guessing, and fails with
    a legible message rather than a `TypeError` about an unexpected keyword.
    """
    parameters = inspect.signature(factory).parameters
    kwargs: dict[str, Any] = {"use_fp16": use_fp16}
    if "devices" in parameters:
        kwargs["devices"] = device
    elif "device" in parameters:
        kwargs["device"] = device
    else:  # pragma: no cover - would mean a third incompatible rename
        raise RuntimeError(
            f"{factory.__name__} accepts neither `device` nor `devices`; "
            f"this FlagEmbedding version is not supported."
        )
    return factory(path, **kwargs)


class FlagBackend:
    """Satisfies `model_service.backend.InferenceBackend` on a real GPU.

    Not thread-safe, and not required to be: `app.py` holds a lock across every
    call. Two concurrent forward passes on one card is how a 6 GB device turns a
    tolerable latency into an out-of-memory error that can leave the CUDA
    context unusable for the remaining life of the process.
    """

    def __init__(self, settings: Settings) -> None:
        from FlagEmbedding import BGEM3FlagModel, FlagReranker

        self._settings = settings
        embedding_path = Path(settings.embedding_model_path)

        # Load the tokenizer first. It is the cheapest thing here and the most
        # likely to be missing, so a fetch that never ran fails in a second
        # rather than after a minute of loading weights.
        tokenizer_file = embedding_path / TOKENIZER_FILENAME
        self._tokenizer = load_tokenizer(tokenizer_file)
        self._tokenizer_hash = fingerprint(tokenizer_file)

        self._model = _construct(
            BGEM3FlagModel,
            str(embedding_path),
            use_fp16=settings.use_fp16,
            device=settings.device,
        )
        self._reranker = (
            _construct(
                FlagReranker,
                settings.reranker_model_path,
                use_fp16=settings.use_fp16,
                device=settings.device,
            )
            if settings.reranker_enabled
            else None
        )

        # Probe the real width rather than hardcoding 1024. If a variant is ever
        # configured here, the number reported to callers — and written into
        # every chunk row and the vector store's collection schema — must be the
        # one this model actually produces.
        self._dimensions = len(self.embed(["dimension probe"], mode="passage")[0].dense)

    def info(self) -> BackendInfo:
        return BackendInfo(
            embedding_model=Path(self._settings.embedding_model_path).name,
            embedding_version=self._settings.embedding_model_version,
            dimensions=self._dimensions,
            tokenizer_hash=self._tokenizer_hash,
            backend="flag",
            reranker_model=(
                Path(self._settings.reranker_model_path).name
                if self._reranker is not None
                else None
            ),
        )

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        return count_tokens(self._tokenizer, texts)

    def embed(self, texts: Sequence[str], *, mode: str) -> list[RawEmbedding]:
        # BGE-M3 uses no asymmetric instruction prefix, so query and passage
        # embeddings are identical. Accepted and discarded — see
        # `rag.domain.embedding.EmbedMode` for why the parameter exists at all.
        del mode

        output = self._model.encode(
            list(texts),
            batch_size=len(texts),
            max_length=self._settings.max_sequence_tokens,
            return_dense=True,
            return_sparse=True,
            # ColBERT multi-vectors are ~100x the dense footprint per chunk.
            # Deferred to M10 as an eval-driven experiment (docs/adr/0004).
            return_colbert_vecs=False,
        )

        dense_vectors = output["dense_vecs"]
        # A list of {token_id_as_string: weight}. The string keys are
        # FlagEmbedding's choice, not ours, and the weights arrive as numpy
        # float16 — both are converted here so nothing numpy-shaped escapes this
        # module into the JSON serialiser.
        lexical_weights = output["lexical_weights"]

        embeddings: list[RawEmbedding] = []
        for dense, sparse in zip(dense_vectors, lexical_weights, strict=True):
            indices = sorted(int(token_id) for token_id in sparse)
            embeddings.append(
                RawEmbedding(
                    # Normalised explicitly rather than trusting `encode`'s
                    # default, which has changed across versions. The contract
                    # says these are unit vectors; that must not depend on a
                    # library default staying put.
                    dense=normalise_l2([float(value) for value in dense]),
                    sparse_indices=tuple(indices),
                    sparse_values=tuple(float(sparse[str(index)]) for index in indices),
                )
            )
        return embeddings

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        if self._reranker is None:
            raise RuntimeError("Reranking is disabled on this service.")

        scores = self._reranker.compute_score(
            [[query, passage] for passage in passages],
            normalize=False,
        )
        # `compute_score` returns a bare float for a single pair and a list for
        # several. Handling only the list case works in every test with two
        # passages and fails in production on the day someone reranks one.
        if isinstance(scores, (int, float)):
            return [float(scores)]
        return [float(score) for score in scores]
