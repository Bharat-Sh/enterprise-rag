"""What an embedding *is*, independent of who computed it.

These types are the vocabulary shared by the ingestion pipeline, the retrieval
service, the vector store, and the HTTP client that talks to the model service.
None of them knows that BGE-M3 exists, which is the point: swapping to Voyage or
Cohere changes an adapter, not this module and not its callers.

Why dense *and* sparse in one type
----------------------------------
BGE-M3 produces both from a single forward pass (docs/adr/0004), and they are
always written to and read from the index together. Modelling them as two
separate results would let a caller persist one without the other — a chunk
indexed densely but not sparsely is silently half-retrievable, and nothing
errors. Keeping them in one frozen object makes that state unrepresentable.

Everything here is frozen. An embedding is a measurement of a specific string by
a specific model; mutating one in place means the vector in memory no longer
corresponds to anything that was ever computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "EmbedMode",
    "Embedding",
    "ModelInfo",
    "RerankResult",
    "SparseVector",
]


class EmbedMode(StrEnum):
    """Whether a text is being embedded as a search query or as stored content.

    BGE-M3 **ignores this** — unlike E5 or Instructor it uses no asymmetric
    instruction prefix, so `query` and `passage` produce identical vectors.
    The parameter exists anyway, for two reasons: it is part of the contract in
    docs/adr/0004, and every hosted alternative we might substitute (Voyage,
    Cohere) requires it. Retrofitting a required argument later means touching
    every call site *and* re-embedding every document indexed before the fix,
    because the vectors would no longer match. Carrying an unused parameter is
    much the cheaper of the two.
    """

    QUERY = "query"
    PASSAGE = "passage"


@dataclass(frozen=True, slots=True)
class SparseVector:
    """Lexical term weights, as two parallel arrays.

    `indices` are vocabulary token ids and `values` their learned weights. This
    is exactly Qdrant's sparse-vector shape, so indexing needs no conversion.

    A dict of ``{token_id: weight}`` was rejected as the wire and storage form:
    JSON object keys are strings, which roughly doubles the payload and forces
    an int cast on every read, and dict ordering then becomes something callers
    accidentally depend on.

    Only non-zero terms appear — that is what makes it sparse. A vocabulary of
    250k with a few dozen active terms is the normal case.
    """

    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        # Two arrays that disagree in length are not a sparse vector, they are
        # corruption. Caught at construction because the alternative is a
        # confusing failure deep inside the vector store, or worse, silent
        # truncation to the shorter of the two.
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"SparseVector has {len(self.indices)} indices but "
                f"{len(self.values)} values; they must correspond."
            )

    def __len__(self) -> int:
        """Number of non-zero terms."""
        return len(self.indices)

    @property
    def is_empty(self) -> bool:
        """True when no term carries weight.

        Legitimate for pathological input (a chunk of pure punctuation), so it
        is a property to check rather than an error to raise. A caller that
        cares — indexing does — decides what to do about it.
        """
        return not self.indices


@dataclass(frozen=True, slots=True)
class Embedding:
    """One text's vectors, as produced by a single forward pass."""

    dense: tuple[float, ...]
    sparse: SparseVector

    @property
    def dimensions(self) -> int:
        return len(self.dense)


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Identity of the models behind a provider.

    Not decoration. `embedding_model` and `embedding_version` are stamped onto
    every chunk row (`chunks.embedding_model`, `chunks.embedding_version`),
    which is what makes changing model survivable: embed into a new named
    vector, backfill the old rows, cut over, drop the old vector. Without a
    recorded version there is no way to tell which rows have been backfilled and
    the only safe migration is to re-embed the entire corpus.

    `tokenizer_hash` exists so a mismatch between the tokenizer used for
    *chunking* and the one used for *embedding* is detectable. If they diverge,
    chunks sized to 512 tokens are silently truncated at embedding time and lose
    their tails — invisible, permanent, and the single worst failure shape here.
    """

    embedding_model: str
    embedding_version: str
    dimensions: int
    max_sequence_tokens: int
    tokenizer_hash: str
    reranker_model: str | None = None


@dataclass(frozen=True, slots=True)
class RerankResult:
    """One passage's relevance to a query.

    Carries the passage's **index in the request**, not the passage text. The
    caller already holds the passages and their chunk ids; echoing the text back
    would multiply the response size for nothing, and would make the response a
    second copy of tenant data crossing the network.

    `score` is a raw cross-encoder logit — unbounded, and comparable only within
    one query's results. Do not threshold it against a constant across queries.
    """

    index: int
    score: float
