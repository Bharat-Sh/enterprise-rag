"""BGE-M3's real tokenizer, for sizing chunks exactly.

This is what M4 replaces `HeuristicTokenCounter` with. The estimator was honest
about being ~10% out, which was fine while nothing consumed the number; once
there is a model with a hard sequence limit on the other end, "roughly right"
means chunks that occasionally exceed the window and get truncated — losing
their tails into a vector that still looks perfectly valid.

Runs **in this process**, not over HTTP
---------------------------------------
`rag.domain.chunking` calls `count` once per candidate span while it recursively
splits, which is hundreds of calls per document. A counter that made a network
round trip would turn one document into hundreds of them, and would have to be
async, which would drag the thread hop into the domain layer. Only *inference*
is remote; the vocabulary is a 17 MB file and there is no reason for it not to
be local.

`tokenizers`, not `transformers`
--------------------------------
The Rust library is a few megabytes and does nothing but map strings to ids.
`transformers.AutoTokenizer` would do the same job while pulling in a large
dependency that drags torch behind it on most install paths — into the CPU-only
worker whose entire purpose is not to need one.

The model service loads the *same file* and publishes its fingerprint from
`/v1/info`, so "did chunking and embedding agree about what a token is" is a
question with an observable answer rather than an assumption.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from rag.core.errors import ConfigurationError

if TYPE_CHECKING:
    from tokenizers import Tokenizer

__all__ = ["FINGERPRINT_LENGTH", "BgeTokenCounter"]

#: Must match `model_service.tokenizer.FINGERPRINT_LENGTH`, or the two ends
#: publish digests of different lengths and can never be compared.
FINGERPRINT_LENGTH = 16


class BgeTokenCounter:
    """Satisfies `rag.domain.ports.TokenCounter` using BGE-M3's vocabulary."""

    def __init__(self, tokenizer_path: str | Path) -> None:
        """Load `tokenizer.json`.

        Raises `ConfigurationError` if it is missing. That is a *boot* failure by
        the time it reaches a caller, and deliberately not a fallback to the
        estimator: falling back would make chunk boundaries depend on whether a
        file happened to be present, so the same document would chunk
        differently on two machines with nothing logged and nothing failing.
        """
        from tokenizers import Tokenizer

        path = Path(tokenizer_path)
        if not path.is_file():
            raise ConfigurationError(
                f"Tokenizer file {path} was not found, but RAG_INGESTION__TOKENIZER is "
                f"'bge-m3'. Run `uv run python scripts/fetch_models.py`, or set the "
                f"tokenizer to 'heuristic' to use the estimator instead.",
                details={"tokenizer_path": str(path)},
            )

        self._tokenizer: Tokenizer = Tokenizer.from_file(str(path))
        # Digest the file's bytes, not the loaded object: `Tokenizer` has no
        # stable serialisation across library versions, and the two processes
        # being compared may not be running the same one.
        self._fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()[:FINGERPRINT_LENGTH]
        self._path = path

    @property
    def fingerprint(self) -> str:
        """Short digest of the vocabulary file, to compare against `/v1/info`."""
        return self._fingerprint

    @property
    def path(self) -> Path:
        return self._path

    def count(self, text: str) -> int:
        """Exact token count, including the special tokens the model will add.

        `add_special_tokens=True` because the budget being spent is the model's
        sequence limit, and BGE-M3 spends two of it on `<s>` and `</s>` before it
        sees any of the input. Counting without them produces chunks that are
        over budget by precisely the amount not counted — which is the whole
        class of bug this class exists to remove.

        Zero for empty input, matching `HeuristicTokenCounter`: the chunker uses
        a non-zero count as its proof that a span made progress, and an empty
        span that reported the two special tokens would let it loop.
        """
        if not text:
            return 0
        return len(self._tokenizer.encode(text, add_special_tokens=True).ids)
