"""Loading BGE-M3's vocabulary, and proving both sides loaded the same one.

Duplicated, on purpose, from `rag.adapters.tokenize.bge`. The worker needs this
to size chunks and this service needs it to reject over-length input, but
`model_service` may not import `rag.adapters` — that boundary is what keeps the
API's dependency closure free of torch, and it is not worth breaching to save
twenty lines.

The duplication is safe because both sides load *the same file* and both compute
`fingerprint` the same way. `/v1/info` publishes the fingerprint, so a mismatch
between the vocabulary that sized a chunk and the vocabulary that embeds it is
an observable fact rather than something to hope about. That mismatch is worth
detecting: chunks sized by a different tokenizer can exceed the model window,
and the tail is then lost with nothing raised anywhere.

`tokenizers` (the Rust library) rather than `transformers.AutoTokenizer`: a few
megabytes against a large dependency that drags torch behind it on most install
paths, for a job that is pure string-to-ids.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tokenizers import Tokenizer

__all__ = ["FINGERPRINT_LENGTH", "count_tokens", "fingerprint", "load_tokenizer"]

#: Enough of the digest to make a collision irrelevant, short enough to eyeball
#: in a log line or a probe response.
FINGERPRINT_LENGTH = 16


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load `tokenizer.json` from disk.

    Raises `FileNotFoundError` if it is absent, which in every caller here means
    a startup failure. Reaching the network for a vocabulary instead was
    rejected twice over: it puts a download in the critical path of every
    restart, and it means the bytes we tokenize with are whatever the hub served
    today rather than whatever the image was built with.
    """
    from tokenizers import Tokenizer

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Tokenizer file {resolved}: not found. Run `uv run python scripts/fetch_models.py` "
            f"to download it."
        )
    return Tokenizer.from_file(str(resolved))


def fingerprint(path: str | Path) -> str:
    """A stable short digest of the vocabulary file.

    Computed from the file's bytes rather than from the loaded object, because
    `Tokenizer` has no stable serialisation guarantee across library versions
    and we need two *different processes*, possibly running different versions,
    to agree.
    """
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return digest[:FINGERPRINT_LENGTH]


def count_tokens(tokenizer: Tokenizer, texts: Sequence[str]) -> list[int]:
    """Token counts including the special tokens the model will add.

    `add_special_tokens=True` because the limit being checked is the model's
    sequence budget, and BGE-M3 spends two of it on `<s>` and `</s>` before it
    sees a single word of the input. Counting without them accepts inputs that
    are then truncated by exactly the amount we failed to count.
    """
    if not texts:
        return []
    encodings = tokenizer.encode_batch(list(texts), add_special_tokens=True)
    return [len(encoding.ids) for encoding in encodings]
