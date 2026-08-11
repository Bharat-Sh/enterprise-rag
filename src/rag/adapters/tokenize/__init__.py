"""Token counting adapters.

Two implementations of `rag.domain.ports.TokenCounter`, selected by
`IngestionSettings.tokenizer`:

- `HeuristicTokenCounter` — a character-ratio estimate. No files, no
  dependencies, roughly 10% out. The default, so a fresh clone runs with no
  model download.
- `BgeTokenCounter` — BGE-M3's real vocabulary, exact, and the same one the
  model service embeds with. What production should use.

The choice is explicit configuration and never an automatic fallback, because
the two produce different chunk boundaries: silently picking one based on
whether a file exists would mean the same document chunks differently on two
machines, with nothing logged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.adapters.tokenize.bge import BgeTokenCounter
from rag.adapters.tokenize.estimator import HeuristicTokenCounter
from rag.core.config import TokenizerKind

if TYPE_CHECKING:
    from rag.core.config import IngestionSettings
    from rag.domain.ports import TokenCounter

__all__ = ["BgeTokenCounter", "HeuristicTokenCounter", "build_token_counter"]


def build_token_counter(settings: IngestionSettings) -> TokenCounter:
    """Construct the configured counter.

    Called at startup in both the API and the worker, so a misconfigured or
    missing vocabulary kills the process at boot rather than surfacing as a
    failed ingestion job an hour later.
    """
    if settings.tokenizer is TokenizerKind.BGE_M3:
        return BgeTokenCounter(settings.tokenizer_path)
    return HeuristicTokenCounter()
