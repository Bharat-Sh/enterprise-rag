"""Token counting adapters.

An estimator in M3; the real BGE-M3 tokenizer, served by the model service,
in M4.
"""

from __future__ import annotations

from rag.adapters.tokenize.estimator import HeuristicTokenCounter

__all__ = ["HeuristicTokenCounter"]
