"""Concrete authentication adapters.

The only place `jwt`, `cryptography`, and `argon2` are imported. `rag.domain`
and `rag.services` are forbidden from importing them by contract — see the
import-linter section of `pyproject.toml`.
"""

from __future__ import annotations
