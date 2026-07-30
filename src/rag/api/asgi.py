"""ASGI entry point: ``uvicorn rag.api.asgi:app``.

This module exists solely so that building the default application is an
explicit act rather than an import side effect.

If the module-level ``app = create_app()`` lived in `rag.api.main`, then merely
importing `create_app` — which the test suite does — would construct a full
application from the ambient environment and the developer's local `.env`. Tests
would stop being hermetic, and a `.env` containing `RAG_ENVIRONMENT=prod` would
crash test *collection* with a configuration error, which is a genuinely
baffling failure to debug.

Keeping the factory import-safe and the instantiation in its own module costs
one file and removes that entire class of problem.
"""

from __future__ import annotations

from rag.api.main import create_app

app = create_app()
