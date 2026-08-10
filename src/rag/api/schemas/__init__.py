"""Request and response models shared across routers.

Routers have owned their own Pydantic models until now, which is right while a
model has one call site. The auth models are used by two routers and by the
tests that assert the contract, so they live here instead of being imported
across router modules — a router importing another router's schema is the first
step towards two routers importing each other.
"""

from __future__ import annotations
