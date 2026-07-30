"""Persistence: SQLAlchemy models, repositories, session and unit-of-work.

Separate from `rag.adapters` because Postgres is not a swappable detail here —
it is the system of record (docs/adr/0001), and the repositories deliberately
use relational features that a generic port would have to abstract away.

Populated in M1.
"""
