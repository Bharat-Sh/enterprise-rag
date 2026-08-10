"""Blob storage adapters (docs/adr/0009).

Filesystem today; an S3-compatible adapter behind the same `BlobStore` port at
deployment, changing no call site.
"""

from __future__ import annotations
