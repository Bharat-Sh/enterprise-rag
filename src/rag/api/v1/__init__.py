"""Version 1 of the public HTTP API.

Versioning by URL prefix (`/api/v1`) rather than by header: it is visible in
logs, trivially routable at a proxy, and easy to curl. When v2 arrives, both
packages coexist and share the same service layer underneath.
"""
