"""Shared helpers for building test credentials and configurations.

Keys are generated per call rather than checked in. A fixture private key in a
repository is a private key that eventually gets trusted somewhere real, and
every secret scanner would be right to flag it.
"""

from __future__ import annotations

from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa

from rag.core.config import SigningAlgorithm

#: A database password that passes the production guard, so a test can exercise
#: some *other* production rule without tripping the default-credential check.
SAFE_PASSWORD = "not-the-development-default"


def generate_private_pem(algorithm: SigningAlgorithm = SigningAlgorithm.EDDSA) -> str:
    """A fresh PKCS#8 PEM private key. Ed25519 generation is microseconds."""
    key: ed25519.Ed25519PrivateKey | rsa.RSAPrivateKey = (
        ed25519.Ed25519PrivateKey.generate()
        if algorithm is SigningAlgorithm.EDDSA
        else rsa.generate_private_key(public_exponent=65537, key_size=2048)
    )
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")


def public_pem(private_pem: str) -> str:
    """The matching public key, for exercising retired-key verification."""
    private = serialization.load_pem_private_key(private_pem.encode("ascii"), password=None)
    return (
        private.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("ascii")
    )


def production_overrides(**extra: Any) -> dict[str, Any]:
    """Settings that satisfy every production guard.

    Centralised so that adding a new production requirement means updating one
    helper rather than hunting through the suite — and so a test that *should*
    fail the guard fails for the reason it names, not an unrelated one.
    """
    overrides: dict[str, Any] = {
        "database": {"password": SAFE_PASSWORD},
        "auth": {
            "private_key_pem": generate_private_pem(),
            "issuer": "https://rag.example.test",
        },
    }
    overrides.update(extra)
    return overrides
