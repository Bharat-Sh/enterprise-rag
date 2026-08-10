"""Argon2id password hashing.

Cost parameters are turned down to the configured minimum throughout. These
tests assert *behaviour* — that a wrong password fails, that a missing hash
still does the work, that raising cost triggers a rehash — none of which depends
on how expensive the hash is, and all of which would take a minute per run at
production settings.
"""

from __future__ import annotations

import argon2
import pytest

from rag.adapters.auth.passwords import Argon2PasswordHasher
from rag.core.config import AuthSettings

CHEAP = AuthSettings(argon2_time_cost=1, argon2_memory_cost_kib=8192, argon2_parallelism=1)


@pytest.fixture
def hasher() -> Argon2PasswordHasher:
    return Argon2PasswordHasher(CHEAP)


class TestHashAndVerify:
    async def test_a_correct_password_verifies(self, hasher: Argon2PasswordHasher) -> None:
        stored = await hasher.hash("correct-horse-battery-staple")

        assert await hasher.verify(stored, "correct-horse-battery-staple") is True

    async def test_a_wrong_password_does_not(self, hasher: Argon2PasswordHasher) -> None:
        stored = await hasher.hash("correct-horse-battery-staple")

        assert await hasher.verify(stored, "correct-horse-battery-stapler") is False

    async def test_the_same_password_hashes_differently_every_time(
        self, hasher: Argon2PasswordHasher
    ) -> None:
        # Per-hash salt. Without it, identical passwords are visibly identical
        # in a stolen dump, and one cracked hash cracks every account sharing it.
        first = await hasher.hash("same-password-twice")
        second = await hasher.hash("same-password-twice")

        assert first != second
        assert await hasher.verify(first, "same-password-twice") is True
        assert await hasher.verify(second, "same-password-twice") is True

    async def test_the_stored_hash_does_not_contain_the_password(
        self, hasher: Argon2PasswordHasher
    ) -> None:
        stored = await hasher.hash("plaintext-should-not-appear")

        assert "plaintext-should-not-appear" not in stored

    async def test_argon2id_is_the_variant_used(self, hasher: Argon2PasswordHasher) -> None:
        # Not argon2i (weaker against time-memory trade-offs) and not argon2d
        # (weaker against side channels). The hybrid is the OWASP recommendation.
        assert (await hasher.hash("anything")).startswith("$argon2id$")

    async def test_long_passphrases_are_not_truncated(self, hasher: Argon2PasswordHasher) -> None:
        # The concrete reason bcrypt was rejected: it silently ignores
        # everything past 72 bytes, so these two would be the same password.
        base = "a" * 72
        stored = await hasher.hash(base + "distinct-tail")

        assert await hasher.verify(stored, base + "different-tail") is False
        assert await hasher.verify(stored, base + "distinct-tail") is True


class TestMissingHash:
    """A user with no password must not be distinguishable by timing or result."""

    async def test_a_none_hash_never_verifies(self, hasher: Argon2PasswordHasher) -> None:
        assert await hasher.verify(None, "any-password") is False

    async def test_a_none_hash_still_does_the_work(
        self, hasher: Argon2PasswordHasher, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Asserted on the code path, not the wall clock.

        The property under test is that "this account has no password" costs the
        same as "wrong password" — otherwise a stopwatch tells an attacker which
        email addresses exist. Timing it directly would be a flaky test wearing a
        security badge, so the spy goes on argon2's own `verify`: a real
        verification against a real argon2id hash is what makes the two paths
        cost the same, and its occurrence is observable without measuring
        anything.
        """
        verified: list[str] = []
        original = argon2.PasswordHasher.verify

        def spy(self: argon2.PasswordHasher, stored: str, password: str) -> bool:
            verified.append(stored)
            return original(self, stored, password)

        monkeypatch.setattr(argon2.PasswordHasher, "verify", spy)

        assert await hasher.verify(None, "any-password") is False

        assert len(verified) == 1, "the missing-hash path must not short-circuit"
        assert verified[0].startswith("$argon2id$")

    async def test_an_unparseable_hash_is_refused_rather_than_raising(
        self, hasher: Argon2PasswordHasher
    ) -> None:
        # A corrupted row makes an account unusable; it must not make the login
        # endpoint return 500 for everyone who hits that user.
        assert await hasher.verify("not-an-argon2-hash", "whatever") is False


class TestRehash:
    async def test_a_hash_at_current_parameters_needs_no_rehash(
        self, hasher: Argon2PasswordHasher
    ) -> None:
        assert hasher.needs_rehash(await hasher.hash("password-one")) is False

    async def test_raising_the_cost_marks_old_hashes_for_rehash(self) -> None:
        # Without this, raising cost protects only accounts created afterwards,
        # and the oldest passwords — the ones most worth protecting — keep their
        # original parameters for ever.
        weak = Argon2PasswordHasher(CHEAP)
        stronger = Argon2PasswordHasher(
            AuthSettings(argon2_time_cost=3, argon2_memory_cost_kib=8192, argon2_parallelism=1)
        )

        old = await weak.hash("password-one")

        assert stronger.needs_rehash(old) is True
        # And it must still verify, or every user is locked out on deploy day.
        assert await stronger.verify(old, "password-one") is True

    def test_an_unparseable_hash_is_marked_for_rehash(self, hasher: Argon2PasswordHasher) -> None:
        assert hasher.needs_rehash("$not$argon2$at$all") is True
