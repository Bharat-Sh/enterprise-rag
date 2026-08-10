"""`rag-admin` — the bootstrap and break-glass command line.

A fresh deployment has no tenant and no user, and from M2 every endpoint
requires a credential. Something has to create the first one.

**Why a CLI and not a `POST /bootstrap` endpoint.** An unauthenticated
tenant-creating endpoint is a permanent liability guarded, at best, by a shared
secret that is one empty environment variable away from being nothing — and it
stays reachable long after the single moment it was needed. A CLI runs where the
database credentials already are, which is exactly the authority the operation
requires and no more.

Deliberately small: create a tenant, create a user, set a password, generate a
signing key. Anything an operator does more than once belongs in the API behind
a role.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa

from rag.adapters.auth.passwords import Argon2PasswordHasher
from rag.core.config import SigningAlgorithm, get_settings
from rag.db.session import create_engine, create_session_factory
from rag.db.uow import SqlAlchemyUnitOfWork
from rag.domain.enums import Role, UserStatus

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag.core.config import Settings

__all__ = ["main", "run"]

_RSA_KEY_SIZE = 2048


async def _create_tenant(settings: Settings, *, slug: str, name: str) -> None:
    engine = create_engine(settings)
    try:
        async with SqlAlchemyUnitOfWork(create_session_factory(engine)) as uow:
            if await uow.tenants.get_by_slug(slug) is not None:
                print(f"Tenant {slug!r} already exists.", file=sys.stderr)
                raise SystemExit(1)
            tenant = await uow.tenants.create(slug=slug, name=name)
            await uow.commit()
            print(f"Created tenant {tenant.slug} ({tenant.id})")
    finally:
        await engine.dispose()


async def _create_user(
    settings: Settings, *, tenant_slug: str, email: str, full_name: str, role: Role, password: str
) -> None:
    hasher = Argon2PasswordHasher(settings.auth)
    engine = create_engine(settings)
    try:
        async with SqlAlchemyUnitOfWork(create_session_factory(engine)) as uow:
            tenant = await uow.tenants.get_by_slug(tenant_slug)
            if tenant is None:
                print(f"No such tenant: {tenant_slug!r}", file=sys.stderr)
                raise SystemExit(1)

            # Scope before touching `users` — it is under row-level security,
            # and an unscoped session would see nothing and then fail the
            # insert's WITH CHECK. The CLI obeys the same rule as the API.
            await uow.scope_to_tenant(tenant.id)

            if await uow.users.get_by_email(email) is not None:
                print(f"User {email!r} already exists in {tenant_slug!r}.", file=sys.stderr)
                raise SystemExit(1)

            user = await uow.users.create(
                tenant_id=tenant.id,
                email=email,
                full_name=full_name,
                role=role,
                # ACTIVE, not INVITED: a bootstrap user that cannot log in has
                # not bootstrapped anything.
                status=UserStatus.ACTIVE,
                password_hash=await hasher.hash(password),
            )
            await uow.commit()
            print(f"Created {role.value} {user.email} ({user.id}) in {tenant.slug}")
    finally:
        await engine.dispose()


async def _set_password(settings: Settings, *, tenant_slug: str, email: str, password: str) -> None:
    hasher = Argon2PasswordHasher(settings.auth)
    engine = create_engine(settings)
    try:
        async with SqlAlchemyUnitOfWork(create_session_factory(engine)) as uow:
            tenant = await uow.tenants.get_by_slug(tenant_slug)
            if tenant is None:
                print(f"No such tenant: {tenant_slug!r}", file=sys.stderr)
                raise SystemExit(1)
            await uow.scope_to_tenant(tenant.id)

            user = await uow.users.get_by_email(email)
            if user is None:
                print(f"No such user: {email!r}", file=sys.stderr)
                raise SystemExit(1)

            await uow.users.set_password_hash(user.id, await hasher.hash(password))
            # Everything issued before now stops working. A password reset that
            # leaves the old sessions alive has reset nothing.
            now = datetime.now(UTC)
            await uow.users.invalidate_tokens_before(user.id, at=now)
            await uow.refresh_tokens.revoke_for_user(user.id, at=now)
            await uow.commit()
            print(f"Password updated for {email}; all sessions revoked.")
    finally:
        await engine.dispose()


def _generate_key(algorithm: SigningAlgorithm) -> None:
    """Print a PEM private key for `RAG_AUTH__PRIVATE_KEY_PEM`.

    Written to stdout and nowhere else. Choosing a file path for someone else's
    secret is how a key ends up world-readable in a repository.
    """
    key: ed25519.Ed25519PrivateKey | rsa.RSAPrivateKey = (
        ed25519.Ed25519PrivateKey.generate()
        if algorithm is SigningAlgorithm.EDDSA
        else rsa.generate_private_key(public_exponent=65537, key_size=_RSA_KEY_SIZE)
    )
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    sys.stdout.write(pem.decode("ascii"))


def _read_password(supplied: str | None) -> str:
    """Prompt when no password was given on the command line.

    Prompting is the default because an argument lands in shell history and in
    the process table, where any other user on the box can read it.
    """
    if supplied is not None:
        return supplied
    password = getpass.getpass("Password: ")
    if password != getpass.getpass("Repeat: "):
        print("Passwords do not match.", file=sys.stderr)
        raise SystemExit(1)
    return password


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-admin", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    tenant = sub.add_parser("create-tenant", help="Create a tenant.")
    tenant.add_argument("--slug", required=True)
    tenant.add_argument("--name", required=True)

    user = sub.add_parser("create-user", help="Create a user with a password.")
    user.add_argument("--tenant", required=True)
    user.add_argument("--email", required=True)
    user.add_argument("--full-name", default="")
    user.add_argument("--role", choices=[role.value for role in Role], default=Role.OWNER.value)
    user.add_argument("--password", default=None, help="Prompted for if omitted.")

    password = sub.add_parser("set-password", help="Reset a password and revoke all sessions.")
    password.add_argument("--tenant", required=True)
    password.add_argument("--email", required=True)
    password.add_argument("--password", default=None, help="Prompted for if omitted.")

    key = sub.add_parser("generate-key", help="Print a new signing key as PEM.")
    key.add_argument(
        "--algorithm",
        choices=[alg.value for alg in SigningAlgorithm],
        default=SigningAlgorithm.EDDSA.value,
    )

    return parser


async def run(argv: Sequence[str] | None = None) -> int:
    """The command line's actual behaviour, as a coroutine.

    Separate from `main` so the test suite can await it from inside its own
    event loop. `asyncio.run` cannot be called from a running loop, so a CLI
    whose only entry point starts a loop is only testable by spawning a process
    — and a bootstrap path that is awkward to test is a bootstrap path that
    quietly rots.
    """
    args = _build_parser().parse_args(argv)

    if args.command == "generate-key":
        # The only subcommand that touches neither the database nor `Settings`,
        # so it works before anything is configured — which is exactly when an
        # operator needs a key.
        _generate_key(SigningAlgorithm(args.algorithm))
        return 0

    settings = get_settings()

    if args.command == "create-tenant":
        await _create_tenant(settings, slug=args.slug, name=args.name)
    elif args.command == "create-user":
        await _create_user(
            settings,
            tenant_slug=args.tenant,
            email=args.email,
            full_name=args.full_name,
            role=Role(args.role),
            password=_read_password(args.password),
        )
    elif args.command == "set-password":
        await _set_password(
            settings,
            tenant_slug=args.tenant,
            email=args.email,
            password=_read_password(args.password),
        )

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point. Owns the event loop and nothing else."""
    return asyncio.run(run(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
