"""Fail CI if the integration or security suites did not actually run.

Both suites skip themselves when no database is reachable. That is right on a
developer laptop and unacceptable in CI, where the database is a service
container that is supposed to be up — a silently skipped isolation suite turns a
data-leak regression into a green build. CLAUDE.md additionally requires that
`tests/security` never be skipped at all.

Reads the JUnit XML produced by the main test run rather than re-running
anything. Two reasons: the assertion then covers the *same* run that produced
the coverage report, and CI does not pay twice for the slowest suite.

Why not grep the output for "N passed": pytest omits that summary line under
`-q` when everything passes, so the grep matched nothing and the check reported
failure on a perfectly good run. Counting elements is not sensitive to how
pytest chooses to phrase itself.

M4 added a narrow exception mechanism — see `ALLOWED_SKIPS`. It exists because
one assertion genuinely cannot be made without a GPU, and a runner without one
would otherwise force the choice between deleting the assertion and weakening
this check for everything.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

#: Suites that must run, keyed by the `classname` prefix pytest gives them.
REQUIRED: dict[str, str] = {
    "tests.integration": "integration",
    "tests.security": "security",
}

#: Tests permitted to skip, keyed by `classname::name`, each with the reason.
#:
#: An allowlist rather than a relaxed rule, because "no skips" is what makes the
#: check worth having and every exception should cost someone a line of
#: justification here. M4 needed exactly one: an assertion that can only be made
#: against a real GPU backend, on a runner that has no GPU (docs/adr/0011).
#:
#: Entries are also checked for *staleness* below. An allowlist that quietly
#: accumulates ids for tests that no longer exist stops being a list of known
#: exceptions and becomes a list of things nobody has looked at.
ALLOWED_SKIPS: dict[str, str] = {
    "tests.integration.test_model_service.TestTheTokenizersAgree"
    "::test_a_real_service_shares_the_workers_vocabulary": (
        "compares the model service's published tokenizer fingerprint against "
        "the worker's; needs the GPU backend, and CI runs the stub"
    ),
}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} <junit.xml>", file=sys.stderr)
        return 2

    report = Path(argv[1])
    if not report.is_file():
        print(f"::error::{report} does not exist — did the test step run?")
        return 1

    # S314 flags stdlib XML parsing of *untrusted* input. This file is written
    # by pytest moments earlier in the same job; if an attacker can choose its
    # contents they already run arbitrary code here. Pulling in `defusedxml` for
    # a CI helper would be a dependency bought with nothing.
    cases = ET.parse(report).getroot().iter("testcase")  # noqa: S314

    totals: dict[str, int] = dict.fromkeys(REQUIRED, 0)
    unexpected_skips: dict[str, list[str]] = {prefix: [] for prefix in REQUIRED}
    seen_ids: set[str] = set()

    for case in cases:
        classname = case.get("classname", "")
        test_id = f"{classname}::{case.get('name', '')}"
        seen_ids.add(test_id)
        for prefix in REQUIRED:
            if classname.startswith(prefix):
                totals[prefix] += 1
                if case.find("skipped") is not None and test_id not in ALLOWED_SKIPS:
                    unexpected_skips[prefix].append(test_id)
                break

    failed = False
    for prefix, name in REQUIRED.items():
        total, skips = totals[prefix], unexpected_skips[prefix]
        print(f"{name}: {total} tests, {len(skips)} unexpected skips")

        if total == 0:
            print(
                f"::error::The {name} suite collected no tests. Either it was "
                f"not run, or collection failed silently."
            )
            failed = True
        elif skips:
            print(
                f"::error::The {name} suite skipped {len(skips)} test(s) — almost "
                f"certainly an unreachable dependency. A skipped isolation suite "
                f"is a green build that verified nothing."
            )
            for test_id in skips:
                print(f"::error::  skipped: {test_id}")
            failed = True

    # A stale allowlist is how a deliberate exception rots into an unnoticed
    # one. If an id no longer exists the entry has outlived its justification
    # and someone has to decide whether it is still needed.
    stale = sorted(set(ALLOWED_SKIPS) - seen_ids)
    if stale:
        for test_id in stale:
            print(f"::error::ALLOWED_SKIPS names a test that no longer exists: {test_id}")
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
