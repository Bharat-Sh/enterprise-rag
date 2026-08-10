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
    skipped: dict[str, int] = dict.fromkeys(REQUIRED, 0)

    for case in cases:
        classname = case.get("classname", "")
        for prefix in REQUIRED:
            if classname.startswith(prefix):
                totals[prefix] += 1
                if case.find("skipped") is not None:
                    skipped[prefix] += 1
                break

    failed = False
    for prefix, name in REQUIRED.items():
        total, skips = totals[prefix], skipped[prefix]
        print(f"{name}: {total} tests, {skips} skipped")

        if total == 0:
            print(
                f"::error::The {name} suite collected no tests. Either it was "
                f"not run, or collection failed silently."
            )
            failed = True
        elif skips:
            print(
                f"::error::The {name} suite skipped {skips} test(s) — almost "
                f"certainly an unreachable database. A skipped isolation suite "
                f"is a green build that verified nothing."
            )
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
