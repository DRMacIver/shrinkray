#!/usr/bin/env python3
"""Enforce that source-affecting changes ship a RELEASE.md changelog entry.

Run on pull requests (and locally via ``just check-release``). If the change
touches ``src/`` or ``pyproject.toml`` but there is no ``RELEASE.md`` at the
repository root, this exits non-zero.

See the "Changelog" section of CLAUDE.md and the writing-a-changelog-entry skill
for how to write the entry.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def touches_source(files: list[str]) -> bool:
    """True if any changed path affects the published package."""
    return any(f == "pyproject.toml" or f.startswith("src/") for f in files)


def needs_release_file(files: list[str], release_exists: bool) -> bool:
    """True if these changes require a RELEASE.md but it is missing."""
    return touches_source(files) and not release_exists


def changed_files(base: str) -> list[str]:
    """Names of files changed between the merge-base of ``base``/HEAD and HEAD."""
    merge_base = subprocess.run(
        ["git", "merge-base", base, "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--name-only", f"{merge_base}..HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line for line in diff.splitlines() if line]


def main(argv: list[str]) -> int:
    base = argv[1] if len(argv) > 1 else "main"
    files = changed_files(base)
    if needs_release_file(files, (ROOT / "RELEASE.md").exists()):
        print(
            "ERROR: this change modifies src/ or pyproject.toml but has no "
            "RELEASE.md.\n"
            "Add a RELEASE.md at the repository root describing the user-visible "
            "effect of the change (see the writing-a-changelog-entry skill). It "
            "will be folded into CHANGELOG.md at release time.",
            file=sys.stderr,
        )
        return 1
    print("RELEASE.md requirement satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
