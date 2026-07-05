#!/usr/bin/env python3
"""Assemble CHANGELOG.md from per-change RELEASE.md files.

Every change that affects users ships a RELEASE.md at the repository root
describing its user-visible effect (see the writing-a-changelog-entry skill and
the "Changelog" section of CLAUDE.md). At release time the auto-release job calls
``consume_release_file()``: it prepends the RELEASE.md body to CHANGELOG.md under
the new version's heading and deletes RELEASE.md. Because Shrink Ray uses calver,
there is no release-type flag to parse -- the RELEASE.md is just the entry body.

The functions here are pure or operate on explicit paths so they can be unit
tested without a real release (see tests/test_changelog.py).
"""

from __future__ import annotations

from pathlib import Path


CHANGELOG_NAME = "CHANGELOG.md"
RELEASE_NAME = "RELEASE.md"
CHANGELOG_HEADER = "# Changelog\n\n"


def render_entry(version: str, date: str, body: str) -> str:
    """Render a single changelog section for a release.

    ``date`` is an ISO date string (``YYYY-MM-DD``); ``body`` is the raw
    RELEASE.md content.
    """
    return f"## {version} — {date}\n\n{body.strip()}\n"


def prepend_entry(changelog_text: str, entry: str) -> str:
    """Insert ``entry`` above the most recent release section.

    The entry goes immediately before the first ``## `` heading (the newest
    existing release), or at the end if there are none yet, so the file stays
    ordered newest-first.
    """
    lines = changelog_text.splitlines(keepends=True)
    idx = next(
        (i for i, line in enumerate(lines) if line.startswith("## ")),
        len(lines),
    )
    head = "".join(lines[:idx])
    tail = "".join(lines[idx:])
    if head and not head.endswith("\n"):
        head += "\n"
    entry_block = entry.rstrip() + "\n"
    if tail:
        return f"{head}{entry_block}\n{tail}"
    return f"{head}{entry_block}"


def consume_release_file(root: Path, version: str, date: str) -> bool:
    """Fold ``root/RELEASE.md`` into ``root/CHANGELOG.md`` and delete it.

    Returns ``True`` if a RELEASE.md was present and consumed, ``False`` if there
    was nothing to do.
    """
    release_path = root / RELEASE_NAME
    if not release_path.exists():
        return False
    body = release_path.read_text(encoding="utf-8")
    entry = render_entry(version, date, body)
    changelog_path = root / CHANGELOG_NAME
    existing = (
        changelog_path.read_text(encoding="utf-8")
        if changelog_path.exists()
        else CHANGELOG_HEADER
    )
    changelog_path.write_text(prepend_entry(existing, entry), encoding="utf-8")
    release_path.unlink()
    return True
