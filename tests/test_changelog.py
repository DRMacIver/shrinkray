"""Tests for the changelog assembly and RELEASE.md gate used at release time.

These modules live under ``scripts/`` (they are release tooling, not part of the
shrinkray package), so they are loaded by path rather than imported.
"""

import importlib.util
from pathlib import Path


_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


changelog = _load("changelog")
check_release_file = _load("check_release_file")


# === render_entry ===


def test_render_entry_formats_heading_and_body():
    entry = changelog.render_entry("26.7.5.0", "2026-07-05", "- Did a thing.\n")
    assert entry == "## 26.7.5.0 — 2026-07-05\n\n- Did a thing.\n"


def test_render_entry_strips_surrounding_whitespace():
    entry = changelog.render_entry("1.0", "2026-01-01", "\n\n- x\n\n")
    assert entry == "## 1.0 — 2026-01-01\n\n- x\n"


# === prepend_entry ===


def test_prepend_entry_inserts_above_latest_release():
    existing = "# Changelog\n\n## 1.0 — 2026-01-01\n\n- old\n"
    entry = changelog.render_entry("2.0", "2026-02-02", "- new")
    result = changelog.prepend_entry(existing, entry)
    assert result == (
        "# Changelog\n\n## 2.0 — 2026-02-02\n\n- new\n\n## 1.0 — 2026-01-01\n\n- old\n"
    )
    # newest entry comes first
    assert result.index("2.0") < result.index("1.0")


def test_prepend_entry_appends_when_no_existing_release():
    result = changelog.prepend_entry("# Changelog\n\n", "## 1.0 — d\n\n- x\n")
    assert result == "# Changelog\n\n## 1.0 — d\n\n- x\n"


def test_prepend_entry_handles_missing_trailing_newline_in_head():
    result = changelog.prepend_entry("# Changelog", "## 1.0 — d\n\n- x\n")
    assert result == "# Changelog\n## 1.0 — d\n\n- x\n"


# === consume_release_file ===


def test_consume_release_file_folds_and_deletes(tmp_path):
    (tmp_path / "RELEASE.md").write_text("- A visible change.\n")
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## 1.0 — 2026-01-01\n\n- old\n"
    )
    consumed = changelog.consume_release_file(tmp_path, "2.0", "2026-02-02")
    assert consumed is True
    assert not (tmp_path / "RELEASE.md").exists()
    text = (tmp_path / "CHANGELOG.md").read_text()
    assert "## 2.0 — 2026-02-02" in text
    assert text.index("2.0") < text.index("1.0")
    assert "- A visible change." in text


def test_consume_release_file_creates_changelog_when_missing(tmp_path):
    (tmp_path / "RELEASE.md").write_text("- First entry.\n")
    consumed = changelog.consume_release_file(tmp_path, "1.0", "2026-01-01")
    assert consumed is True
    text = (tmp_path / "CHANGELOG.md").read_text()
    assert text.startswith("# Changelog\n")
    assert "- First entry." in text


def test_consume_release_file_absent_is_noop(tmp_path):
    consumed = changelog.consume_release_file(tmp_path, "1.0", "2026-01-01")
    assert consumed is False
    assert not (tmp_path / "CHANGELOG.md").exists()


# === check_release_file: pure predicates ===


def test_touches_source_detects_src_and_pyproject():
    assert check_release_file.touches_source(["src/shrinkray/problem.py"])
    assert check_release_file.touches_source(["pyproject.toml"])


def test_touches_source_ignores_unrelated_files():
    assert not check_release_file.touches_source(
        ["tests/test_x.py", "CHANGELOG.md", "notes/foo.md", "scripts/release.py"]
    )
    assert not check_release_file.touches_source([])


def test_needs_release_file_true_when_source_changed_without_release():
    assert check_release_file.needs_release_file(["src/shrinkray/x.py"], False)


def test_needs_release_file_false_when_release_present():
    assert not check_release_file.needs_release_file(["src/shrinkray/x.py"], True)


def test_needs_release_file_false_without_source_changes():
    assert not check_release_file.needs_release_file(["tests/test_x.py"], False)
