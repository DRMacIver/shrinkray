"""Vendored copy of pytest-textual-snapshot with syrupy 5.0 compatibility.

Source: https://github.com/Textualize/pytest-textual-snapshot

The upstream package uses `_file_extension = "svg"` but syrupy 5.0 changed
the attribute to `file_extension` (no underscore), causing snapshots to be
saved as .raw files instead of .svg files.

This vendored copy fixes the attribute name.

---

The MIT License (MIT)

Copyright (c) 2023 Textualize

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from __future__ import annotations

import inspect
import os
import pickle
import re
import shutil
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from operator import attrgetter
from os import PathLike
from pathlib import Path, PurePath
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any, cast

import pytest
from _pytest.config import ExitCode
from _pytest.fixtures import FixtureRequest
from _pytest.main import Session
from _pytest.terminal import TerminalReporter
from jinja2 import Template
from rich.console import Console
from syrupy.assertion import SnapshotAssertion
from syrupy.extensions.single_file import SingleFileSnapshotExtension, WriteMode
from textual.app import App


if TYPE_CHECKING:
    from _pytest.nodes import Item
    from textual.pilot import Pilot


class SVGImageExtension(SingleFileSnapshotExtension):
    # syrupy 5.0 changed from _file_extension to file_extension
    file_extension = "svg"
    _write_mode = WriteMode.TEXT


class TemporaryDirectory:
    """A temporary that survives forking.

    This provides something akin to tempfile.TemporaryDirectory, but this
    version is not removed automatically when a process exits.
    """

    def __init__(self, name: str = ""):
        if name:
            self.name = name
        else:
            self.name = mkdtemp(None, None, None)

    def cleanup(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.name, ignore_errors=True)


@dataclass
class PseudoConsole:
    """Something that looks enough like a Console to fill a Jinja2 template."""

    legacy_windows: bool
    size: tuple[int, int]


@dataclass
class PseudoApp:
    """Something that looks enough like an App to fill a Jinja2 template.

    This can be pickled OK, whereas the 'real' application involved in a test
    may contain unpickleable data.
    """

    console: PseudoConsole


def rename_styles(svg: str, suffix: str) -> str:
    """Rename style names to prevent clashes when combined in HTML report."""
    return re.sub(r"terminal-(\d+)-r(\d+)", rf"terminal-\1-r\2-{suffix}", svg)


def pytest_addoption(parser):
    parser.addoption(
        "--snapshot-report",
        action="store",
        default="snapshot_report.html",
        help="Snapshot test output HTML path.",
    )


_app_stash_key: pytest.StashKey[App] | None = None


def app_stash_key() -> pytest.StashKey[App]:
    global _app_stash_key
    if _app_stash_key is None:
        _app_stash_key = pytest.StashKey[App]()
    return _app_stash_key


def node_to_report_path(node: Item) -> Path:
    """Generate a report file name for a test node."""
    tempdir = get_tempdir()
    path_raw, _, name = node.reportinfo()
    path = Path(path_raw)
    temp = path.parent
    base = []
    while temp != temp.parent and temp.name != "tests":
        base.append(temp.name)
        temp = temp.parent
    parts = []
    if base:
        parts.append("_".join(reversed(base)))
    parts.append(path.name.replace(".", "_"))
    parts.append(name.replace("[", "_").replace("]", "_"))
    return Path(tempdir.name) / "_".join(parts)


@pytest.fixture
def snap_compare(
    snapshot: SnapshotAssertion, request: FixtureRequest
) -> Callable[[str | PurePath], bool]:
    """
    This fixture returns a function which can be used to compare the output of a Textual
    app with the output of the same app in the past. This is snapshot testing, and it
    used to catch regressions in output.
    """
    # Switch so one file per snapshot, stored as plain simple SVG file.
    snapshot = snapshot.use_extension(SVGImageExtension)

    def compare(
        app: str | PurePath | App[Any],
        press: Iterable[str] = (),
        terminal_size: tuple[int, int] = (80, 24),
        run_before: Callable[[Pilot], Awaitable[None] | None] | None = None,
    ) -> bool:
        """
        Compare a current screenshot of the app running at app_path, with
        a previously accepted (validated by human) snapshot stored on disk.
        When the `--snapshot-update` flag is supplied (provided by syrupy),
        the snapshot on disk will be updated to match the current screenshot.

        Args:
            app (str): An `App` instance or the path to an App. Relative paths are relative to the location of the
                test this function is called from. If the path contains an App, that file should *not* call `App.run`
                itself, as this is done automatically by this function.
            press (Iterable[str]): Key presses to run before taking screenshot. "_" is a short pause.
            terminal_size (tuple[int, int]): A pair of integers (WIDTH, HEIGHT), representing terminal size.
            run_before: An arbitrary callable that runs arbitrary code before taking the
                screenshot. Use this to simulate complex user interactions with the app
                that cannot be simulated by key presses.

        Returns:
            Whether the screenshot matches the snapshot.
        """
        from textual._import_app import import_app

        node = request.node

        if isinstance(app, App):
            app_instance = app
            app_path = ""
        else:
            path = Path(app)
            if path.is_absolute():
                # If the user supplies an absolute path, just use it directly.
                app_path = str(path.resolve())
                app_instance = import_app(app_path)
            else:
                # If a relative path is supplied by the user, it's relative to the location of the pytest node,
                # NOT the location that `pytest` was invoked from.
                node_path = node.path.parent
                resolved = (node_path / app).resolve()
                app_path = str(resolved)
                app_instance = import_app(app_path)

        from textual._doc import take_svg_screenshot

        actual_screenshot = take_svg_screenshot(
            app=app_instance,
            press=press,
            terminal_size=terminal_size,
            run_before=run_before,
        )
        console = Console(legacy_windows=False, force_terminal=True)
        p_app = PseudoApp(PseudoConsole(console.legacy_windows, console.size))

        result = snapshot == actual_screenshot

        # This code must come below the comparison above, as it uses data generated by the comparison.
        execution_index = (
            snapshot._custom_index
            and snapshot._execution_name_index.get(snapshot._custom_index)
        ) or snapshot.num_executions - 1
        assertion_result = snapshot.executions.get(execution_index)

        snapshot_exists = (
            execution_index in snapshot.executions
            and assertion_result
            and assertion_result.final_data is not None
        )

        expected_svg_text = str(snapshot)
        full_path, line_number, name = node.reportinfo()

        data = (
            result,
            expected_svg_text,
            actual_screenshot,
            p_app,
            full_path,
            line_number,
            name,
            inspect.getdoc(node.function) or "",
            app_path,
            snapshot_exists,
        )
        data_path = node_to_report_path(request.node)
        data_path.write_bytes(pickle.dumps(data))

        return result

    return compare


@dataclass
class SvgSnapshotDiff:
    """Model representing a diff between current screenshot of an app,
    and the snapshot on disk. This is ultimately intended to be used in
    a Jinja2 template."""

    snapshot: str | None
    actual: str | None
    test_name: str
    path: PathLike
    line_number: int
    app: App[Any]
    """The app instance which was tested."""
    environment: dict[str, str]
    """The environment variables from the host which ran the test."""
    docstring: str
    """If the underlying test functions contains a docstring, we'll include it in the test report."""
    app_path: Path | None
    """If the app was loaded from a path, we'll include that path in the test report for easier access.
    This will be None if an App instance was directly passed to `snap_compare`."""
    snapshot_exists: bool
    """True if the there was a snapshot available to compare the test output to, otherwise False."""


def pytest_sessionstart(
    session: Session,
) -> None:
    """Set up a temporary directory to store snapshots.

    The temporary directory name is stored in an environment vairable so that
    pytest-xdist worker child processes can retrieve it.
    """
    if os.environ.get("PYTEST_XDIST_WORKER") is None:
        tempdir = TemporaryDirectory()
        os.environ["TEXTUAL_SNAPSHOT_TEMPDIR"] = tempdir.name


def get_tempdir():
    """Get the TemporaryDirectory."""
    return TemporaryDirectory(os.environ["TEXTUAL_SNAPSHOT_TEMPDIR"])


def pytest_sessionfinish_snapshot_report(
    session: Session,
    exitstatus: int | ExitCode,
) -> None:
    """Called after whole test run finished, right before returning the exit status to the system.
    Generates the snapshot report and writes it to disk.

    Note: This is renamed from pytest_sessionfinish to avoid conflicts with
    the hook in test_tui_snapshots.py. It's called from the conftest.py hook.
    """
    if os.environ.get("PYTEST_XDIST_WORKER") is None:
        tempdir = get_tempdir()
        diffs, num_snapshots_passing = retrieve_svg_diffs(tempdir)
        save_svg_diffs(diffs, session, num_snapshots_passing)
        tempdir.cleanup()


def retrieve_svg_diffs(
    tempdir: TemporaryDirectory,
) -> tuple[list[SvgSnapshotDiff], int]:
    """Retrieve snapshot diffs from the temporary directory."""
    diffs: list[SvgSnapshotDiff] = []
    pass_count = 0

    n = 0
    for data_path in Path(tempdir.name).iterdir():
        (
            passed,
            expect_svg_text,
            svg_text,
            app,
            full_path,
            line_index,
            name,
            docstring,
            app_path,
            snapshot_exists,
        ) = pickle.loads(data_path.read_bytes())
        pass_count += 1 if passed else 0
        if not passed:
            n += 1
            diffs.append(
                SvgSnapshotDiff(
                    snapshot=rename_styles(str(expect_svg_text), f"exp{n}"),
                    actual=rename_styles(svg_text, f"act{n}"),
                    test_name=name,
                    path=full_path,
                    line_number=line_index + 1,
                    app=app,
                    environment=dict(os.environ),
                    docstring=docstring,
                    app_path=Path(app_path) if app_path else None,
                    snapshot_exists=snapshot_exists,
                )
            )
    return diffs, pass_count


def save_svg_diffs(
    diffs: list[SvgSnapshotDiff],
    session: Session,
    num_snapshots_passing: int,
) -> None:
    """Save any detected differences to an HTML formatted report."""
    if diffs:
        diff_sort_key = attrgetter("test_name")
        diffs = sorted(diffs, key=diff_sort_key)

        # Use the upstream package's template (we depend on it for this resource)
        import pytest_textual_snapshot as upstream

        snapshot_template_path = (
            Path(upstream.__file__).parent
            / "resources"
            / "snapshot_report_template.jinja2"
        )

        snapshot_report_path = session.config.getoption("--snapshot-report")
        snapshot_report_path = Path(snapshot_report_path)
        snapshot_report_path = Path.cwd() / snapshot_report_path
        snapshot_report_path.parent.mkdir(parents=True, exist_ok=True)
        template = Template(snapshot_template_path.read_text())

        num_fails = len(diffs)
        num_snapshot_tests = len(diffs) + num_snapshots_passing

        rendered_report = template.render(
            diffs=diffs,
            passes=num_snapshots_passing,
            fails=num_fails,
            pass_percentage=100 * (num_snapshots_passing / max(num_snapshot_tests, 1)),
            fail_percentage=100 * (num_fails / max(num_snapshot_tests, 1)),
            num_snapshot_tests=num_snapshot_tests,
            now=datetime.now(UTC),
            file_open_prefix=os.getenv("TEXTUAL_SNAPSHOT_FILE_OPEN_PREFIX", "file://"),
        )
        with open(snapshot_report_path, "w+", encoding="utf-8") as snapshot_file:
            snapshot_file.write(rendered_report)

        # Store on config for use in pytest_terminal_summary
        # Cast to Any to allow setting private attributes
        cfg = cast(Any, session.config)
        cfg._textual_snapshots = diffs
        cfg._textual_snapshot_html_report = snapshot_report_path


def pytest_terminal_summary(
    terminalreporter: TerminalReporter,
    exitstatus: ExitCode,
    config: pytest.Config,
) -> None:
    """Add a section to terminal summary reporting.
    Displays the link to the snapshot report that was generated in a prior hook.
    """
    if os.environ.get("PYTEST_XDIST_WORKER") is None:
        cfg = cast(Any, config)
        diffs = cfg._textual_snapshots if hasattr(config, "_textual_snapshots") else None
        console = Console(legacy_windows=False, force_terminal=True)
        if diffs:
            snapshot_report_location = cfg._textual_snapshot_html_report
            console.print("[b red]Textual Snapshot Report", style="red")
            console.print(
                f"\n[black on red]{len(diffs)} mismatched snapshots[/]\n"
                f"\n[b]View the [link=file://{snapshot_report_location}]failure report[/].\n"
            )
            console.print(f"[dim]{snapshot_report_location}\n")
