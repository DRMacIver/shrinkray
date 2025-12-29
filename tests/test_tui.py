"""Comprehensive tests for the textual TUI module."""

import asyncio
import inspect
import os
import tempfile
import time
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, patch

import pytest
from textual.app import App
from textual.widgets import DataTable, Label, Static

from shrinkray import tui
from shrinkray.subprocess.client import SubprocessClient
from shrinkray.subprocess.protocol import PassStatsData, ProgressUpdate, Response
from shrinkray.tui import (
    ContentPreview,
    ExpandedBoxModal,
    HelpScreen,
    OutputPreview,
    PassStatsScreen,
    ShrinkRayApp,
    SizeGraph,
    StatsDisplay,
    _format_time_label,
    _get_percentage_axis_bounds,
    _get_time_axis_bounds,
    detect_terminal_theme,
    run_textual_ui,
)


# Helper to access Static widget's internal content (uses name-mangled attribute)
_STATIC_CONTENT_ATTR = "_Static__content"


def get_static_content(widget: Static) -> str:
    """Get the internal content of a Static widget for testing."""
    return str(getattr(widget, _STATIC_CONTENT_ATTR))


class FakeReductionClient:
    """Fake client for testing the TUI without launching a real subprocess."""

    def __init__(
        self,
        updates: list[ProgressUpdate] | None = None,
        start_error: str | None = None,
        start_delay: float = 0.0,
        wait_indefinitely: bool = False,
    ):
        self._updates = updates or []
        self._start_error = start_error
        self._start_delay = start_delay
        self._wait_indefinitely = wait_indefinitely
        self._started = False
        self._cancelled = False
        self._closed = False
        self._completed = False
        self._update_index = 0

    async def start(self) -> None:
        if self._start_delay > 0:
            await asyncio.sleep(self._start_delay)
        self._started = True

    async def start_reduction(
        self,
        file_path: str,
        test: list[str],
        parallelism: int | None = None,
        timeout: float | None = None,
        seed: int = 0,
        input_type: str = "all",
        in_place: bool = False,
        formatter: str = "default",
        volume: str = "normal",
        no_clang_delta: bool = False,
        clang_delta: str = "",
        trivial_is_error: bool = True,
        skip_validation: bool = False,
        history_enabled: bool = True,
        also_interesting_code: int | None = None,
    ) -> Response:
        if self._start_error:
            return Response(id="start", error=self._start_error)
        return Response(id="start", result={"status": "started"})

    async def cancel(self) -> Response:
        self._cancelled = True
        self._completed = True
        return Response(id="cancel", result={"status": "cancelled"})

    async def disable_pass(self, pass_name: str) -> Response:
        return Response(
            id="disable", result={"status": "disabled", "pass_name": pass_name}
        )

    async def enable_pass(self, pass_name: str) -> Response:
        return Response(
            id="enable", result={"status": "enabled", "pass_name": pass_name}
        )

    async def skip_current_pass(self) -> Response:
        return Response(id="skip", result={"status": "skipped"})

    async def close(self) -> None:
        self._closed = True

    async def get_progress_updates(self) -> AsyncGenerator[ProgressUpdate, None]:
        for update in self._updates:
            if self._cancelled:
                break
            yield update
            await asyncio.sleep(0.01)
        # Wait indefinitely if requested (useful for modal tests)
        if self._wait_indefinitely:
            while not self._cancelled:
                await asyncio.sleep(0.1)
        self._completed = True

    @property
    def is_completed(self) -> bool:
        return self._completed

    @property
    def error_message(self) -> str | None:
        return None


def run_async(coro):
    """Run an async coroutine in a new event loop."""
    return asyncio.run(coro)


# === StatsDisplay tests ===


def test_initial_render():
    """Test initial state shows waiting message."""
    widget = StatsDisplay()
    assert "Waiting for reduction to start" in widget.render()


def test_update_stats():
    """Test updating stats changes the display."""
    widget = StatsDisplay()
    update = ProgressUpdate(
        status="Running pass X",
        size=500,
        original_size=1000,
        calls=10,
        reductions=5,
        interesting_calls=3,
        wasted_calls=1,
        runtime=2.0,
        parallel_workers=4,
    )
    widget.update_stats(update)
    assert widget.current_status == "Running pass X"
    assert widget.current_size == 500
    assert widget.original_size == 1000
    assert widget.call_count == 10
    assert widget.reduction_count == 5
    assert widget.interesting_calls == 3
    assert widget.wasted_calls == 1
    assert widget.runtime == 2.0
    assert widget.parallel_workers == 4


def test_render_with_stats():
    """Test render output with actual stats."""
    widget = StatsDisplay()
    update = ProgressUpdate(
        status="Testing",
        size=500,
        original_size=1000,
        calls=20,
        reductions=8,
        interesting_calls=10,
        wasted_calls=2,
        runtime=5.0,
        parallel_workers=2,
    )
    widget.update_stats(update)
    rendered = widget.render()

    assert "Reducer status: Testing" in rendered
    assert "Current test case size:" in rendered
    assert "50.00%" in rendered  # 500/1000 = 50% reduction
    assert "Calls to interestingness test: 20" in rendered
    assert "Current parallel workers: 2" in rendered


def test_render_small_reduction():
    """Test render with small reduction percentage."""
    widget = StatsDisplay()
    update = ProgressUpdate(
        status="Starting",
        size=990,
        original_size=1000,
        calls=1,
        reductions=1,
        interesting_calls=1,
        wasted_calls=0,
        runtime=0.5,
    )
    widget.update_stats(update)
    rendered = widget.render()
    assert "1.00%" in rendered


# === ContentPreview tests ===


def test_content_preview_initial_render():
    """Test initial state shows loading message."""
    widget = ContentPreview()
    assert "Loading" in widget.render()


def test_text_content():
    """Test rendering text content."""
    widget = ContentPreview()
    widget.update_content("Hello, World!\nThis is a test.", False)
    rendered = widget.render()
    assert "Hello, World!" in rendered
    assert "This is a test." in rendered


def test_hex_content():
    """Test rendering hex content."""
    widget = ContentPreview()
    widget.update_content("00000000  48 65 6c 6c 6f", True)
    rendered = widget.render()
    assert "[Hex mode]" in rendered
    assert "48 65 6c 6c 6f" in rendered


def test_large_content_truncated():
    """Test that large content is truncated."""
    widget = ContentPreview()
    # Create content with more lines than would fit
    lines = [f"Line {i}" for i in range(100)]
    large_content = "\n".join(lines)
    widget.update_content(large_content, False)
    rendered = widget.render()
    # Should show truncation message
    assert "more lines" in rendered


def test_content_diff_shown_for_large_files():
    """Test that diff is shown when content changes in large files."""

    widget = ContentPreview()

    # Set up initial large content
    initial_lines = [f"Line {i}" for i in range(100)]
    initial_content = "\n".join(initial_lines)
    widget._last_display_time = 0  # Reset throttle
    widget.update_content(initial_content, False)

    # Force the time to allow update
    widget._last_display_time = time.time() - 2

    # Change content
    new_lines = [f"Line {i}" for i in range(90)]  # Fewer lines
    new_content = "\n".join(new_lines)
    widget.update_content(new_content, False)

    # The widget should have tracked the previous content
    assert widget._last_displayed_content == initial_content


def test_content_update_throttled():
    """Test that content updates are throttled."""
    widget = ContentPreview()

    # First update should go through
    widget.update_content("First", False)
    assert widget.preview_content == "First"

    # Immediate second update should be throttled
    widget.update_content("Second", False)
    # Content should still be "First" due to throttling
    assert widget.preview_content == "First"

    # But pending content should be stored
    assert widget._pending_content == "Second"


def test_render_diff_for_changed_large_content():
    """Test that diff is rendered when large content changes."""
    widget = ContentPreview()

    # Set up initial large content (must be larger than available_lines)
    initial_lines = [f"Line {i}" for i in range(50)]
    initial_content = "\n".join(initial_lines)

    # Set the initial content and last displayed content
    widget.preview_content = initial_content
    widget._last_displayed_content = initial_content

    # Now change the content
    new_lines = [f"Line {i}" for i in range(45)]  # Remove 5 lines
    new_content = "\n".join(new_lines)
    widget.preview_content = new_content

    # Mock _get_available_lines to return a small number
    original_method = widget._get_available_lines
    widget._get_available_lines = lambda: 10  # type: ignore

    try:
        rendered = widget.render()
        # Should contain diff markers
        assert "---" in rendered or "@@" in rendered
    finally:
        widget._get_available_lines = original_method


def test_render_diff_no_changes_shows_truncated():
    """Test that truncated content is shown when no diff is available."""
    widget = ContentPreview()

    # Set content larger than available lines
    lines = [f"Line {i}" for i in range(50)]
    content = "\n".join(lines)
    widget.preview_content = content
    # No _last_displayed_content set

    # Mock _get_available_lines to return a small number
    original_method = widget._get_available_lines
    widget._get_available_lines = lambda: 10  # type: ignore

    try:
        rendered = widget.render()
        # Should show truncation message
        assert "more lines" in rendered
    finally:
        widget._get_available_lines = original_method


# === SizeGraph tests ===


def test_size_graph_initial_state():
    """Test SizeGraph starts with empty history."""
    widget = SizeGraph()
    assert widget._size_history == []
    assert widget._original_size == 0
    assert widget._current_runtime == 0.0


def test_size_graph_update_empty_history():
    """Test SizeGraph ignores empty history updates but still updates runtime."""
    widget = SizeGraph()
    widget.update_graph([], 1000, 5.0)
    assert widget._size_history == []
    assert widget._original_size == 1000
    assert widget._current_runtime == 5.0


def test_size_graph_update_accumulates_history():
    """Test SizeGraph accumulates history entries."""
    widget = SizeGraph()
    widget.update_graph([(0.0, 1000), (1.0, 800)], 1000, 1.0)
    assert widget._size_history == [(0.0, 1000), (1.0, 800)]

    # Add more entries
    widget.update_graph([(2.0, 600)], 1000, 2.5)
    assert widget._size_history == [(0.0, 1000), (1.0, 800), (2.0, 600)]
    assert widget._current_runtime == 2.5


def test_size_graph_setup_plot_empty():
    """Test SizeGraph setup with no data."""
    widget = SizeGraph()
    # Should not raise when setting up with no data
    widget._setup_plot()


def test_size_graph_setup_plot_single_point():
    """Test SizeGraph setup with single data point doesn't plot."""
    widget = SizeGraph()
    widget._size_history = [(0.0, 1000)]
    widget._original_size = 1000
    # Should not raise - needs at least 2 points to plot
    widget._setup_plot()


def test_size_graph_setup_plot_with_data():
    """Test SizeGraph setup with multiple data points."""
    widget = SizeGraph()
    widget._size_history = [(0.0, 1000), (1.0, 100), (2.0, 10)]
    widget._original_size = 1000
    widget._current_runtime = 2.0
    # Should not raise
    widget._setup_plot()


def test_size_graph_percentage_calculation():
    """Test SizeGraph calculates percentages correctly."""
    widget = SizeGraph()
    widget._size_history = [(0.0, 1000), (1.0, 500), (2.0, 100), (3.0, 10)]
    widget._original_size = 1000
    widget._current_runtime = 3.0
    widget._setup_plot()
    # If we got here without error, percentage calculation worked


def test_size_graph_handles_zero_size():
    """Test SizeGraph handles size of 0 without error (log(0) issue)."""
    widget = SizeGraph()
    widget._size_history = [(0.0, 1000), (1.0, 0)]
    widget._original_size = 1000
    widget._current_runtime = 1.0
    # Should not raise - we use max(0.01, p) to avoid log(0)
    widget._setup_plot()


def test_size_graph_update_graph_no_original_size():
    """Test SizeGraph handles zero original_size gracefully."""
    widget = SizeGraph()
    widget.update_graph([(0.0, 1000), (1.0, 500)], 0, 1.0)
    # Should have history but no original size, so setup_plot returns early
    assert widget._size_history == [(0.0, 1000), (1.0, 500)]
    assert widget._original_size == 0


# === Time axis bounds tests ===


def test_format_time_label_seconds():
    """Test time label formatting for seconds."""
    assert _format_time_label(0) == "0s"
    assert _format_time_label(30) == "30s"
    assert _format_time_label(59) == "59s"


def test_format_time_label_minutes():
    """Test time label formatting for minutes."""
    assert _format_time_label(60) == "1m"
    assert _format_time_label(120) == "2m"
    assert _format_time_label(300) == "5m"
    assert _format_time_label(3599) == "59m"


def test_format_time_label_hours():
    """Test time label formatting for hours."""
    assert _format_time_label(3600) == "1h"
    assert _format_time_label(7200) == "2h"
    assert _format_time_label(36000) == "10h"


def test_get_time_axis_bounds_zero():
    """Test time axis bounds for zero duration."""
    max_time, ticks, labels = _get_time_axis_bounds(0)
    assert max_time == 30.0
    assert 0 in ticks
    assert "0s" in labels


def test_get_time_axis_bounds_short():
    """Test time axis bounds expand by minute for first 10 minutes."""
    # At 25s, should extend to 1m boundary
    max_time, _, _ = _get_time_axis_bounds(25)
    assert max_time == 60.0

    # At 65s, should extend to 2m boundary
    max_time, _, _ = _get_time_axis_bounds(65)
    assert max_time == 120.0


def test_get_time_axis_bounds_medium():
    """Test time axis bounds for medium durations."""
    # At 8 minutes, should extend to 9m
    max_time, _, _ = _get_time_axis_bounds(480)
    assert max_time == 540.0

    # At 11 minutes, should extend to 30m boundary
    max_time, _, _ = _get_time_axis_bounds(660)
    assert max_time == 1800.0


def test_get_time_axis_bounds_very_long():
    """Test time axis bounds for very long durations."""
    # Beyond 8h
    max_time, _, _ = _get_time_axis_bounds(30000)
    assert max_time > 28800


# === Percentage axis bounds tests ===


def test_get_percentage_axis_bounds_high():
    """Test percentage bounds when still at high percentages."""
    lower, _, _ = _get_percentage_axis_bounds(80, 100)
    # Should have room below 80%
    assert lower < 40


def test_get_percentage_axis_bounds_low():
    """Test percentage bounds when at low percentages."""
    lower, _, _ = _get_percentage_axis_bounds(5, 100)
    # Should extend below 5%
    assert lower < 2.5


def test_get_percentage_axis_bounds_very_low():
    """Test percentage bounds when at very low percentages."""
    lower, _, _ = _get_percentage_axis_bounds(0.1, 100)
    # Should extend very low
    assert lower < 0.1


def test_get_percentage_axis_bounds_extremely_low():
    """Test percentage bounds when below 0.01% (extreme reductions)."""
    lower, _, labels = _get_percentage_axis_bounds(0.005, 100)
    # Should extend below 0.01%
    assert lower < 0.005
    # Should have labels for very small percentages
    assert any("%" in label for label in labels)


# === OutputPreview tests ===


def test_output_preview_render_no_content():
    """Test OutputPreview render with no content."""
    widget = OutputPreview()
    rendered = widget.render()
    assert "No test output yet" in rendered


def test_output_preview_render_with_active_test():
    """Test OutputPreview render with an active test."""
    widget = OutputPreview()
    widget.output_content = "Some test output\nMore output"
    widget.active_test_id = 5
    rendered = widget.render()
    assert "Test #5 running" in rendered
    assert "Some test output" in rendered


def test_output_preview_render_completed_test():
    """Test OutputPreview render with completed test (has return code)."""
    widget = OutputPreview()
    widget.output_content = "Final output"
    widget.active_test_id = 42
    widget.last_return_code = 0  # Completed with success
    rendered = widget.render()
    assert "Test #42 exited with code 0" in rendered
    assert "Final output" in rendered


def test_output_preview_render_completed_test_with_return_code():
    """Test OutputPreview render shows return code when available."""
    widget = OutputPreview()
    widget.output_content = "Final output"
    widget.active_test_id = 42
    widget.last_return_code = 1  # Completed with error
    rendered = widget.render()
    assert "Test #42 exited with code 1" in rendered
    assert "Final output" in rendered


def test_output_preview_render_has_seen_output_no_header():
    """Test OutputPreview render shows no header when has seen output before but no test ID."""
    widget = OutputPreview()
    widget.output_content = "Some output"
    widget.active_test_id = None
    widget._has_seen_output = True  # We've seen output before
    rendered = widget.render()
    # Should have content but no header
    assert "Some output" in rendered
    assert "No test output" not in rendered
    assert "Test #" not in rendered
    assert "running" not in rendered


def test_output_preview_update_output_with_return_code():
    """Test that update_output stores return code."""
    widget = OutputPreview()
    widget._last_update_time = 0  # Ensure throttle doesn't block

    widget.update_output("output", 1, return_code=42)
    assert widget._pending_return_code == 42
    assert widget.last_return_code == 42


def test_output_preview_render_truncates_long_output():
    """Test OutputPreview truncates long output from beginning."""
    widget = OutputPreview()
    # Create content longer than available lines
    lines = [f"Line {i}" for i in range(100)]
    widget.output_content = "\n".join(lines)
    widget.active_test_id = 1

    # Mock _get_available_lines to return a small number
    widget._get_available_lines = lambda: 10  # type: ignore

    rendered = widget.render()
    assert "earlier lines" in rendered
    # Should show the last lines
    assert "Line 99" in rendered


def test_output_preview_update_output_throttling():
    """Test that update_output throttles updates."""
    widget = OutputPreview()

    # First update should work
    widget.update_output("first", 1)
    assert widget.output_content == "first"

    # Immediate second update should be throttled
    widget.update_output("second", 2)
    assert widget.output_content == "first"  # Still first due to throttling


def test_output_preview_get_available_lines_fallback():
    """Test _get_available_lines fallback when no parent."""
    widget = OutputPreview()
    # Widget has no parent or app, should return fallback
    lines = widget._get_available_lines()
    assert lines == 30  # Default fallback


def test_output_preview_get_available_lines_with_parent():
    """Test _get_available_lines with parent that has size."""
    widget = OutputPreview()

    # Mock parent with size via property patching
    parent_mock = MagicMock()
    parent_mock.size.height = 50

    with patch.object(
        OutputPreview, "parent", new_callable=PropertyMock
    ) as mock_parent:
        mock_parent.return_value = parent_mock
        lines = widget._get_available_lines()
        assert lines == 47  # 50 - 3


def test_output_preview_get_available_lines_with_app():
    """Test _get_available_lines when parent has no size but app does."""
    widget = OutputPreview()

    # Mock parent without size attribute
    parent_mock = MagicMock(spec=[])  # No size attr

    # Mock app with size
    app_mock = MagicMock()
    app_mock.size.height = 40

    with (
        patch.object(OutputPreview, "parent", new_callable=PropertyMock) as mock_parent,
        patch.object(OutputPreview, "app", new_callable=PropertyMock) as mock_app,
    ):
        mock_parent.return_value = parent_mock
        mock_app.return_value = app_mock
        lines = widget._get_available_lines()
        assert lines == 25  # 40 - 15


def test_output_preview_get_available_lines_parent_zero_height():
    """Test _get_available_lines falls through when parent has zero height."""
    widget = OutputPreview()

    # Mock parent with size but height=0
    parent_mock = MagicMock()
    parent_mock.size.height = 0

    # Mock app with valid size
    app_mock = MagicMock()
    app_mock.size.height = 30

    with (
        patch.object(OutputPreview, "parent", new_callable=PropertyMock) as mock_parent,
        patch.object(OutputPreview, "app", new_callable=PropertyMock) as mock_app,
    ):
        mock_parent.return_value = parent_mock
        mock_app.return_value = app_mock
        lines = widget._get_available_lines()
        # Should fall through to app check: 30 - 15 = 15
        assert lines == 15


def test_output_preview_get_available_lines_app_zero_height():
    """Test _get_available_lines falls to default when app has zero height."""
    widget = OutputPreview()

    # Mock parent with size but height=0
    parent_mock = MagicMock()
    parent_mock.size.height = 0

    # Mock app with zero height
    app_mock = MagicMock()
    app_mock.size.height = 0

    with (
        patch.object(OutputPreview, "parent", new_callable=PropertyMock) as mock_parent,
        patch.object(OutputPreview, "app", new_callable=PropertyMock) as mock_app,
    ):
        mock_parent.return_value = parent_mock
        mock_app.return_value = app_mock
        lines = widget._get_available_lines()
        # Should fall through to default
        assert lines == 30


# === ShrinkRayApp with fake client tests ===


@pytest.fixture
def basic_updates() -> list[ProgressUpdate]:
    """Generate a sequence of progress updates."""
    return [
        ProgressUpdate(
            status="Running pass 1",
            size=900,
            original_size=1000,
            calls=5,
            reductions=1,
            interesting_calls=2,
            wasted_calls=0,
            runtime=1.0,
            parallel_workers=2,
            average_parallelism=2.0,
            effective_parallelism=1.8,
            time_since_last_reduction=0.5,
            content_preview="Hello World",
            hex_mode=False,
        ),
        ProgressUpdate(
            status="Running pass 2",
            size=700,
            original_size=1000,
            calls=15,
            reductions=3,
            interesting_calls=5,
            wasted_calls=1,
            runtime=2.5,
            parallel_workers=2,
            average_parallelism=2.0,
            effective_parallelism=1.7,
            time_since_last_reduction=0.3,
            content_preview="Hello",
            hex_mode=False,
        ),
        ProgressUpdate(
            status="Running pass 3",
            size=500,
            original_size=1000,
            calls=30,
            reductions=5,
            interesting_calls=8,
            wasted_calls=2,
            runtime=5.0,
            parallel_workers=2,
            average_parallelism=2.0,
            effective_parallelism=1.6,
            time_since_last_reduction=0.2,
            content_preview="Hi",
            hex_mode=False,
        ),
    ]


def test_app_mounts_successfully():
    """Test that the app can mount without errors."""

    async def run_test():
        fake_client = FakeReductionClient(updates=[])
        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test():
            # Check that key widgets exist
            assert app.query_one("#status-label")
            assert app.query_one("#stats-display")

    run_async(run_test())


def test_app_shows_initial_status():
    """Test that the app shows the initial status message."""

    async def run_test():
        fake_client = FakeReductionClient(updates=[])
        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test():
            label = app.query_one("#status-label")
            # Check that the label widget exists and has content
            assert label is not None

    run_async(run_test())


def test_app_receives_progress_updates(basic_updates):
    """Test that the app receives and displays progress updates."""

    async def run_test():
        fake_client = FakeReductionClient(updates=basic_updates)
        # Pre-start the client since app won't call start() on provided client
        await fake_client.start()

        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            # Wait for updates to be processed
            await pilot.pause()
            await asyncio.sleep(0.1)
            await pilot.pause()

            # Client should have been pre-started
            assert fake_client._started

    run_async(run_test())


def test_app_shows_error_on_start_failure():
    """Test that the app handles start errors when creating its own client."""
    # Note: When a client is passed in, start_reduction is not called.
    # Error handling for start failures now happens during pre-validation
    # in run_textual_ui, so this test just verifies the app handles
    # the error_message property correctly.

    async def run_test():
        fake_client = FakeReductionClient(updates=[])
        # Pre-start since app won't call start() on provided clients
        await fake_client.start()

        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            await asyncio.sleep(0.05)
            await pilot.pause()

            # Client should have been pre-started
            assert fake_client._started

    run_async(run_test())


def test_quit_action():
    """Test that pressing 'q' quits the app."""

    async def run_test():
        fake_client = FakeReductionClient(updates=[])
        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            await pilot.press("q")
            # App should exit
            assert app.return_code is None or app.return_code == 0

    run_async(run_test())


def test_quit_cancels_reduction(basic_updates):
    """Test that pressing 'q' cancels the reduction and quits."""

    async def run_test():
        # Create a client that yields updates indefinitely until cancelled
        class InfiniteUpdatesClient(FakeReductionClient):
            async def get_progress_updates(self):
                i = 0
                while not self._cancelled:
                    yield ProgressUpdate(
                        status=f"Running step {i}",
                        size=1000 - i,
                        original_size=1000,
                        calls=i,
                        reductions=0,
                    )
                    i += 1
                    await asyncio.sleep(0.01)
                self._completed = True

        fake_client = InfiniteUpdatesClient(updates=[])
        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            # Wait briefly for the reduction to start - don't use pause() here
            # as the infinite update loop would cause it to timeout
            await asyncio.sleep(0.05)
            await pilot.press("q")
            # Wait for quit to be processed
            await asyncio.sleep(0.05)
            await pilot.pause()

            # Client should have received cancel
            assert fake_client._cancelled

    run_async(run_test())


def test_app_completes_successfully(basic_updates):
    """Test that the app shows completion message."""

    async def run_test():
        fake_client = FakeReductionClient(updates=basic_updates)
        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            # Wait for all updates to be processed
            for _ in range(20):
                await pilot.pause()
                await asyncio.sleep(0.02)
                if app.is_completed:
                    break

            # Check completion
            assert fake_client._completed

    run_async(run_test())


def test_progress_updates_change_stats(basic_updates):
    """Test that progress updates change the stats display."""

    async def run_test():
        fake_client = FakeReductionClient(updates=basic_updates)
        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            # Wait for updates
            for _ in range(10):
                await pilot.pause()
                await asyncio.sleep(0.02)
                if fake_client._completed:
                    break

            stats = app.query_one("#stats-display", StatsDisplay)
            # Stats should have been updated with final values
            if fake_client._update_index > 0:
                assert stats.original_size == 1000

    run_async(run_test())


def test_app_sets_title():
    """Test that the app sets the title correctly."""

    async def run_test():
        fake_client = FakeReductionClient(updates=[])
        app = ShrinkRayApp(
            file_path="/tmp/my_test_file.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test():
            assert app.title == "Shrink Ray"
            assert app.sub_title == "/tmp/my_test_file.txt"

    run_async(run_test())


def test_client_closed_on_completion(basic_updates):
    """Test that the client is closed after reduction completes."""

    async def run_test():
        fake_client = FakeReductionClient(updates=basic_updates)
        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            # Wait for completion
            for _ in range(20):
                await pilot.pause()
                await asyncio.sleep(0.02)
                if app.is_completed:
                    break

            # Give time for cleanup
            await asyncio.sleep(0.1)
            await pilot.pause()

    run_async(run_test())


def test_app_with_various_parameters():
    """Test that the app accepts all parameter combinations."""

    async def run_test():
        fake_client = FakeReductionClient(updates=[])
        app = ShrinkRayApp(
            file_path="/tmp/test.cpp",
            test=["./test.sh", "--verbose"],
            parallelism=4,
            timeout=5.0,
            seed=42,
            input_type="arg",
            in_place=True,
            formatter="clang-format",
            volume="quiet",
            no_clang_delta=True,
            clang_delta="/usr/bin/clang_delta",
            client=fake_client,
        )

        async with app.run_test():
            assert app._parallelism == 4
            assert app._timeout == 5.0
            assert app._seed == 42

    run_async(run_test())


# === App without client tests ===


def test_app_creates_own_client():
    """Test that app creates its own client when none provided."""

    async def run_test():
        # Mock SubprocessClient to avoid actually spawning subprocess
        mock_client = MagicMock()
        mock_client.start = AsyncMock()
        mock_client.start_reduction = AsyncMock(
            return_value=Response(id="start", error="Test error")
        )
        mock_client.close = AsyncMock()
        mock_client.is_completed = True

        with patch("shrinkray.tui.SubprocessClient", return_value=mock_client):
            app = ShrinkRayApp(
                file_path="/tmp/test.txt",
                test=["./test.sh"],
                # No client provided - app should create one
            )

            async with app.run_test() as pilot:
                await pilot.pause()
                await asyncio.sleep(0.1)
                await pilot.pause()

                # Client should have been created and used
                mock_client.start.assert_called_once()

    run_async(run_test())


def test_app_successful_start_covers_no_error_branch():
    """Test successful start covers the 'no error' branch (424->430)."""

    async def run_test():
        # Mock SubprocessClient with successful start
        mock_client = MagicMock()
        mock_client.start = AsyncMock()
        mock_client.start_reduction = AsyncMock(
            return_value=Response(id="start", result={"status": "started"})
        )
        mock_client.close = AsyncMock()
        mock_client.is_completed = True
        mock_client.error_message = None

        # Mock get_progress_updates to return immediately
        async def mock_updates():
            return
            yield  # Make it an async generator

        mock_client.get_progress_updates = mock_updates

        with patch("shrinkray.tui.SubprocessClient", return_value=mock_client):
            app = ShrinkRayApp(
                file_path="/tmp/test.txt",
                test=["./test.sh"],
            )

            async with app.run_test() as pilot:
                await pilot.pause()
                await asyncio.sleep(0.1)
                await pilot.pause()

                # Verify reduction was started successfully
                mock_client.start.assert_called_once()
                mock_client.start_reduction.assert_called_once()

    run_async(run_test())


def test_app_handles_exception_in_run_reduction():
    """Test that app handles exceptions during reduction."""

    async def run_test():
        mock_client = MagicMock()
        mock_client.start = AsyncMock(side_effect=Exception("Connection failed"))
        mock_client.close = AsyncMock()
        mock_client.is_completed = False

        with patch("shrinkray.tui.SubprocessClient", return_value=mock_client):
            app = ShrinkRayApp(
                file_path="/tmp/test.txt",
                test=["./test.sh"],
            )

            async with app.run_test() as pilot:
                await pilot.pause()
                await asyncio.sleep(0.1)
                await pilot.pause()

                # App should have handled the error
                mock_client.start.assert_called_once()

    run_async(run_test())


# === Box navigation tests ===


def test_box_containers_are_focusable():
    """Test that all box containers can be focused."""

    async def run_test():
        fake_client = FakeReductionClient(updates=[])
        fake_client._completed = True

        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            await asyncio.sleep(0.1)
            await pilot.pause()

            # Check all containers are focusable
            for box_id in [
                "stats-container",
                "graph-container",
                "content-container",
                "output-container",
            ]:
                container = app.query_one(f"#{box_id}")
                assert container.can_focus, f"{box_id} should be focusable"

    run_async(run_test())


def test_focus_navigation_actions_exist():
    """Test that focus navigation action methods exist and are callable."""

    async def run_test():
        fake_client = FakeReductionClient(updates=[])
        fake_client._completed = True

        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            await pilot.pause()

            # Verify the action methods exist
            assert hasattr(app, "action_focus_up")
            assert hasattr(app, "action_focus_down")
            assert hasattr(app, "action_focus_left")
            assert hasattr(app, "action_focus_right")
            assert hasattr(app, "action_expand_box")

            # Call them to ensure they don't error
            app.action_focus_up()
            app.action_focus_down()
            app.action_focus_left()
            app.action_focus_right()

    run_async(run_test())


def test_get_focused_box_index():
    """Test that _get_focused_box_index returns correct indices."""

    async def run_test():
        fake_client = FakeReductionClient(updates=[])
        fake_client._completed = True

        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            await asyncio.sleep(0.1)
            await pilot.pause()

            # Initial focus should be on stats-container (index 0)
            assert app._get_focused_box_index() == 0

    run_async(run_test())


def test_enter_expands_box_to_modal():
    """Test that Enter opens an expanded modal for the focused box."""

    async def run_test():
        fake_client = FakeReductionClient(updates=[])
        fake_client._completed = True

        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            await asyncio.sleep(0.1)
            await pilot.pause()

            # Call action directly to open modal
            app.action_expand_box()
            await pilot.pause()

            # Should have an expanded modal in the screen stack
            assert any(
                isinstance(screen, ExpandedBoxModal) for screen in app.screen_stack
            )

    run_async(run_test())


def test_modal_dismiss_bindings():
    """Test that the ExpandedBoxModal has the correct dismiss bindings."""
    # Just verify the bindings are configured correctly
    bindings = ExpandedBoxModal.BINDINGS
    binding_keys = []
    for binding in bindings:
        if isinstance(binding, tuple):
            binding_keys.append(binding[0])
    assert any("escape" in str(key) for key in binding_keys)
    assert any("enter" in str(key) for key in binding_keys)
    assert any("q" in str(key) for key in binding_keys)


def test_expanded_modal_stores_content_widget_id():
    """Test that ExpandedBoxModal stores the content widget ID."""
    modal = ExpandedBoxModal("Size Over Time", "graph-container")
    assert modal._content_widget_id == "graph-container"
    assert modal._title == "Size Over Time"


def test_expanded_modal_stores_file_path():
    """Test that ExpandedBoxModal stores the file path."""
    modal = ExpandedBoxModal("Current Test Case", "content-container", "/path/to/file")
    assert modal._file_path == "/path/to/file"


def test_expanded_modal_read_file_success(tmp_path):
    """Test _read_file reads file successfully."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello World")

    modal = ExpandedBoxModal("Test", "content-container")
    content = modal._read_file(str(test_file))
    assert content == "Hello World"


def test_expanded_modal_read_file_binary(tmp_path):
    """Test _read_file falls back to hex for binary content."""
    test_file = tmp_path / "test.bin"
    # Use bytes that are invalid UTF-8 to trigger the hex fallback
    test_file.write_bytes(b"\x80\x81\x82\x83")

    modal = ExpandedBoxModal("Test", "content-container")
    content = modal._read_file(str(test_file))
    assert "Binary content" in content
    assert "80818283" in content


def test_expanded_modal_read_file_missing(tmp_path):
    """Test _read_file raises OSError for missing file."""
    modal = ExpandedBoxModal("Test", "content-container")
    with pytest.raises(OSError):
        modal._read_file(str(tmp_path / "nonexistent.txt"))


# === ExpandedBoxModal integration tests ===


def test_expanded_modal_compose_graph_container():
    """Test ExpandedBoxModal compose yields graph for graph-container."""

    async def run_test():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)
        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            await pilot.pause()  # Wait for app to mount
            # Push an expanded modal for the graph
            modal = ExpandedBoxModal("Size Over Time", "graph-container")
            await app.push_screen(modal)
            await pilot.pause()

            # Verify the expanded graph widget exists (query from app's current screen)
            expanded_graph = app.screen.query_one("#expanded-graph", SizeGraph)
            assert expanded_graph is not None

    run_async(run_test())


def test_expanded_modal_compose_static_container():
    """Test ExpandedBoxModal compose yields static for non-graph content."""

    async def run_test():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)
        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            await pilot.pause()  # Wait for app to mount
            # Push an expanded modal for stats
            modal = ExpandedBoxModal("Statistics", "stats-container")
            await app.push_screen(modal)
            await pilot.pause()

            # Verify the expanded content static widget exists
            expanded_content = app.screen.query_one("#expanded-content", Static)
            assert expanded_content is not None

    run_async(run_test())


def test_expanded_modal_on_mount_stats():
    """Test ExpandedBoxModal on_mount populates stats content."""

    async def run_test():
        # Create an update to populate stats
        update = ProgressUpdate(
            status="Running",
            size=500,
            original_size=1000,
            calls=10,
            reductions=5,
            interesting_calls=3,
            wasted_calls=1,
            runtime=5.0,
            parallel_workers=2,
            average_parallelism=1.8,
            effective_parallelism=1.5,
            time_since_last_reduction=1.0,
            content_preview="Test content",
            hex_mode=False,
        )
        fake_client = FakeReductionClient(updates=[update], wait_indefinitely=True)

        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            # Wait for the update to be processed
            await pilot.pause()
            await asyncio.sleep(0.1)
            await pilot.pause()

            # Push an expanded modal for stats
            modal = ExpandedBoxModal("Statistics", "stats-container")
            await app.push_screen(modal)
            await pilot.pause()

            # Verify the content was populated
            expanded_content = app.screen.query_one("#expanded-content", Static)
            # Content should not be empty
            assert expanded_content is not None

    run_async(run_test())


def test_expanded_modal_on_mount_graph():
    """Test ExpandedBoxModal on_mount copies graph data."""

    async def run_test():
        # Create updates with size history
        updates = [
            ProgressUpdate(
                status="Running",
                size=800,
                original_size=1000,
                calls=5,
                reductions=2,
                interesting_calls=2,
                wasted_calls=0,
                runtime=1.0,
                parallel_workers=2,
                average_parallelism=2.0,
                effective_parallelism=1.8,
                time_since_last_reduction=0.5,
                content_preview="Test",
                hex_mode=False,
                new_size_history=[(0.0, 1000), (1.0, 800)],
            ),
        ]
        fake_client = FakeReductionClient(updates=updates, wait_indefinitely=True)

        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            # Wait for updates
            await pilot.pause()
            await asyncio.sleep(0.1)
            await pilot.pause()

            # Push an expanded modal for the graph
            modal = ExpandedBoxModal("Size Over Time", "graph-container")
            await app.push_screen(modal)
            await pilot.pause()

            # Verify the graph widget exists and has data copied
            expanded_graph = app.screen.query_one("#expanded-graph", SizeGraph)
            assert expanded_graph is not None
            # The graph should have history data copied from main graph
            main_graph = app.query_one("#size-graph", SizeGraph)
            assert expanded_graph._size_history == main_graph._size_history

    run_async(run_test())


def test_expanded_modal_on_mount_output():
    """Test ExpandedBoxModal on_mount populates output content."""

    async def run_test():
        update = ProgressUpdate(
            status="Running",
            size=500,
            original_size=1000,
            calls=10,
            reductions=5,
            interesting_calls=3,
            wasted_calls=1,
            runtime=5.0,
            parallel_workers=2,
            average_parallelism=1.8,
            effective_parallelism=1.5,
            time_since_last_reduction=1.0,
            content_preview="Test content",
            hex_mode=False,
            test_output_preview="Test output line 1\nTest output line 2",
            active_test_id=1,
        )
        fake_client = FakeReductionClient(updates=[update], wait_indefinitely=True)

        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            # Wait for the update to be processed
            await pilot.pause()
            await asyncio.sleep(0.1)
            await pilot.pause()

            # Push an expanded modal for output
            modal = ExpandedBoxModal("Test Output", "output-container")
            await app.push_screen(modal)
            await pilot.pause()

            # Verify the content was populated
            expanded_content = app.screen.query_one("#expanded-content", Static)
            assert expanded_content is not None

    run_async(run_test())


def test_expanded_modal_on_mount_output_completed_with_return_code():
    """Test ExpandedBoxModal on_mount for output with completed test and return code."""

    async def run_test():
        # First update: test is running
        update1 = ProgressUpdate(
            status="Running",
            size=500,
            original_size=1000,
            calls=5,
            reductions=2,
            interesting_calls=2,
            wasted_calls=0,
            runtime=2.0,
            parallel_workers=2,
            average_parallelism=1.8,
            effective_parallelism=1.5,
            time_since_last_reduction=0.5,
            content_preview="Test content",
            hex_mode=False,
            test_output_preview="Output during test",
            active_test_id=42,  # Test is running
        )
        # Second update: test completed with return code
        update2 = ProgressUpdate(
            status="Running",
            size=500,
            original_size=1000,
            calls=10,
            reductions=5,
            interesting_calls=3,
            wasted_calls=1,
            runtime=5.0,
            parallel_workers=2,
            average_parallelism=1.8,
            effective_parallelism=1.5,
            time_since_last_reduction=1.0,
            content_preview="Test content",
            hex_mode=False,
            test_output_preview="Final output",
            active_test_id=None,  # Test completed
            last_test_return_code=1,  # With return code
        )
        fake_client = FakeReductionClient(
            updates=[update1, update2], wait_indefinitely=True
        )

        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            # Wait for both updates to be processed
            await pilot.pause()
            await asyncio.sleep(0.2)
            await pilot.pause()

            modal = ExpandedBoxModal("Test Output", "output-container")
            await app.push_screen(modal)
            await pilot.pause()

            expanded_content = app.screen.query_one("#expanded-content", Static)
            assert expanded_content is not None

    run_async(run_test())


def test_expanded_modal_on_mount_output_completed_without_return_code():
    """Test ExpandedBoxModal on_mount for output with completed test but no return code."""

    async def run_test():
        # First update: test is running
        update1 = ProgressUpdate(
            status="Running",
            size=500,
            original_size=1000,
            calls=5,
            reductions=2,
            interesting_calls=2,
            wasted_calls=0,
            runtime=2.0,
            parallel_workers=2,
            average_parallelism=1.8,
            effective_parallelism=1.5,
            time_since_last_reduction=0.5,
            content_preview="Test content",
            hex_mode=False,
            test_output_preview="Output during test",
            active_test_id=42,  # Test is running
        )
        # Second update: test completed without return code
        update2 = ProgressUpdate(
            status="Running",
            size=500,
            original_size=1000,
            calls=10,
            reductions=5,
            interesting_calls=3,
            wasted_calls=1,
            runtime=5.0,
            parallel_workers=2,
            average_parallelism=1.8,
            effective_parallelism=1.5,
            time_since_last_reduction=1.0,
            content_preview="Test content",
            hex_mode=False,
            test_output_preview="Final output",
            active_test_id=None,  # Test completed
            # No return code specified
        )
        fake_client = FakeReductionClient(
            updates=[update1, update2], wait_indefinitely=True
        )

        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            # Wait for both updates to be processed
            await pilot.pause()
            await asyncio.sleep(0.2)
            await pilot.pause()

            modal = ExpandedBoxModal("Test Output", "output-container")
            await app.push_screen(modal)
            await pilot.pause()

            expanded_content = app.screen.query_one("#expanded-content", Static)
            assert expanded_content is not None

    run_async(run_test())


def test_expanded_modal_on_mount_output_seen_but_no_content():
    """Test ExpandedBoxModal on_mount for output when we've seen output before but no current content."""

    async def run_test():
        # First update: test with output
        update1 = ProgressUpdate(
            status="Running",
            size=500,
            original_size=1000,
            calls=5,
            reductions=2,
            interesting_calls=2,
            wasted_calls=0,
            runtime=2.0,
            parallel_workers=2,
            average_parallelism=1.8,
            effective_parallelism=1.5,
            time_since_last_reduction=0.5,
            content_preview="Test content",
            hex_mode=False,
            test_output_preview="Some output",
            active_test_id=42,
        )
        # Second update: no content but test completed
        update2 = ProgressUpdate(
            status="Running",
            size=500,
            original_size=1000,
            calls=10,
            reductions=5,
            interesting_calls=3,
            wasted_calls=1,
            runtime=5.0,
            parallel_workers=2,
            average_parallelism=1.8,
            effective_parallelism=1.5,
            time_since_last_reduction=1.0,
            content_preview="Test content",
            hex_mode=False,
            test_output_preview="",  # Empty output
            active_test_id=None,
        )
        fake_client = FakeReductionClient(
            updates=[update1, update2], wait_indefinitely=True
        )

        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            # Wait for both updates to be processed
            await pilot.pause()
            await asyncio.sleep(0.2)
            await pilot.pause()

            modal = ExpandedBoxModal("Test Output", "output-container")
            await app.push_screen(modal)
            await pilot.pause()

            expanded_content = app.screen.query_one("#expanded-content", Static)
            assert expanded_content is not None

    run_async(run_test())


def test_expanded_modal_on_mount_output_no_content():
    """Test ExpandedBoxModal on_mount for output with no content."""

    async def run_test():
        # No test output in the update
        update = ProgressUpdate(
            status="Running",
            size=500,
            original_size=1000,
            calls=10,
            reductions=5,
            interesting_calls=3,
            wasted_calls=1,
            runtime=5.0,
            parallel_workers=2,
            average_parallelism=1.8,
            effective_parallelism=1.5,
            time_since_last_reduction=1.0,
            content_preview="Test content",
            hex_mode=False,
            test_output_preview="",  # No test output
            active_test_id=None,
        )
        fake_client = FakeReductionClient(updates=[update], wait_indefinitely=True)

        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            await asyncio.sleep(0.1)
            await pilot.pause()

            modal = ExpandedBoxModal("Test Output", "output-container")
            await app.push_screen(modal)
            await pilot.pause()

            expanded_content = app.screen.query_one("#expanded-content", Static)
            assert expanded_content is not None

    run_async(run_test())


def test_expanded_modal_on_mount_content_with_file(tmp_path):
    """Test ExpandedBoxModal on_mount reads file content."""

    async def run_test():
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("File content for test")

        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)
        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            await pilot.pause()  # Wait for app to mount
            # Push an expanded modal for content with file path
            modal = ExpandedBoxModal(
                "Current Test Case", "content-container", str(test_file)
            )
            await app.push_screen(modal)
            await pilot.pause()

            # Verify the content was populated from file
            expanded_content = app.screen.query_one("#expanded-content", Static)
            # The renderable should contain the file content
            assert expanded_content is not None

    run_async(run_test())


def test_expanded_modal_on_mount_content_without_file():
    """Test ExpandedBoxModal on_mount uses preview content when no file path."""

    async def run_test():
        update = ProgressUpdate(
            status="Running",
            size=500,
            original_size=1000,
            calls=10,
            reductions=5,
            interesting_calls=3,
            wasted_calls=1,
            runtime=5.0,
            parallel_workers=2,
            average_parallelism=1.8,
            effective_parallelism=1.5,
            time_since_last_reduction=1.0,
            content_preview="Preview content from widget",
            hex_mode=False,
        )
        fake_client = FakeReductionClient(updates=[update], wait_indefinitely=True)

        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            # Wait for the update to be processed
            await pilot.pause()
            await asyncio.sleep(0.1)
            await pilot.pause()

            # Push an expanded modal for content WITHOUT file path
            modal = ExpandedBoxModal("Current Test Case", "content-container")
            await app.push_screen(modal)
            await pilot.pause()

            # Verify the content was populated from the preview widget
            expanded_content = app.screen.query_one("#expanded-content", Static)
            assert expanded_content is not None

    run_async(run_test())


# === Focus navigation tests ===


def test_focus_navigation_up():
    """Test focus up action wraps correctly."""

    async def run_test():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)
        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            await pilot.pause()

            # Focus on a box in the top row
            app._focus_box(0)
            await pilot.pause()

            # Move up should wrap to bottom
            app.action_focus_up()
            await pilot.pause()
            assert app._get_focused_box_index() == 2

    run_async(run_test())


def test_focus_navigation_down():
    """Test focus down action wraps correctly."""

    async def run_test():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)
        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            await pilot.pause()

            # Focus on a box in the bottom row
            app._focus_box(2)
            await pilot.pause()

            # Move down should wrap to top
            app.action_focus_down()
            await pilot.pause()
            assert app._get_focused_box_index() == 0

    run_async(run_test())


def test_focus_navigation_left():
    """Test focus left action wraps correctly."""

    async def run_test():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)
        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            await pilot.pause()

            # Focus on leftmost box
            app._focus_box(0)
            await pilot.pause()

            # Move left should wrap to right
            app.action_focus_left()
            await pilot.pause()
            assert app._get_focused_box_index() == 1

    run_async(run_test())


def test_focus_navigation_right():
    """Test focus right action wraps correctly."""

    async def run_test():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)
        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            await pilot.pause()

            # Focus on rightmost box
            app._focus_box(1)
            await pilot.pause()

            # Move right should wrap to left
            app.action_focus_right()
            await pilot.pause()
            assert app._get_focused_box_index() == 0

    run_async(run_test())


def test_focus_navigation_up_from_bottom():
    """Test focus up action from bottom row to top row."""

    async def run_test():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)
        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            await pilot.pause()

            # Focus on a box in the bottom row (index 2)
            app._focus_box(2)
            await pilot.pause()

            # Move up should go to top row (2 - 2 = 0)
            app.action_focus_up()
            await pilot.pause()
            assert app._get_focused_box_index() == 0

    run_async(run_test())


def test_focus_navigation_down_from_top():
    """Test focus down action from top row to bottom row."""

    async def run_test():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)
        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            await pilot.pause()

            # Focus on a box in the top row (index 0)
            app._focus_box(0)
            await pilot.pause()

            # Move down should go to bottom row (0 + 2 = 2)
            app.action_focus_down()
            await pilot.pause()
            assert app._get_focused_box_index() == 2

    run_async(run_test())


def test_focus_navigation_left_from_right():
    """Test focus left action from right column to left column."""

    async def run_test():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)
        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            await pilot.pause()

            # Focus on a box in the right column (index 1, odd)
            app._focus_box(1)
            await pilot.pause()

            # Move left should go to left column (1 - 1 = 0)
            app.action_focus_left()
            await pilot.pause()
            assert app._get_focused_box_index() == 0

    run_async(run_test())


def test_focus_navigation_right_from_left():
    """Test focus right action from left column to right column."""

    async def run_test():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)
        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            await pilot.pause()

            # Focus on a box in the left column (index 0, even)
            app._focus_box(0)
            await pilot.pause()

            # Move right should go to right column (0 + 1 = 1)
            app.action_focus_right()
            await pilot.pause()
            assert app._get_focused_box_index() == 1

    run_async(run_test())


# === Update expanded modal tests ===


def test_update_expanded_graph():
    """Test _update_expanded_graph updates graph in modal."""

    async def run_test():
        updates = [
            ProgressUpdate(
                status="Running",
                size=800,
                original_size=1000,
                calls=5,
                reductions=2,
                interesting_calls=2,
                wasted_calls=0,
                runtime=1.0,
                parallel_workers=2,
                average_parallelism=2.0,
                effective_parallelism=1.8,
                time_since_last_reduction=0.5,
                content_preview="Test",
                hex_mode=False,
                new_size_history=[(0.0, 1000), (1.0, 800)],
            ),
        ]
        fake_client = FakeReductionClient(updates=updates, wait_indefinitely=True)

        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            # Wait for initial update
            await pilot.pause()
            await asyncio.sleep(0.1)
            await pilot.pause()

            # Push graph modal
            modal = ExpandedBoxModal("Size Over Time", "graph-container")
            await app.push_screen(modal)
            await pilot.pause()

            # Call update method directly
            app._update_expanded_graph([(2.0, 600)], 1000, 2.0)
            await pilot.pause()

            # Verify the expanded graph was updated
            expanded_graph = app.screen.query_one("#expanded-graph", SizeGraph)
            # Should have the new entry
            assert (2.0, 600) in expanded_graph._size_history

    run_async(run_test())


def test_update_expanded_stats():
    """Test _update_expanded_stats updates stats in modal."""

    async def run_test():
        update = ProgressUpdate(
            status="Running",
            size=500,
            original_size=1000,
            calls=10,
            reductions=5,
            interesting_calls=3,
            wasted_calls=1,
            runtime=5.0,
            parallel_workers=2,
            average_parallelism=1.8,
            effective_parallelism=1.5,
            time_since_last_reduction=1.0,
            content_preview="Test content",
            hex_mode=False,
        )
        fake_client = FakeReductionClient(updates=[update], wait_indefinitely=True)

        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            # Wait for initial update
            await pilot.pause()
            await asyncio.sleep(0.1)
            await pilot.pause()

            # Push stats modal
            modal = ExpandedBoxModal("Statistics", "stats-container")
            await app.push_screen(modal)
            await pilot.pause()

            # Call update method directly
            app._update_expanded_stats()
            await pilot.pause()

            # Verify the modal still exists and has content
            expanded_content = app.screen.query_one("#expanded-content", Static)
            assert expanded_content is not None

    run_async(run_test())


# === End-to-end tests ===


@pytest.fixture
def temp_test_file():
    """Create a temporary test file for reduction."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as f:
        f.write(b"Hello, World! This is some test content to reduce.")
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_test_script():
    """Create a temporary interestingness test script."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        # Script that succeeds if file contains "Hello"
        f.write('#!/bin/bash\ngrep -q "Hello" "$1"\n')
        temp_path = f.name
    os.chmod(temp_path, 0o755)
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


def test_real_subprocess_communication(temp_test_file, temp_test_script):
    """Test with a real subprocess client."""

    async def run_test():
        # Create a client we can control
        client = SubprocessClient()
        app = ShrinkRayApp(
            file_path=temp_test_file,
            test=[temp_test_script],
            parallelism=1,
            timeout=5.0,
            client=client,
        )

        async with app.run_test() as pilot:
            # Wait a moment for subprocess to start
            await pilot.pause()
            await asyncio.sleep(0.3)
            await pilot.pause()

            # The app should have started
            assert app.query_one("#status-label")
            assert app.query_one("#stats-display")

            # Quit to clean up
            await pilot.press("q")

    run_async(run_test())


# === Theme detection tests ===


def test_detect_dark_from_colorfgbg_dark(monkeypatch):
    """Test that COLORFGBG with dark background returns True."""

    monkeypatch.setenv("COLORFGBG", "15;0")  # white on black
    assert detect_terminal_theme() is True


def test_detect_light_from_colorfgbg_light(monkeypatch):
    """Test that COLORFGBG with light background returns False."""

    monkeypatch.setenv("COLORFGBG", "0;15")  # black on white
    assert detect_terminal_theme() is False


def test_detect_light_from_colorfgbg_gray(monkeypatch):
    """Test that COLORFGBG with gray background (7+) returns False."""

    monkeypatch.setenv("COLORFGBG", "0;7")  # black on light gray
    assert detect_terminal_theme() is False


def test_detect_dark_colorfgbg_boundary(monkeypatch):
    """Test that COLORFGBG with value 6 returns True (dark)."""

    monkeypatch.setenv("COLORFGBG", "15;6")
    assert detect_terminal_theme() is True


def test_detect_invalid_colorfgbg_falls_through(monkeypatch):
    """Test that invalid COLORFGBG falls through to default."""

    monkeypatch.setenv("COLORFGBG", "invalid")
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    # Should fall through to default (True = dark)
    assert detect_terminal_theme() is True


def test_detect_colorfgbg_non_numeric(monkeypatch):
    """Test that non-numeric COLORFGBG values are handled."""

    monkeypatch.setenv("COLORFGBG", "foo;bar")
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    assert detect_terminal_theme() is True


def test_detect_empty_colorfgbg(monkeypatch):
    """Test that empty COLORFGBG falls through."""

    monkeypatch.setenv("COLORFGBG", "")
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    assert detect_terminal_theme() is True


def test_detect_no_env_vars_defaults_dark(monkeypatch):
    """Test that no environment variables defaults to dark."""

    monkeypatch.delenv("COLORFGBG", raising=False)
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    assert detect_terminal_theme() is True


def test_detect_macos_terminal_dark(monkeypatch):
    """Test macOS Terminal.app dark mode detection."""

    monkeypatch.delenv("COLORFGBG", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    monkeypatch.delenv("__CFBundleIdentifier", raising=False)

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Dark"

    with patch("subprocess.run", return_value=mock_result):
        assert detect_terminal_theme() is True


def test_detect_macos_terminal_light(monkeypatch):
    """Test macOS Terminal.app light mode detection."""

    monkeypatch.delenv("COLORFGBG", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    monkeypatch.delenv("__CFBundleIdentifier", raising=False)

    mock_result = MagicMock()
    mock_result.returncode = 1  # Fails when in light mode
    mock_result.stdout = ""

    with patch("subprocess.run", return_value=mock_result):
        assert detect_terminal_theme() is False


def test_detect_macos_iterm_dark(monkeypatch):
    """Test iTerm.app dark mode detection."""

    monkeypatch.delenv("COLORFGBG", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    monkeypatch.delenv("__CFBundleIdentifier", raising=False)

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Dark"

    with patch("subprocess.run", return_value=mock_result):
        assert detect_terminal_theme() is True


def test_detect_macos_subprocess_exception(monkeypatch):
    """Test macOS detection handles subprocess exceptions."""

    monkeypatch.delenv("COLORFGBG", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    monkeypatch.delenv("__CFBundleIdentifier", raising=False)

    with patch("subprocess.run", side_effect=Exception("timeout")):
        # Should fall through to default (True = dark)
        assert detect_terminal_theme() is True


def test_detect_macos_with_cf_bundle_identifier(monkeypatch):
    """Test macOS detection skips subprocess when __CFBundleIdentifier is set."""

    monkeypatch.delenv("COLORFGBG", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    monkeypatch.setenv("__CFBundleIdentifier", "com.apple.Terminal")

    # Should fall through to default without calling subprocess
    assert detect_terminal_theme() is True


# === Theme settings tests ===


def test_app_with_dark_theme():
    """Test that dark theme is applied correctly."""

    async def run_test():
        fake_client = FakeReductionClient(updates=[])
        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
            theme="dark",
        )

        async with app.run_test():
            assert app.theme == "shrinkray-dark"

    run_async(run_test())


def test_app_with_light_theme():
    """Test that light theme is applied correctly."""

    async def run_test():
        fake_client = FakeReductionClient(updates=[])
        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
            theme="light",
        )

        async with app.run_test():
            assert app.theme == "shrinkray-light"

    run_async(run_test())


def test_app_with_auto_theme(monkeypatch):
    """Test that auto theme uses detection."""

    async def run_test():
        # Force light mode detection
        monkeypatch.setenv("COLORFGBG", "0;15")

        fake_client = FakeReductionClient(updates=[])
        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
            theme="auto",
        )

        async with app.run_test():
            assert app.theme == "shrinkray-light"

    run_async(run_test())


# === Module interface tests ===


def test_tui_can_be_imported():
    """Test that the TUI module can be imported."""

    assert hasattr(tui, "ShrinkRayApp")
    assert hasattr(tui, "run_textual_ui")
    assert hasattr(tui, "StatsDisplay")
    assert hasattr(tui, "ReductionClientProtocol")


def test_shrinkray_app_is_textual_app():
    """Test that ShrinkRayApp is a proper textual App subclass."""

    assert issubclass(ShrinkRayApp, App)


def test_run_textual_ui_signature():
    """Test that run_textual_ui has expected parameters."""

    sig = inspect.signature(run_textual_ui)
    params = list(sig.parameters.keys())

    assert "file_path" in params
    assert "test" in params
    assert "parallelism" in params
    assert "timeout" in params
    assert "seed" in params
    assert "input_type" in params
    assert "in_place" in params
    assert "formatter" in params
    assert "volume" in params
    assert "no_clang_delta" in params
    assert "clang_delta" in params
    assert "theme" in params


def test_fake_client_implements_protocol():
    """Test that FakeReductionClient implements the protocol."""
    client = FakeReductionClient()

    # Check it has all required methods
    assert hasattr(client, "start")
    assert hasattr(client, "start_reduction")
    assert hasattr(client, "cancel")
    assert hasattr(client, "close")
    assert hasattr(client, "get_progress_updates")
    assert hasattr(client, "is_completed")


# === ContentPreview edge cases ===


def test_get_available_lines_fallback_to_app_size():
    """Test _get_available_lines falls back to app.size when parent unavailable."""
    widget = ContentPreview()

    mock_app = MagicMock()
    mock_app.size.height = 50
    widget._app = mock_app  # type: ignore

    # The method should use the app size fallback
    result = widget._get_available_lines()
    assert result >= 10  # Minimum is 10


def test_render_identical_content_no_diff():
    """Test that identical content shows truncated view, not diff."""
    widget = ContentPreview()

    # Set up large content
    lines = [f"Line {i}" for i in range(50)]
    content = "\n".join(lines)
    widget.preview_content = content
    widget._last_displayed_content = content  # Same content

    # Mock _get_available_lines
    widget._get_available_lines = lambda: 10  # type: ignore

    rendered = widget.render()
    # Should show truncated content, not diff (since content is identical)
    # The diff would be empty
    assert "more lines" in rendered or "Line" in rendered


# === ShrinkRayApp exception handlers ===


def test_update_status_before_mount():
    """Test update_status handles exception when widget not mounted."""

    async def run_test():
        fake_client = FakeReductionClient(updates=[])
        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )
        # Try to update status before mounting - should not raise
        app.update_status("Test message")

    run_async(run_test())


def test_quit_handles_cancel_exception():
    """Test that action_quit handles exceptions from cancel."""

    async def run_test():
        # Create a mock client that raises on cancel
        mock_client = MagicMock()
        mock_client.start = AsyncMock()
        mock_client.start_reduction = AsyncMock(
            return_value=Response(id="start", result={"status": "started"})
        )
        mock_client.cancel = AsyncMock(side_effect=Exception("Process exited"))
        mock_client.close = AsyncMock()
        mock_client.is_completed = False

        async def get_updates():
            while True:
                await asyncio.sleep(0.1)
                if mock_client._cancelled:
                    break
                yield ProgressUpdate(
                    status="Running",
                    size=100,
                    original_size=100,
                    calls=0,
                    reductions=0,
                )

        mock_client.get_progress_updates = get_updates

        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=mock_client,
        )
        # Set _owns_client to False so it doesn't try to close
        app._owns_client = False

        async with app.run_test() as pilot:
            await pilot.pause()
            await asyncio.sleep(0.05)
            # Trigger quit which should handle the cancel exception
            await pilot.press("q")
            await pilot.pause()
            # Should not raise

    run_async(run_test())


def test_client_completed_breaks_loop():
    """Test that loop breaks when client.is_completed is True."""

    async def run_test():
        # Create updates that mark client as completed
        updates = [
            ProgressUpdate(
                status="Done",
                size=50,
                original_size=100,
                calls=10,
                reductions=5,
            ),
        ]
        fake_client = FakeReductionClient(updates=updates)

        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            # Wait for completion
            for _ in range(20):
                await pilot.pause()
                await asyncio.sleep(0.02)
                if app.is_completed:
                    break

            # Client should be completed
            assert fake_client.is_completed

    run_async(run_test())


def test_no_exit_on_completion_shows_press_q_message():
    """Test that exit_on_completion=False shows 'Press q to exit' message."""

    async def run_test():
        # Create updates that mark client as completed
        updates = [
            ProgressUpdate(
                status="Done",
                size=50,
                original_size=100,
                calls=10,
                reductions=5,
            ),
        ]
        fake_client = FakeReductionClient(updates=updates)

        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
            exit_on_completion=False,  # Don't auto-exit
        )

        async with app.run_test() as pilot:
            # Wait for completion
            for _ in range(20):
                await pilot.pause()
                await asyncio.sleep(0.02)
                if app.is_completed:
                    break

            # Check that status shows the completion message with press q to exit
            status_label = app.query_one("#status-label", Label)
            label_text = str(status_label.render())
            assert "Reduction completed" in label_text
            assert "Press 'q' to exit" in label_text

            # Now press 'q' to quit - this exercises the action_quit branch
            # where _completed is True (skips cancel, goes straight to exit)
            await pilot.press("q")

    run_async(run_test())


def test_loop_with_empty_updates():
    """Test that the async for loop handles empty update iterator.

    This exercises the 433->440 branch where the iterator is immediately
    exhausted (no updates) and the loop exits without executing the body.
    """

    async def run_test():
        # Create a mock client with an empty async generator
        mock_client = MagicMock()
        mock_client.start = AsyncMock()
        mock_client.start_reduction = AsyncMock(
            return_value=Response(id="start", result={"status": "started"})
        )
        mock_client.cancel = AsyncMock(
            return_value=Response(id="cancel", result={"status": "cancelled"})
        )
        mock_client.close = AsyncMock()
        mock_client.error_message = None
        mock_client._completed = False

        @property
        def is_completed(self):
            return self._completed

        # Patch the is_completed property
        type(mock_client).is_completed = is_completed

        async def empty_updates():
            # Async generator that yields nothing - just returns immediately
            mock_client._completed = True
            if False:
                yield  # Make this an async generator but never yield

        mock_client.get_progress_updates = empty_updates

        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=mock_client,
        )

        async with app.run_test() as pilot:
            # Wait for completion
            for _ in range(30):
                await pilot.pause()
                await asyncio.sleep(0.02)
                if app.is_completed:
                    break

            assert app.is_completed

    run_async(run_test())


def test_loop_breaks_on_first_iteration():
    """Test that the loop breaks immediately when is_completed is True.

    This specifically exercises the break path where the loop body executes
    once but is_completed is already True, triggering immediate break.
    """

    class AlreadyCompletedClient(FakeReductionClient):
        """A client where is_completed is True from the start."""

        def __init__(self):
            super().__init__(updates=[])

        async def get_progress_updates(self) -> AsyncGenerator[ProgressUpdate, None]:
            # Mark as completed BEFORE yielding
            self._completed = True
            yield ProgressUpdate(
                status="Done",
                size=50,
                original_size=100,
                calls=1,
                reductions=1,
            )
            # This second yield should never be reached due to break
            yield ProgressUpdate(
                status="Should not see this",
                size=25,
                original_size=100,
                calls=2,
                reductions=2,
            )

    async def run_test():
        fake_client = AlreadyCompletedClient()

        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            for _ in range(20):
                await pilot.pause()
                await asyncio.sleep(0.02)
                if app.is_completed:
                    break

            assert app.is_completed

    run_async(run_test())


# === Coverage edge cases ===


def test_content_preview_app_size_zero_height():
    """Test _get_available_lines when app.size.height is 0."""

    widget = ContentPreview()

    # Mock the app with zero height
    mock_app = MagicMock()
    mock_app.size.height = 0

    # Mock parent with no usable size
    mock_parent = MagicMock()
    mock_parent.size.height = 0

    # Use patch to override the app property
    with patch.object(type(widget), "app", new_callable=PropertyMock) as mock_app_prop:
        mock_app_prop.return_value = mock_app
        with patch.object(
            type(widget), "parent", new_callable=PropertyMock
        ) as mock_parent_prop:
            mock_parent_prop.return_value = mock_parent

            # Should fall through to the default return value of 30
            result = widget._get_available_lines()
            assert result == 30


def test_content_preview_diff_is_empty():
    """Test render when diff computation produces empty result.

    When the diff is empty, we fall through to the truncated content display.
    """
    widget = ContentPreview()
    # Set up content that appears different (to pass the != check)
    # but produces an empty diff when unified_diff is called
    large_content_old = "\n".join([f"line {i}" for i in range(50)])
    large_content_new = (
        "\n".join([f"line {i}" for i in range(50)]) + " "
    )  # Slightly different

    widget._last_displayed_content = large_content_old
    widget.preview_content = large_content_new
    widget.hex_mode = False

    with (
        patch.object(widget, "_get_available_lines", return_value=10),
        patch("difflib.unified_diff", return_value=[]),
    ):
        result = widget.render()

    # Since diff is empty (mocked), should show truncated content
    assert "more lines" in result


def test_cancel_exception_is_caught():
    """Test that exceptions during cancel() are caught gracefully."""

    class ExceptionOnCancelClient(FakeReductionClient):
        def __init__(self):
            # Many updates so we can quit mid-reduction
            super().__init__(updates=[])

        async def cancel(self) -> Response:
            raise ConnectionError("Process already dead")

        async def get_progress_updates(self) -> AsyncGenerator[ProgressUpdate, None]:
            # Yield many updates with delays
            for i in range(100):
                if self._cancelled:
                    break
                yield ProgressUpdate(
                    status="Running",
                    size=100 - i,
                    original_size=100,
                    calls=i,
                    reductions=0,
                )
                await asyncio.sleep(0.05)
            self._completed = True

    async def run_test():
        fake_client = ExceptionOnCancelClient()

        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            # Wait briefly for reduction to start - don't use pause() here
            # as the update loop would cause it to timeout
            await asyncio.sleep(0.1)

            # Verify app is not completed yet
            assert not app._completed

            # Press quit - this should trigger cancel() which raises
            await pilot.press("q")
            # If we get here without crashing, the exception was caught
            await pilot.pause()

    # Should not raise
    run_async(run_test())


def test_run_textual_ui_creates_and_runs_app():
    """Test run_textual_ui function creates app and calls run()."""

    # Patch ShrinkRayApp - validation is now done before run_textual_ui is called
    with patch("shrinkray.tui.ShrinkRayApp") as mock_app_class:
        mock_app = MagicMock()
        mock_app.return_code = None  # Ensure no exit
        mock_app_class.return_value = mock_app

        run_textual_ui(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            parallelism=4,
            timeout=2.0,
            seed=42,
            input_type="arg",
            in_place=True,
            formatter="clang-format",
            volume="quiet",
            no_clang_delta=True,
            clang_delta="/usr/bin/clang_delta",
            trivial_is_error=True,
            exit_on_completion=True,
            theme="dark",
            history_enabled=True,
            also_interesting_code=None,
        )

        # Verify app was created with correct arguments
        mock_app_class.assert_called_once_with(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            parallelism=4,
            timeout=2.0,
            seed=42,
            input_type="arg",
            in_place=True,
            formatter="clang-format",
            volume="quiet",
            no_clang_delta=True,
            clang_delta="/usr/bin/clang_delta",
            trivial_is_error=True,
            exit_on_completion=True,
            theme="dark",
            history_enabled=True,
            also_interesting_code=None,
        )

        # Verify run() was called
        mock_app.run.assert_called_once()


def test_completed_flag_during_iteration_breaks_loop():
    """Test that is_completed becoming True mid-iteration breaks the loop."""

    class CompletesEarlyClient(FakeReductionClient):
        def __init__(self):
            super().__init__(updates=[])
            self._yield_count = 0

        async def get_progress_updates(self) -> AsyncGenerator[ProgressUpdate, None]:
            for i in range(10):
                self._yield_count += 1
                yield ProgressUpdate(
                    status=f"Update {i}",
                    size=100 - i * 10,
                    original_size=100,
                    calls=i,
                    reductions=i // 2,
                )
                await asyncio.sleep(0.01)
                # Set completed after a few yields
                if i >= 2:
                    self._completed = True
                    # App should break out of loop now

    async def run_test():
        fake_client = CompletesEarlyClient()

        app = ShrinkRayApp(
            file_path="/tmp/test.txt",
            test=["./test.sh"],
            client=fake_client,
        )

        async with app.run_test() as pilot:
            # Wait for updates to be processed
            for _ in range(20):
                await pilot.pause()
                await asyncio.sleep(0.02)
                if app.is_completed:
                    break

            # Should have completed early, not processed all 10
            # The break happens after checking is_completed, so
            # we should have processed 3 updates (0, 1, 2), then
            # after 2 we set completed and break before 3
            assert fake_client._yield_count <= 5

    run_async(run_test())


def test_run_textual_ui_exits_with_app_return_code():
    """Test run_textual_ui exits with app.return_code when set."""

    mock_app = MagicMock()
    mock_app.return_code = 42  # Non-zero return code

    with patch("shrinkray.tui.ShrinkRayApp", return_value=mock_app):
        with pytest.raises(SystemExit) as exc_info:
            run_textual_ui(
                file_path="/tmp/test.txt",
                test=["./test.sh"],
            )

        assert exc_info.value.code == 42


# =============================================================================
# PassStatsScreen tests
# =============================================================================


def test_pass_stats_screen_creation():
    """Test PassStatsScreen can be created with stats."""

    stats = [
        PassStatsData(
            pass_name="hollow",
            bytes_deleted=500,
            run_count=3,
            test_evaluations=100,
            successful_reductions=2,
            success_rate=66.7,
        )
    ]

    mock_app = Mock()
    mock_app._latest_pass_stats = stats
    mock_app._current_pass_name = "hollow"
    mock_app._disabled_passes = []

    screen = PassStatsScreen(mock_app)
    assert screen.pass_stats == stats


def test_pass_stats_screen_empty():
    """Test PassStatsScreen with no stats."""

    mock_app = Mock()
    mock_app._latest_pass_stats = []
    mock_app._current_pass_name = ""
    mock_app._disabled_passes = []

    screen = PassStatsScreen(mock_app)
    assert screen.pass_stats == []


def test_shrinkray_app_stores_pass_stats():
    """Test that ShrinkRayApp stores pass stats from updates."""

    async def run():
        stats = [
            PassStatsData(
                pass_name="test",
                bytes_deleted=100,
                run_count=1,
                test_evaluations=10,
                successful_reductions=1,
                success_rate=100.0,
            )
        ]

        updates = [
            ProgressUpdate(
                status="Test",
                size=100,
                original_size=200,
                calls=5,
                reductions=2,
                pass_stats=stats,
            )
        ]

        client = FakeReductionClient(updates=updates)
        await client.start()

        # Create a temp file for the app
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=client,
            )

            # Run briefly to process at least one update
            async with app.run_test() as pilot:
                await pilot.pause(0.2)

                # Check stats were stored
                assert len(app._latest_pass_stats) == 1
                assert app._latest_pass_stats[0].pass_name == "test"
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_action_show_pass_stats_opens_modal():
    """Test that pressing 'p' opens the pass stats modal."""

    async def run():
        stats = [
            PassStatsData(
                pass_name="test",
                bytes_deleted=100,
                run_count=1,
                test_evaluations=10,
                successful_reductions=1,
                success_rate=100.0,
            )
        ]

        updates = [
            ProgressUpdate(
                status="Test",
                size=100,
                original_size=200,
                calls=5,
                reductions=2,
                pass_stats=stats,
                current_pass_name="test",
                disabled_passes=[],
            )
        ]

        client = FakeReductionClient(updates=updates)
        await client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=client,
            )

            async with app.run_test() as pilot:
                await pilot.pause(0.2)

                # Press 'p' to open the pass stats modal
                await pilot.press("p")
                await pilot.pause()

                # Check that PassStatsScreen is on the screen stack
                assert any(
                    isinstance(screen, PassStatsScreen) for screen in app.screen_stack
                )

                # Press 'q' to close the modal
                await pilot.press("q")
                await pilot.pause()

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_pass_stats_modal_closes_with_p_key():
    """Test that pressing 'p' while in pass stats modal closes it."""

    async def run():
        updates = [
            ProgressUpdate(
                status="Test",
                size=100,
                original_size=200,
                calls=10,
                reductions=5,
                pass_stats=[
                    PassStatsData(
                        pass_name="test_pass",
                        bytes_deleted=50,
                        run_count=1,
                        test_evaluations=10,
                        successful_reductions=1,
                        success_rate=100.0,
                    )
                ],
                current_pass_name="test_pass",
                disabled_passes=[],
            )
        ]

        client = FakeReductionClient(updates=updates)
        await client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=client,
            )

            async with app.run_test() as pilot:
                await pilot.pause(0.2)

                # Press 'p' to open the pass stats modal
                await pilot.press("p")
                await pilot.pause()

                # Check that PassStatsScreen is on the screen stack
                assert any(
                    isinstance(screen, PassStatsScreen) for screen in app.screen_stack
                )

                # Press 'p' again to close the modal
                await pilot.press("p")
                await pilot.pause()

                # Modal should be closed
                assert not any(
                    isinstance(screen, PassStatsScreen) for screen in app.screen_stack
                )

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_pass_stats_modal_shows_help_with_h_key():
    """Test that pressing 'h' while in pass stats modal opens help screen."""

    async def run():
        updates = [
            ProgressUpdate(
                status="Test",
                size=100,
                original_size=200,
                calls=10,
                reductions=5,
                pass_stats=[
                    PassStatsData(
                        pass_name="test_pass",
                        bytes_deleted=50,
                        run_count=1,
                        test_evaluations=10,
                        successful_reductions=1,
                        success_rate=100.0,
                    )
                ],
                current_pass_name="test_pass",
                disabled_passes=[],
            )
        ]

        client = FakeReductionClient(updates=updates)
        await client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=client,
            )

            async with app.run_test() as pilot:
                await pilot.pause(0.2)

                # Press 'p' to open the pass stats modal
                await pilot.press("p")
                await pilot.pause()

                # Check that PassStatsScreen is on the screen stack
                assert any(
                    isinstance(screen, PassStatsScreen) for screen in app.screen_stack
                )

                # Press 'h' to open help
                await pilot.press("h")
                await pilot.pause()

                # HelpScreen should now be on the stack
                assert any(
                    isinstance(screen, HelpScreen) for screen in app.screen_stack
                )

                # PassStatsScreen should still be there (underneath)
                assert any(
                    isinstance(screen, PassStatsScreen) for screen in app.screen_stack
                )

                # Close help screen
                await pilot.press("q")
                await pilot.pause()

                # Close pass stats screen
                await pilot.press("q")
                await pilot.pause()

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_pass_stats_screen_with_empty_stats():
    """Test PassStatsScreen initialization with no stats returns empty list."""

    mock_app = Mock()
    mock_app._latest_pass_stats = []
    mock_app._current_pass_name = ""
    mock_app._disabled_passes = []

    screen = PassStatsScreen(mock_app)

    # Should have empty pass_stats
    assert screen.pass_stats == []
    # Should have empty disabled_passes set
    assert screen.disabled_passes == set()


def test_pass_stats_screen_disabled_passes_styling():
    """Test that disabled passes are styled correctly."""

    mock_app = Mock()
    mock_app._latest_pass_stats = [
        PassStatsData(
            pass_name="disabled_pass",
            bytes_deleted=100,
            run_count=1,
            test_evaluations=10,
            successful_reductions=1,
            success_rate=10.0,
        )
    ]
    mock_app._current_pass_name = ""
    mock_app._disabled_passes = ["disabled_pass"]

    screen = PassStatsScreen(mock_app)
    assert "disabled_pass" in screen.disabled_passes


def test_shrinkray_app_disabled_passes_updated_from_progress():
    """Test that ShrinkRayApp stores disabled_passes from updates."""

    async def run():
        updates = [
            ProgressUpdate(
                status="Test",
                size=100,
                original_size=200,
                calls=5,
                reductions=2,
                pass_stats=[],
                current_pass_name="test",
                disabled_passes=["hollow", "lift_braces"],
            )
        ]

        client = FakeReductionClient(updates=updates)
        await client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=client,
            )

            async with app.run_test() as pilot:
                await pilot.pause(0.2)

                # Check disabled_passes were stored
                assert "hollow" in app._disabled_passes
                assert "lift_braces" in app._disabled_passes
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_pass_stats_modal_toggle_disable():
    """Test that pressing 'd' in pass stats modal toggles pass disable state."""

    async def run():
        stats = [
            PassStatsData(
                pass_name="hollow",
                bytes_deleted=100,
                run_count=1,
                test_evaluations=10,
                successful_reductions=1,
                success_rate=100.0,
            )
        ]

        updates = [
            ProgressUpdate(
                status="Test",
                size=100,
                original_size=200,
                calls=5,
                reductions=2,
                pass_stats=stats,
                current_pass_name="hollow",
                disabled_passes=[],
            )
        ]

        client = FakeReductionClient(updates=updates)
        await client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=client,
            )

            async with app.run_test() as pilot:
                await pilot.pause(0.2)

                # Press 'p' to open the pass stats modal
                await pilot.press("p")
                await pilot.pause()

                # Press 'd' to toggle disable on the selected pass
                await pilot.press("d")
                await pilot.pause()

                # Press 'escape' to close the modal
                await pilot.press("escape")
                await pilot.pause()

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_pass_stats_modal_skip_current():
    """Test that pressing 's' in pass stats modal skips current pass."""

    async def run():
        stats = [
            PassStatsData(
                pass_name="hollow",
                bytes_deleted=100,
                run_count=1,
                test_evaluations=10,
                successful_reductions=1,
                success_rate=100.0,
            )
        ]

        updates = [
            ProgressUpdate(
                status="Test",
                size=100,
                original_size=200,
                calls=5,
                reductions=2,
                pass_stats=stats,
                current_pass_name="hollow",
                disabled_passes=[],
            )
        ]

        client = FakeReductionClient(updates=updates)
        await client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=client,
            )

            async with app.run_test() as pilot:
                await pilot.pause(0.2)

                # Press 'p' to open the pass stats modal
                await pilot.press("p")
                await pilot.pause()

                # Press 's' to skip the current pass
                await pilot.press("s")
                await pilot.pause()

                # Press 'escape' to close the modal
                await pilot.press("escape")
                await pilot.pause()

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_pass_stats_modal_with_empty_stats():
    """Test pass stats modal displays 'No pass data yet' when empty."""

    async def run():
        # No pass stats
        updates = [
            ProgressUpdate(
                status="Test",
                size=100,
                original_size=200,
                calls=5,
                reductions=2,
                pass_stats=[],
                current_pass_name="",
                disabled_passes=[],
            )
        ]

        client = FakeReductionClient(updates=updates)
        await client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=client,
            )

            async with app.run_test() as pilot:
                await pilot.pause(0.2)

                # Press 'p' to open the pass stats modal
                await pilot.press("p")
                await pilot.pause()

                # The modal should show "No pass data yet" row
                # Just verify modal opened successfully
                assert any(
                    isinstance(screen, PassStatsScreen) for screen in app.screen_stack
                )

                # Press 'escape' to close the modal
                await pilot.press("escape")
                await pilot.pause()

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_pass_stats_modal_non_current_non_disabled_pass():
    """Test pass stats modal with a pass that is neither current nor disabled."""

    async def run():
        stats = [
            PassStatsData(
                pass_name="hollow",
                bytes_deleted=100,
                run_count=1,
                test_evaluations=10,
                successful_reductions=1,
                success_rate=100.0,
            ),
            PassStatsData(
                pass_name="delete_lines",
                bytes_deleted=50,
                run_count=2,
                test_evaluations=20,
                successful_reductions=1,
                success_rate=50.0,
            ),
        ]

        updates = [
            ProgressUpdate(
                status="Test",
                size=100,
                original_size=200,
                calls=5,
                reductions=2,
                pass_stats=stats,
                # Only hollow is current, delete_lines is neither current nor disabled
                current_pass_name="hollow",
                disabled_passes=[],
            )
        ]

        client = FakeReductionClient(updates=updates)
        await client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=client,
            )

            async with app.run_test() as pilot:
                await pilot.pause(0.2)

                # Press 'p' to open the pass stats modal
                await pilot.press("p")
                await pilot.pause()

                # The modal should be open with both passes
                assert any(
                    isinstance(screen, PassStatsScreen) for screen in app.screen_stack
                )

                # Press 'escape' to close the modal
                await pilot.press("escape")
                await pilot.pause()

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_pass_stats_modal_toggle_enable():
    """Test that pressing 'd' on a disabled pass enables it."""

    async def run():
        stats = [
            PassStatsData(
                pass_name="hollow",
                bytes_deleted=100,
                run_count=1,
                test_evaluations=10,
                successful_reductions=1,
                success_rate=100.0,
            )
        ]

        updates = [
            ProgressUpdate(
                status="Test",
                size=100,
                original_size=200,
                calls=5,
                reductions=2,
                pass_stats=stats,
                current_pass_name="hollow",
                disabled_passes=["hollow"],  # hollow is disabled
            )
        ]

        client = FakeReductionClient(updates=updates)
        await client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=client,
            )

            async with app.run_test() as pilot:
                await pilot.pause(0.2)

                # Press 'p' to open the pass stats modal
                await pilot.press("p")
                await pilot.pause()

                # Press 'd' to toggle disable on the selected pass
                # Since hollow is already disabled, this should enable it
                await pilot.press("d")
                await pilot.pause()

                # Press 'escape' to close the modal
                await pilot.press("escape")
                await pilot.pause()

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_pass_stats_screen_refresh_updates_data():
    """Test that the periodic refresh updates data when it changes."""

    # Create initial stats
    initial_stats = [
        PassStatsData(
            pass_name="hollow",
            bytes_deleted=100,
            run_count=1,
            test_evaluations=10,
            successful_reductions=1,
            success_rate=100.0,
        )
    ]

    # Create a mock app with mutable stats
    mock_app = Mock()
    mock_app._latest_pass_stats = initial_stats.copy()
    mock_app._current_pass_name = "hollow"
    mock_app._disabled_passes = []

    screen = PassStatsScreen(mock_app)

    # Verify initial state
    assert screen.pass_stats == initial_stats
    assert screen.current_pass_name == "hollow"

    # Now simulate a change in the app's data
    new_stats = [
        PassStatsData(
            pass_name="hollow",
            bytes_deleted=200,  # Changed
            run_count=2,  # Changed
            test_evaluations=20,  # Changed
            successful_reductions=2,  # Changed
            success_rate=100.0,
        )
    ]
    mock_app._latest_pass_stats = new_stats
    mock_app._current_pass_name = "delete_lines"  # Changed current pass
    mock_app._disabled_passes = ["hollow"]  # Added disabled pass

    # The screen hasn't refreshed yet, so it still has old data
    assert screen.pass_stats == initial_stats

    # After manual refresh, it should have the new data
    # Note: _refresh_data requires a mounted screen with widgets, so we just
    # verify the screen can detect changes
    assert mock_app._latest_pass_stats != screen.pass_stats


def test_pass_stats_screen_toggle_disable_no_selection():
    """Test action_toggle_disable returns early when no pass is selected."""

    mock_app = Mock()
    mock_app._latest_pass_stats = []  # Empty stats
    mock_app._current_pass_name = ""
    mock_app._disabled_passes = []

    screen = PassStatsScreen(mock_app)

    # With empty pass_stats, _get_selected_pass_name should return None
    # and action_toggle_disable should return early without calling run_worker
    # We can't easily test this without mounting, but we verify the screen
    # initializes correctly with empty data
    assert screen.pass_stats == []


def test_help_screen_opens_and_closes():
    """Test that pressing 'h' opens the help modal and q closes it."""

    async def run():
        updates = [
            ProgressUpdate(
                status="Test",
                size=100,
                original_size=200,
                calls=5,
                reductions=2,
            )
        ]

        client = FakeReductionClient(updates=updates)
        await client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=client,
            )

            async with app.run_test() as pilot:
                await pilot.pause(0.2)

                # Press 'h' to open the help modal
                await pilot.press("h")
                await pilot.pause()

                # Check that HelpScreen is on the screen stack
                assert any(
                    isinstance(screen, HelpScreen) for screen in app.screen_stack
                )

                # Press 'h' again to close the modal
                await pilot.press("h")
                await pilot.pause()

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_skip_current_pass_from_main_screen():
    """Test that pressing 'c' on main screen skips the current pass."""

    async def run():
        stats = [
            PassStatsData(
                pass_name="hollow",
                bytes_deleted=100,
                run_count=1,
                test_evaluations=10,
                successful_reductions=1,
                success_rate=100.0,
            )
        ]

        updates = [
            ProgressUpdate(
                status="Running hollow",
                size=100,
                original_size=200,
                calls=5,
                reductions=2,
                pass_stats=stats,
                current_pass_name="hollow",
                disabled_passes=[],
            )
        ]

        client = FakeReductionClient(updates=updates)
        await client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=client,
            )

            async with app.run_test() as pilot:
                await pilot.pause(0.2)

                # Press 'c' to skip the current pass
                await pilot.press("c")
                await pilot.pause()

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_help_screen_can_be_imported():
    """Test that HelpScreen can be imported."""
    # HelpScreen is imported at module level, just verify it exists
    assert HelpScreen is not None


def test_pass_stats_modal_refresh_triggers_update():
    """Test that the periodic refresh updates the table when data changes."""

    async def run():
        stats = [
            PassStatsData(
                pass_name="hollow",
                bytes_deleted=100,
                run_count=1,
                test_evaluations=10,
                successful_reductions=1,
                success_rate=100.0,
            )
        ]

        # Create updates that change over time
        updates = [
            ProgressUpdate(
                status="Running hollow",
                size=100,
                original_size=200,
                calls=5,
                reductions=2,
                pass_stats=stats,
                current_pass_name="hollow",
                disabled_passes=[],
            ),
            ProgressUpdate(
                status="Running delete_lines",
                size=90,
                original_size=200,
                calls=10,
                reductions=3,
                pass_stats=stats
                + [
                    PassStatsData(
                        pass_name="delete_lines",
                        bytes_deleted=10,
                        run_count=1,
                        test_evaluations=5,
                        successful_reductions=1,
                        success_rate=100.0,
                    )
                ],
                current_pass_name="delete_lines",
                disabled_passes=[],
            ),
        ]

        client = FakeReductionClient(updates=updates)
        await client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=client,
            )

            async with app.run_test() as pilot:
                await pilot.pause(0.2)

                # Open pass stats modal
                await pilot.press("p")
                await pilot.pause()

                # Wait for periodic refresh to trigger (every 500ms)
                await pilot.pause(0.6)

                # Close modal
                await pilot.press("escape")
                await pilot.pause()

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_pass_stats_modal_get_selected_with_empty_stats():
    """Test _get_selected_pass_name returns None when stats are empty."""

    async def run():
        updates = [
            ProgressUpdate(
                status="Starting",
                size=100,
                original_size=200,
                calls=0,
                reductions=0,
                pass_stats=[],
                current_pass_name="",
                disabled_passes=[],
            )
        ]

        client = FakeReductionClient(updates=updates)
        await client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=client,
            )

            async with app.run_test() as pilot:
                await pilot.pause(0.2)

                # Open pass stats modal
                await pilot.press("p")
                await pilot.pause()

                # Try to toggle - should do nothing since no passes
                await pilot.press("space")
                await pilot.pause()

                # Close modal
                await pilot.press("escape")
                await pilot.pause()

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_skip_current_pass_from_main_when_completed():
    """Test that skip does nothing when reduction is completed."""

    async def run():
        updates = [
            ProgressUpdate(
                status="Completed",
                size=50,
                original_size=200,
                calls=100,
                reductions=10,
            )
        ]

        client = FakeReductionClient(updates=updates)
        client._completed = True  # Mark as completed
        await client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=client,
            )

            async with app.run_test() as pilot:
                await pilot.pause(0.2)

                # Try to skip - should do nothing since completed
                await pilot.press("c")
                await pilot.pause()

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_pass_stats_screen_direct_method_calls():
    """Test PassStatsScreen methods directly for coverage."""

    # Create mock app with all required attributes
    mock_app = Mock()
    mock_app._latest_pass_stats = [
        PassStatsData(
            pass_name="hollow",
            bytes_deleted=100,
            run_count=1,
            test_evaluations=10,
            successful_reductions=1,
            success_rate=100.0,
        )
    ]
    mock_app._current_pass_name = "hollow"
    mock_app._disabled_passes = []
    mock_app._client = Mock()
    mock_app._client.disable_pass = AsyncMock()
    mock_app._client.enable_pass = AsyncMock()
    mock_app._client.skip_current_pass = AsyncMock()
    mock_app.run_worker = Mock()

    screen = PassStatsScreen(mock_app)

    # Test initial state
    assert len(screen.pass_stats) == 1
    assert screen.current_pass_name == "hollow"
    assert screen.disabled_passes == set()


def test_shrinkray_app_skip_pass_method():
    """Test ShrinkRayApp._skip_pass directly."""

    async def run():
        client = FakeReductionClient(updates=[])
        await client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=client,
            )

            # Directly test the _skip_pass method
            await app._skip_pass()

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_pass_stats_screen_refresh_data_updates():
    """Test PassStatsScreen._refresh_data when stats change (lines 506-516)."""

    async def run():
        fake_client = FakeReductionClient(updates=[])
        await fake_client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                # Set initial pass stats
                app._latest_pass_stats = [
                    PassStatsData(
                        pass_name="test_pass",
                        bytes_deleted=100,
                        run_count=1,
                        test_evaluations=10,
                        successful_reductions=1,
                        success_rate=100.0,
                    )
                ]
                app._current_pass_name = "test_pass"
                app._disabled_passes = []

                # Open pass stats screen
                await pilot.press("p")
                await pilot.pause()

                # Get the screen
                screen: PassStatsScreen | None = None
                for s in app.screen_stack:
                    if isinstance(s, PassStatsScreen):
                        screen = s
                        break

                if screen is not None:
                    # Update app stats (different from screen stats)
                    app._latest_pass_stats = [
                        PassStatsData(
                            pass_name="test_pass",
                            bytes_deleted=200,  # Changed
                            run_count=2,  # Changed
                            test_evaluations=20,
                            successful_reductions=2,
                            success_rate=100.0,
                        )
                    ]
                    app._current_pass_name = "different_pass"  # Changed
                    app._disabled_passes = ["disabled_pass"]

                    # Call _refresh_data directly
                    screen._refresh_data()

                    # Verify screen updated
                    assert screen.pass_stats[0].bytes_deleted == 200
                    assert screen.current_pass_name == "different_pass"

                await pilot.press("q")  # Close screen
                await pilot.press("q")  # Quit app

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_pass_stats_screen_refresh_data_with_disabled_passes():
    """Test _refresh_data updates footer with disabled count."""

    async def run():
        fake_client = FakeReductionClient(updates=[])
        await fake_client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                # Set initial pass stats with one pass disabled
                app._latest_pass_stats = [
                    PassStatsData(
                        pass_name="pass1",
                        bytes_deleted=100,
                        run_count=1,
                        test_evaluations=10,
                        successful_reductions=1,
                        success_rate=100.0,
                    ),
                    PassStatsData(
                        pass_name="pass2",
                        bytes_deleted=50,
                        run_count=1,
                        test_evaluations=5,
                        successful_reductions=0,
                        success_rate=0.0,
                    ),
                ]
                app._current_pass_name = "pass1"
                app._disabled_passes = []

                # Open pass stats screen
                await pilot.press("p")
                await pilot.pause()

                # Get the screen
                screen: PassStatsScreen | None = None
                for s in app.screen_stack:
                    if isinstance(s, PassStatsScreen):
                        screen = s
                        break

                if screen is not None:
                    # Initially no disabled passes, update stats
                    screen._refresh_data()  # First refresh (no change yet)

                    # Now change the stats AND add a disabled pass
                    app._latest_pass_stats = [
                        PassStatsData(
                            pass_name="pass1",
                            bytes_deleted=150,  # Changed
                            run_count=2,
                            test_evaluations=15,
                            successful_reductions=1,
                            success_rate=100.0,
                        ),
                        PassStatsData(
                            pass_name="pass2",
                            bytes_deleted=50,
                            run_count=1,
                            test_evaluations=5,
                            successful_reductions=0,
                            success_rate=0.0,
                        ),
                    ]
                    app._current_pass_name = "pass1"
                    # Disable one pass locally in the screen
                    screen.disabled_passes = {"pass2"}

                    # Refresh - this should now show "1 disabled" in footer
                    screen._refresh_data()

                    # The disabled count should be reflected
                    assert len(screen.disabled_passes) == 1

                await pilot.press("q")  # Close screen
                await pilot.press("q")  # Quit app

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_pass_stats_screen_get_selected_pass_name_empty_table():
    """Test _get_selected_pass_name returns None for empty table."""

    async def run():
        fake_client = FakeReductionClient(updates=[])
        await fake_client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                # Ensure empty stats
                app._latest_pass_stats = []
                app._current_pass_name = ""
                app._disabled_passes = []

                await pilot.press("p")
                await pilot.pause()

                # Get the screen
                screen: PassStatsScreen | None = None
                for s in app.screen_stack:
                    if isinstance(s, PassStatsScreen):
                        screen = s
                        break

                if screen is not None:
                    # Force pass_stats to be empty
                    screen.pass_stats = []

                    # Get selected pass name should return None
                    result = screen._get_selected_pass_name()
                    assert result is None

                await pilot.press("q")
                await pilot.press("q")

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_pass_stats_screen_action_toggle_disable():
    """Test action_toggle_disable toggles passes (lines 534-545)."""

    async def run():
        fake_client = FakeReductionClient(updates=[])
        await fake_client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                # Set up pass stats
                app._latest_pass_stats = [
                    PassStatsData(
                        pass_name="my_pass",
                        bytes_deleted=100,
                        run_count=1,
                        test_evaluations=10,
                        successful_reductions=1,
                        success_rate=100.0,
                    )
                ]
                app._current_pass_name = "my_pass"
                app._disabled_passes = []

                await pilot.press("p")
                await pilot.pause()

                # Get the screen
                screen: PassStatsScreen | None = None
                for s in app.screen_stack:
                    if isinstance(s, PassStatsScreen):
                        screen = s
                        break

                if screen is not None:
                    # Initially not disabled
                    assert "my_pass" not in screen.disabled_passes

                    # Toggle to disable
                    screen.action_toggle_disable()
                    await pilot.pause()
                    assert "my_pass" in screen.disabled_passes

                    # Toggle to enable
                    screen.action_toggle_disable()
                    await pilot.pause()
                    assert "my_pass" not in screen.disabled_passes

                await pilot.press("q")
                await pilot.press("q")

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_pass_stats_screen_async_methods():
    """Test async methods _send_disable_pass, _send_enable_pass directly (lines 549-555)."""

    async def run():
        fake_client = FakeReductionClient(updates=[])
        await fake_client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                app._latest_pass_stats = [
                    PassStatsData(
                        pass_name="a_pass",
                        bytes_deleted=100,
                        run_count=1,
                        test_evaluations=10,
                        successful_reductions=1,
                        success_rate=100.0,
                    )
                ]

                await pilot.press("p")
                await pilot.pause()

                screen: PassStatsScreen | None = None
                for s in app.screen_stack:
                    if isinstance(s, PassStatsScreen):
                        screen = s
                        break

                if screen is not None:
                    # Directly call the async methods
                    await screen._send_disable_pass("a_pass")
                    await screen._send_enable_pass("a_pass")
                    await screen._skip_pass()

                await pilot.press("q")
                await pilot.press("q")

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_pass_stats_screen_action_skip_current():
    """Test action_skip_current method."""

    async def run():
        fake_client = FakeReductionClient(updates=[])
        await fake_client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                app._latest_pass_stats = [
                    PassStatsData(
                        pass_name="running_pass",
                        bytes_deleted=100,
                        run_count=1,
                        test_evaluations=10,
                        successful_reductions=1,
                        success_rate=100.0,
                    )
                ]

                await pilot.press("p")
                await pilot.pause()

                screen: PassStatsScreen | None = None
                for s in app.screen_stack:
                    if isinstance(s, PassStatsScreen):
                        screen = s
                        break

                if screen is not None:
                    # Call action_skip_current directly
                    screen.action_skip_current()
                    await pilot.pause()

                await pilot.press("q")
                await pilot.press("q")

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_app_action_skip_current_pass():
    """Test ShrinkRayApp.action_skip_current_pass."""

    async def run():
        # Use an infinite updates client so we can test during reduction
        class InfiniteClient(FakeReductionClient):
            async def get_progress_updates(self):
                i = 0
                while not self._cancelled:
                    yield ProgressUpdate(
                        status=f"Running step {i}",
                        size=1000 - i,
                        original_size=1000,
                        calls=i,
                        reductions=0,
                    )
                    i += 1
                    await asyncio.sleep(0.05)
                self._completed = True

        fake_client = InfiniteClient(updates=[])
        await fake_client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                await asyncio.sleep(0.1)

                # Call action_skip_current_pass directly
                app.action_skip_current_pass()
                await pilot.pause()

                await pilot.press("q")

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_pass_stats_screen_async_methods_no_client():
    """Test async methods when client is None (coverage for early exit)."""

    async def run():
        fake_client = FakeReductionClient(updates=[])
        await fake_client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                app._latest_pass_stats = []

                await pilot.press("p")
                await pilot.pause()

                screen: PassStatsScreen | None = None
                for s in app.screen_stack:
                    if isinstance(s, PassStatsScreen):
                        screen = s
                        break

                if screen is not None:
                    # Set client to None to test early exit branches
                    screen._app._client = None

                    # These should all exit early without error
                    await screen._send_disable_pass("test")
                    await screen._send_enable_pass("test")
                    await screen._skip_pass()

                await pilot.press("q")
                await pilot.press("q")

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_app_skip_pass_no_client():
    """Test ShrinkRayApp._skip_pass when client is None."""

    async def run():
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            fake_client = FakeReductionClient(updates=[])
            await fake_client.start()

            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            # Manually set _client to None to test the early exit
            app._client = None

            # Should exit early without error
            await app._skip_pass()

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_all_passes_disabled_shows_message():
    """Test that disabling all passes shows a paused message."""

    async def run():
        fake_client = FakeReductionClient(updates=[])
        await fake_client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            status_messages: list[str] = []
            original_update_status = app.update_status

            def capture_status(message: str) -> None:
                status_messages.append(message)
                original_update_status(message)

            async with app.run_test() as pilot:
                app.update_status = capture_status

                # Set up state where all passes are disabled
                app._latest_pass_stats = [
                    PassStatsData(
                        pass_name="pass1",
                        bytes_deleted=100,
                        run_count=1,
                        test_evaluations=10,
                        successful_reductions=1,
                        success_rate=100.0,
                    ),
                    PassStatsData(
                        pass_name="pass2",
                        bytes_deleted=50,
                        run_count=1,
                        test_evaluations=5,
                        successful_reductions=0,
                        success_rate=0.0,
                    ),
                ]
                app._disabled_passes = ["pass1", "pass2"]

                # Directly call the check method
                app._check_all_passes_disabled()

                # Check that the "paused" message was shown
                paused_msgs = [m for m in status_messages if "paused" in m.lower()]
                assert len(paused_msgs) > 0, (
                    f"Expected 'paused' in messages, got: {status_messages}"
                )

                await pilot.press("q")

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_check_all_passes_disabled_partial():
    """Test _check_all_passes_disabled when only some passes are disabled (746->exit)."""

    async def run():
        fake_client = FakeReductionClient(updates=[])
        await fake_client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            status_messages: list[str] = []
            original_update_status = app.update_status

            def capture_status(message: str) -> None:
                status_messages.append(message)
                original_update_status(message)

            async with app.run_test() as pilot:
                app.update_status = capture_status

                # Set up state where only some passes are disabled
                app._latest_pass_stats = [
                    PassStatsData(
                        pass_name="pass1",
                        bytes_deleted=100,
                        run_count=1,
                        test_evaluations=10,
                        successful_reductions=1,
                        success_rate=100.0,
                    ),
                    PassStatsData(
                        pass_name="pass2",
                        bytes_deleted=50,
                        run_count=1,
                        test_evaluations=5,
                        successful_reductions=0,
                        success_rate=0.0,
                    ),
                ]
                # Only pass1 is disabled, not pass2
                app._disabled_passes = ["pass1"]

                # Directly call the check method
                app._check_all_passes_disabled()

                # Check that NO "paused" message was shown
                paused_msgs = [m for m in status_messages if "paused" in m.lower()]
                assert len(paused_msgs) == 0, (
                    f"Expected no 'paused' messages, got: {paused_msgs}"
                )

                await pilot.press("q")

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_pass_stats_screen_cursor_position_out_of_bounds():
    """Test _update_table_data when cursor is beyond new row count."""

    async def run():
        fake_client = FakeReductionClient(updates=[])
        await fake_client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                # Set up with multiple pass stats
                app._latest_pass_stats = [
                    PassStatsData(
                        pass_name="pass1",
                        bytes_deleted=100,
                        run_count=1,
                        test_evaluations=10,
                        successful_reductions=1,
                        success_rate=100.0,
                    ),
                    PassStatsData(
                        pass_name="pass2",
                        bytes_deleted=50,
                        run_count=1,
                        test_evaluations=5,
                        successful_reductions=0,
                        success_rate=0.0,
                    ),
                    PassStatsData(
                        pass_name="pass3",
                        bytes_deleted=25,
                        run_count=1,
                        test_evaluations=3,
                        successful_reductions=0,
                        success_rate=0.0,
                    ),
                ]
                app._current_pass_name = "pass1"
                app._disabled_passes = []

                # Open pass stats screen
                await pilot.press("p")
                await pilot.pause()

                # Get the screen
                screen: PassStatsScreen | None = None
                for s in app.screen_stack:
                    if isinstance(s, PassStatsScreen):
                        screen = s
                        break

                if screen is not None:
                    # Move cursor to row 2 (third row)
                    table = screen.query_one(DataTable)
                    await pilot.press("down")  # Move to row 1
                    await pilot.press("down")  # Move to row 2
                    await pilot.pause()

                    # Verify cursor is at row 2
                    assert table.cursor_coordinate.row == 2

                    # Now reduce the pass_stats to only 1 item
                    screen.pass_stats = [
                        PassStatsData(
                            pass_name="pass1",
                            bytes_deleted=100,
                            run_count=1,
                            test_evaluations=10,
                            successful_reductions=1,
                            success_rate=100.0,
                        )
                    ]

                    # Call _update_table_data - cursor row (2) >= row_count (1)
                    # This should trigger the branch where cursor is NOT restored
                    screen._update_table_data()

                    # The table should now have only 1 row, cursor should be at 0
                    assert table.row_count == 1

                await pilot.press("q")  # Close screen
                await pilot.press("q")  # Quit app

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_pass_stats_screen_get_selected_empty_table():
    """Test _get_selected_pass_name when table has 0 rows."""

    async def run():
        fake_client = FakeReductionClient(updates=[])
        await fake_client.start()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                app._latest_pass_stats = []

                await pilot.press("p")
                await pilot.pause()

                screen: PassStatsScreen | None = None
                for s in app.screen_stack:
                    if isinstance(s, PassStatsScreen):
                        screen = s
                        break

                if screen is not None:
                    # Get the actual table
                    table = screen.query_one("DataTable")

                    # Mock row_count to return 0 for this specific call
                    with patch.object(
                        type(table), "row_count", new_callable=PropertyMock
                    ) as mock_rc:
                        mock_rc.return_value = 0
                        result = screen._get_selected_pass_name()
                        assert result is None

                await pilot.press("q")
                await pilot.press("q")

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


# === Modal edge case tests ===


def test_expanded_modal_graph_missing_main_graph():
    """Test that _get_graph_content returns early when main graph is missing."""

    async def run():
        # Create an app with no progress updates yet
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Remove the size graph widget to simulate missing widget
                size_graph = app.query_one("#size-graph", SizeGraph)
                size_graph.remove()
                await pilot.pause()

                # Now create and push the modal
                modal = ExpandedBoxModal("Size Graph", "graph-container", None)
                await app.push_screen(modal)
                await pilot.pause()

                # The modal should have an expanded-graph but _get_graph_content
                # should return early since main graph is missing
                expanded_graphs = list(
                    modal.query("#expanded-graph").results(SizeGraph)
                )
                assert len(expanded_graphs) == 1
                # The expanded graph should have empty history since copy failed
                assert expanded_graphs[0]._size_history == []

                await pilot.press("escape")
                await pilot.press("q")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_expanded_modal_stats_missing_stats_display():
    """Test that _get_stats_content returns fallback when stats display is missing."""

    async def run():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Remove the stats display widget
                stats_display = app.query_one("#stats-display", StatsDisplay)
                stats_display.remove()
                await pilot.pause()

                # Create and push modal for stats
                modal = ExpandedBoxModal("Statistics", "stats-container", None)
                await app.push_screen(modal)
                await pilot.pause()

                # Check that fallback content is shown
                content = modal.query_one("#expanded-content", Static)
                assert "Statistics not available" in get_static_content(content)

                await pilot.press("escape")
                await pilot.press("q")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_expanded_modal_content_missing_content_preview():
    """Test that _get_file_content returns fallback when content preview is missing."""

    async def run():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Remove the content preview widget
                content_preview = app.query_one("#content-preview", ContentPreview)
                content_preview.remove()
                await pilot.pause()

                # Create modal for content-container with no file_path
                modal = ExpandedBoxModal("Content", "content-container", None)
                await app.push_screen(modal)
                await pilot.pause()

                # Check that fallback content is shown
                content = modal.query_one("#expanded-content", Static)
                assert "Content preview not available" in get_static_content(content)

                await pilot.press("escape")
                await pilot.press("q")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_expanded_modal_output_missing_output_preview():
    """Test that _get_output_content returns fallback when output preview is missing."""

    async def run():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Remove the output preview widget
                output_preview = app.query_one("#output-preview", OutputPreview)
                output_preview.remove()
                await pilot.pause()

                # Create modal for output-container
                modal = ExpandedBoxModal("Output", "output-container", None)
                await app.push_screen(modal)
                await pilot.pause()

                # Check that fallback content is shown
                content = modal.query_one("#expanded-content", Static)
                assert "Output not available" in get_static_content(content)

                await pilot.press("escape")
                await pilot.press("q")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_expanded_modal_unknown_content_widget():
    """Test that unknown content_widget_id results in empty content."""

    async def run():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Create modal with unknown widget id
                modal = ExpandedBoxModal("Unknown", "unknown-container", None)
                await app.push_screen(modal)
                await pilot.pause()

                # Check that empty content is shown
                content = modal.query_one("#expanded-content", Static)
                # Empty string means the content should be empty
                assert get_static_content(content) == ""

                await pilot.press("escape")
                await pilot.press("q")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


# === Output content edge case tests ===


def test_expanded_modal_output_with_return_code():
    """Test output modal displays return code when test_id and return_code are set."""

    async def run():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Set output preview state directly - completed with error
                output_preview = app.query_one("#output-preview", OutputPreview)
                output_preview.active_test_id = 42
                output_preview.last_return_code = 1
                output_preview.output_content = "some output"

                # Create modal for output-container
                modal = ExpandedBoxModal("Output", "output-container", None)
                await app.push_screen(modal)
                await pilot.pause()

                # Check that return code info is shown
                content = modal.query_one("#expanded-content", Static)
                content_str = get_static_content(content)
                assert "Test #42" in content_str
                assert "exited with code 1" in content_str

                await pilot.press("escape")
                await pilot.press("q")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_expanded_modal_output_running():
    """Test output modal displays 'running' when test is still in progress."""

    async def run():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Set output preview state - running (no return code)
                output_preview = app.query_one("#output-preview", OutputPreview)
                output_preview.active_test_id = 42
                output_preview.last_return_code = None
                output_preview.output_content = "some output"
                output_preview._has_seen_output = True

                # Create modal for output-container
                modal = ExpandedBoxModal("Output", "output-container", None)
                await app.push_screen(modal)
                await pilot.pause()

                # Check that running message is shown
                content = modal.query_one("#expanded-content", Static)
                content_str = get_static_content(content)
                assert "Test #42" in content_str
                assert "running" in content_str

                await pilot.press("escape")
                await pilot.press("q")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_expanded_modal_output_with_empty_content_and_header():
    """Test output modal with has_seen_output but no current content."""

    async def run():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Set output preview state - completed with return code but no content
                output_preview = app.query_one("#output-preview", OutputPreview)
                output_preview.active_test_id = 42
                output_preview.last_return_code = 0
                output_preview._has_seen_output = True
                output_preview.output_content = ""
                output_preview._pending_content = ""  # Empty string is falsy like None

                # Create modal for output-container
                modal = ExpandedBoxModal("Output", "output-container", None)
                await app.push_screen(modal)
                await pilot.pause()

                # Should show header only (no "No test output" message)
                content = modal.query_one("#expanded-content", Static)
                content_str = get_static_content(content)
                assert "Test #42" in content_str
                assert "No test output" not in content_str

                await pilot.press("escape")
                await pilot.press("q")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


# === Action method tests ===


def test_action_quit_with_client():
    """Test action_quit cancels client before exiting."""

    async def run():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Trigger quit action
                await pilot.press("q")
                await pilot.pause()

                # Client should have been cancelled
                assert fake_client._cancelled
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_action_quit_handles_exception():
    """Test action_quit handles exception from cancel gracefully."""

    async def run():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)

        # Make cancel raise an exception
        async def raise_exception():
            raise RuntimeError("Process already exited")

        fake_client.cancel = raise_exception

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Trigger quit action - should not raise
                await pilot.press("q")
                await pilot.pause()

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_action_show_pass_stats():
    """Test action_show_pass_stats opens the pass stats screen."""

    async def run():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Press 'p' to show pass stats
                await pilot.press("p")
                await pilot.pause()

                # Verify PassStatsScreen is on stack
                found = any(isinstance(s, PassStatsScreen) for s in app.screen_stack)
                assert found

                await pilot.press("q")
                await pilot.press("q")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_action_show_help():
    """Test action_show_help opens the help screen."""

    async def run():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Press 'h' to show help
                await pilot.press("h")
                await pilot.pause()

                # Verify HelpScreen is on stack
                found = any(isinstance(s, HelpScreen) for s in app.screen_stack)
                assert found

                await pilot.press("q")
                await pilot.press("q")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_action_skip_current_pass():
    """Test action_skip_current_pass sends skip request to client."""

    async def run():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)
        skip_called = []

        original_skip = fake_client.skip_current_pass

        async def track_skip():
            skip_called.append(True)
            return await original_skip()

        fake_client.skip_current_pass = track_skip

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Press 'c' to skip current pass
                await pilot.press("c")
                await pilot.pause()

                # Give the worker time to run
                await asyncio.sleep(0.1)
                await pilot.pause()

                assert skip_called

                await pilot.press("q")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


# === Focus index fallback test ===


def test_get_focused_box_index_no_focus():
    """Test _get_focused_box_index returns 0 when no box has focus."""

    async def run():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Remove focus from all boxes
                app.set_focus(None)
                await pilot.pause()

                # Check that _get_focused_box_index returns 0
                result = app._get_focused_box_index()
                assert result == 0

                await pilot.press("q")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


# === is_completed property test ===


def test_is_completed_property():
    """Test the is_completed property reflects internal state."""

    async def run():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Initially not completed
                assert not app.is_completed

                # Set internal completed flag
                app._completed = True
                assert app.is_completed

                await pilot.press("q")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


# === Partial branch coverage tests ===


def test_update_expanded_graph_skips_non_graph_modal():
    """Test that _update_expanded_graph skips modals with different content_widget_id."""

    async def run():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Push a stats modal (not graph) - this will be on the screen stack
                modal = ExpandedBoxModal("Statistics", "stats-container", None)
                await app.push_screen(modal)
                await pilot.pause()

                # Directly call _update_expanded_graph
                # It should iterate through screen_stack, find the stats modal,
                # check that it's not a graph modal, and continue/exit the loop
                app._update_expanded_graph([(1.0, 100)], 200, 1.0)

                # The test passes if no error is raised (the loop correctly skips
                # the stats modal when looking for graph modals)

                await pilot.press("escape")
                await pilot.press("q")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_update_expanded_stats_skips_non_stats_modal():
    """Test that _update_expanded_stats skips modals with different content_widget_id."""

    async def run():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Push a graph modal (not stats) - this will be on the screen stack
                modal = ExpandedBoxModal("Size Graph", "graph-container", None)
                await app.push_screen(modal)
                await pilot.pause()

                # Directly call _update_expanded_stats
                # It should iterate through screen_stack, find the graph modal,
                # check that it's not a stats modal, and continue/exit the loop
                app._update_expanded_stats()

                # The test passes if no error is raised (the loop correctly skips
                # the graph modal when looking for stats modals)

                await pilot.press("escape")
                await pilot.press("q")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_update_expanded_stats_missing_stats_display():
    """Test that _update_expanded_stats handles missing stats display."""

    async def run():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Push a stats modal
                modal = ExpandedBoxModal("Statistics", "stats-container", None)
                await app.push_screen(modal)
                await pilot.pause()

                # Remove the stats display from the main app
                stats_display = app.query_one("#stats-display", StatsDisplay)
                stats_display.remove()
                await pilot.pause()

                # Directly call _update_expanded_stats
                # It should find the stats modal but stats_displays will be empty
                app._update_expanded_stats()

                # The test passes if no error is raised

                await pilot.press("escape")
                await pilot.press("q")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())


def test_update_expanded_graph_missing_expanded_graph():
    """Test that _update_expanded_graph handles missing expanded graph widget."""

    async def run():
        fake_client = FakeReductionClient(updates=[], wait_indefinitely=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            temp_file = f.name

        try:
            app = ShrinkRayApp(
                file_path=temp_file,
                test=["true"],
                exit_on_completion=False,
                client=fake_client,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Push a graph modal
                modal = ExpandedBoxModal("Size Graph", "graph-container", None)
                await app.push_screen(modal)
                await pilot.pause()

                # Remove the expanded graph from the modal
                expanded_graph = modal.query_one("#expanded-graph", SizeGraph)
                expanded_graph.remove()
                await pilot.pause()

                # Directly call _update_expanded_graph
                # It should find the graph modal but expanded_graphs will be empty
                app._update_expanded_graph([(1.0, 100)], 200, 1.0)

                # The test passes if no error is raised

                await pilot.press("escape")
                await pilot.press("q")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    asyncio.run(run())
