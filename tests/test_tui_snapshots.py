"""Snapshot tests for the textual TUI."""

import asyncio
from pathlib import Path

import pytest

from shrinkray.subprocess.protocol import PassStatsData, ProgressUpdate, Response
from shrinkray.tui import ShrinkRayApp, StatsDisplay


# Directory where snapshots are stored
SNAPSHOTS_DIR = Path(__file__).parent / "__snapshots__" / "test_tui_snapshots"


def pytest_sessionfinish(session, exitstatus):
    """Verify all snapshots contain expected content after tests complete."""
    if not SNAPSHOTS_DIR.exists():
        return

    # Using vendored pytest_textual_snapshot with syrupy 5.0 compatibility
    for snapshot_file in SNAPSHOTS_DIR.glob("*.svg"):
        content = snapshot_file.read_text()
        # Each snapshot should contain "Reducer" which appears in StatsDisplay
        # Note: SVG uses &#160; for non-breaking spaces, so we just check for "Reducer"
        if "Reducer" not in content:
            raise AssertionError(
                f"Snapshot {snapshot_file.name} does not contain expected text 'Reducer'. "
                "The snapshot may have been captured before the app fully rendered."
            )


class FakeReductionClientForSnapshots:
    """Fake client that provides controlled updates for snapshot testing."""

    def __init__(self, updates: list[ProgressUpdate]):
        self._updates = updates
        self._update_index = 0
        self._cancelled = False
        self._updates_consumed = False

    async def start(self) -> None:
        pass

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
    ) -> Response:
        return Response(id="start", result={"status": "started"})

    async def cancel(self) -> Response:
        self._cancelled = True
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
        pass

    async def get_progress_updates(self):
        for update in self._updates:
            if self._cancelled:
                break
            yield update
        self._updates_consumed = True
        # Keep the app running by waiting indefinitely (until cancelled)
        while not self._cancelled:
            await asyncio.sleep(1)

    @property
    def is_completed(self) -> bool:
        # Only complete when cancelled, not when updates are consumed
        return self._cancelled

    @property
    def error_message(self) -> str | None:
        return None


def make_app_with_updates(updates: list[ProgressUpdate]) -> ShrinkRayApp:
    """Create a ShrinkRayApp with fake updates for testing."""
    client = FakeReductionClientForSnapshots(updates)
    return ShrinkRayApp(
        file_path="/tmp/test.txt",
        test=["./test.sh"],
        client=client,
        theme="dark",  # Explicit theme for deterministic snapshots
    )


@pytest.fixture
def initial_state_update() -> ProgressUpdate:
    """A progress update showing initial state."""
    return ProgressUpdate(
        status="Starting reduction",
        size=10000,
        original_size=10000,
        calls=0,
        reductions=0,
        interesting_calls=0,
        wasted_calls=0,
        runtime=0.0,
        parallel_workers=4,
        average_parallelism=4.0,
        effective_parallelism=4.0,
        time_since_last_reduction=0.0,
        content_preview="Line 1\nLine 2\nLine 3\nLine 4\nLine 5",
        hex_mode=False,
    )


@pytest.fixture
def mid_reduction_update() -> ProgressUpdate:
    """A progress update showing mid-reduction state."""
    return ProgressUpdate(
        status="Running DeleteChunks",
        size=5000,
        original_size=10000,
        calls=150,
        reductions=25,
        interesting_calls=50,
        wasted_calls=10,
        runtime=30.5,
        parallel_workers=4,
        average_parallelism=3.8,
        effective_parallelism=3.5,
        time_since_last_reduction=2.5,
        content_preview="Line 1\nLine 3\nLine 5",
        hex_mode=False,
    )


@pytest.fixture
def hex_mode_update() -> ProgressUpdate:
    """A progress update showing hex mode."""
    return ProgressUpdate(
        status="Running ByteReduction",
        size=256,
        original_size=512,
        calls=50,
        reductions=10,
        interesting_calls=20,
        wasted_calls=5,
        runtime=10.0,
        parallel_workers=2,
        average_parallelism=2.0,
        effective_parallelism=1.8,
        time_since_last_reduction=1.0,
        content_preview="00000000  48 65 6c 6c 6f 20 57 6f  72 6c 64 21              Hello World!",
        hex_mode=True,
    )


@pytest.fixture
def large_file_update() -> ProgressUpdate:
    """A progress update with large file content for diff testing."""
    # Create content with many lines
    lines = [f"Line {i}: some content here" for i in range(100)]
    return ProgressUpdate(
        status="Running pass 5",
        size=50000,
        original_size=100000,
        calls=500,
        reductions=100,
        interesting_calls=200,
        wasted_calls=50,
        runtime=120.0,
        parallel_workers=8,
        average_parallelism=7.5,
        effective_parallelism=6.8,
        time_since_last_reduction=0.5,
        content_preview="\n".join(lines),
        hex_mode=False,
    )


# === TUI snapshot tests ===


async def wait_for_render(pilot):
    """Wait for the app to fully render with actual content.

    Checks that the StatsDisplay has received and rendered the update
    by verifying original_size is non-zero (all test fixtures have non-zero size).
    """
    max_attempts = 50  # Generous limit to avoid flakiness
    for _ in range(max_attempts):
        await pilot.pause()
        stats = pilot.app.query_one("#stats-display", StatsDisplay)
        if stats.original_size > 0:
            # Content has been rendered, give one more pause for layout
            await pilot.pause()
            return

    # Get final state for error message
    final_stats = pilot.app.query_one("#stats-display", StatsDisplay)
    raise AssertionError(
        "App did not render within expected time. "
        f"StatsDisplay.original_size is still {final_stats.original_size}"
    )


def test_initial_state(snap_compare, initial_state_update):
    """Snapshot test for initial app state."""
    app = make_app_with_updates([initial_state_update])
    assert snap_compare(app, terminal_size=(120, 40), run_before=wait_for_render)


def test_mid_reduction(snap_compare, mid_reduction_update):
    """Snapshot test for mid-reduction state."""
    app = make_app_with_updates([mid_reduction_update])
    assert snap_compare(app, terminal_size=(120, 40), run_before=wait_for_render)


def test_hex_mode(snap_compare, hex_mode_update):
    """Snapshot test for hex mode display."""
    app = make_app_with_updates([hex_mode_update])
    assert snap_compare(app, terminal_size=(120, 40), run_before=wait_for_render)


def test_large_file(snap_compare, large_file_update):
    """Snapshot test for large file with potential diff."""
    app = make_app_with_updates([large_file_update])
    assert snap_compare(app, terminal_size=(120, 40), run_before=wait_for_render)


def test_small_terminal(snap_compare, mid_reduction_update):
    """Snapshot test with smaller terminal size."""
    app = make_app_with_updates([mid_reduction_update])
    assert snap_compare(app, terminal_size=(80, 24), run_before=wait_for_render)


def test_wide_terminal(snap_compare, mid_reduction_update):
    """Snapshot test with wider terminal."""
    app = make_app_with_updates([mid_reduction_update])
    assert snap_compare(app, terminal_size=(160, 50), run_before=wait_for_render)


@pytest.fixture
def update_with_pass_stats() -> ProgressUpdate:
    """A progress update with pass statistics."""
    return ProgressUpdate(
        status="Running hollow",
        size=5000,
        original_size=10000,
        calls=150,
        reductions=25,
        interesting_calls=50,
        wasted_calls=10,
        runtime=30.5,
        parallel_workers=4,
        average_parallelism=3.8,
        effective_parallelism=3.5,
        time_since_last_reduction=2.5,
        content_preview="Line 1\nLine 3\nLine 5",
        hex_mode=False,
        pass_stats=[
            PassStatsData(
                pass_name="hollow",
                bytes_deleted=2500,
                run_count=5,
                test_evaluations=100,
                successful_reductions=3,
                success_rate=3.0,
            ),
            PassStatsData(
                pass_name="delete_lines",
                bytes_deleted=1500,
                run_count=3,
                test_evaluations=80,
                successful_reductions=2,
                success_rate=2.5,
            ),
            PassStatsData(
                pass_name="lift_braces",
                bytes_deleted=1000,
                run_count=2,
                test_evaluations=50,
                successful_reductions=1,
                success_rate=2.0,
            ),
        ],
        current_pass_name="hollow",
        disabled_passes=[],
    )


async def test_pass_stats_modal(snap_compare, update_with_pass_stats):
    """Snapshot test for the pass statistics modal."""

    async def interact(pilot):
        # Wait a moment for the app to start
        await pilot.pause()
        # Press 'p' to open the pass stats modal
        await pilot.press("p")
        # Wait for the modal to render
        await pilot.pause()

    app = make_app_with_updates([update_with_pass_stats])
    assert snap_compare(app, terminal_size=(120, 40), run_before=interact)


# === Test output display snapshots ===


@pytest.fixture
def update_with_active_test() -> ProgressUpdate:
    """A progress update with an active test showing output."""
    return ProgressUpdate(
        status="Running DeleteChunks",
        size=5000,
        original_size=10000,
        calls=150,
        reductions=25,
        interesting_calls=50,
        wasted_calls=10,
        runtime=30.5,
        parallel_workers=4,
        average_parallelism=3.8,
        effective_parallelism=3.5,
        time_since_last_reduction=2.5,
        content_preview="Line 1\nLine 3\nLine 5",
        hex_mode=False,
        test_output_preview=(
            "Running test case validation...\n"
            "Checking syntax: OK\n"
            "Checking semantics: OK\n"
            "Executing test script...\n"
            "  + ./test.sh test_case.txt\n"
            "  Assertion failed at line 42\n"
            "  Expected: 0\n"
            "  Got: 1\n"
        ),
        active_test_id=42,
    )


@pytest.fixture
def update_with_completed_test() -> ProgressUpdate:
    """A progress update with a completed test (no active test)."""
    return ProgressUpdate(
        status="Running DeleteChunks",
        size=5000,
        original_size=10000,
        calls=151,
        reductions=26,
        interesting_calls=51,
        wasted_calls=10,
        runtime=31.0,
        parallel_workers=4,
        average_parallelism=3.8,
        effective_parallelism=3.5,
        time_since_last_reduction=0.5,
        content_preview="Line 1\nLine 3\nLine 5",
        hex_mode=False,
        test_output_preview=(
            "Running test case validation...\n"
            "Checking syntax: OK\n"
            "Checking semantics: OK\n"
            "Executing test script...\n"
            "  + ./test.sh test_case.txt\n"
            "Test passed (exit code 0)\n"
        ),
        active_test_id=None,  # Test completed
    )


@pytest.fixture
def update_with_long_output() -> ProgressUpdate:
    """A progress update with long test output that needs truncation."""
    # Generate output longer than the display area
    output_lines = [f"[{i:04d}] Processing item {i}..." for i in range(50)]
    return ProgressUpdate(
        status="Running ByteReduction",
        size=2000,
        original_size=10000,
        calls=500,
        reductions=80,
        interesting_calls=150,
        wasted_calls=30,
        runtime=60.0,
        parallel_workers=4,
        average_parallelism=3.5,
        effective_parallelism=3.2,
        time_since_last_reduction=1.0,
        content_preview="Reduced content here",
        hex_mode=False,
        test_output_preview="\n".join(output_lines),
        active_test_id=123,
    )


def test_active_test_output(snap_compare, update_with_active_test):
    """Snapshot test showing active test with output."""
    app = make_app_with_updates([update_with_active_test])
    assert snap_compare(app, terminal_size=(120, 40), run_before=wait_for_render)


def test_completed_test_output(snap_compare, update_with_completed_test):
    """Snapshot test showing completed test output."""
    app = make_app_with_updates([update_with_completed_test])
    assert snap_compare(app, terminal_size=(120, 40), run_before=wait_for_render)


def test_long_test_output(snap_compare, update_with_long_output):
    """Snapshot test showing truncated long output."""
    app = make_app_with_updates([update_with_long_output])
    assert snap_compare(app, terminal_size=(120, 40), run_before=wait_for_render)
