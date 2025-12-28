"""Textual-based TUI for Shrink Ray."""

import math
import os
import time
import traceback
from collections.abc import AsyncGenerator
from contextlib import aclosing
from datetime import timedelta
from typing import Literal, Protocol

import humanize
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import DataTable, Footer, Header, Label, Static
from textual_plotext import PlotextPlot

from shrinkray.subprocess.client import SubprocessClient
from shrinkray.subprocess.protocol import (
    PassStatsData,
    ProgressUpdate,
    Response,
)


ThemeMode = Literal["auto", "dark", "light"]

# Custom themes with true white/black backgrounds
SHRINKRAY_LIGHT_THEME = Theme(
    name="shrinkray-light",
    primary="#0066cc",
    secondary="#6c757d",
    accent="#007acc",
    background="#ffffff",  # Pure white
    surface="#ffffff",
    panel="#f8f9fa",
    dark=False,
)

SHRINKRAY_DARK_THEME = Theme(
    name="shrinkray-dark",
    primary="#4da6ff",
    secondary="#adb5bd",
    accent="#4dc3ff",
    background="#000000",  # Pure black
    surface="#000000",
    panel="#1a1a1a",
    dark=True,
)


def detect_terminal_theme() -> bool:
    """Detect if terminal is in dark mode. Returns True for dark, False for light."""
    # Check COLORFGBG environment variable (format: "fg;bg" where higher bg = light)
    colorfgbg = os.environ.get("COLORFGBG", "")
    if colorfgbg:
        try:
            parts = colorfgbg.split(";")
            if len(parts) >= 2:
                bg = int(parts[-1])
                # Background values 0-6 are typically dark, 7+ are light
                # Common: 0=black, 15=white, 7=light gray
                return bg < 7
        except (ValueError, IndexError):
            pass

    # Check for macOS Terminal.app / iTerm2 light mode indicators
    term_program = os.environ.get("TERM_PROGRAM", "")
    if term_program in ("Apple_Terminal", "iTerm.app"):
        # Check if system is in light mode via defaults (macOS)
        # This is a heuristic - AppleInterfaceStyle is absent in light mode
        apple_interface = os.environ.get("__CFBundleIdentifier", "")
        if not apple_interface:
            try:
                import subprocess

                result = subprocess.run(
                    ["defaults", "read", "-g", "AppleInterfaceStyle"],
                    capture_output=True,
                    text=True,
                    timeout=1,
                )
                # If this succeeds and returns "Dark", we're in dark mode
                # If it fails (exit code 1), we're in light mode
                return result.returncode == 0 and "Dark" in result.stdout
            except Exception:
                pass

    # Default to dark mode (textual's default)
    return True


class ReductionClientProtocol(Protocol):
    """Protocol for reduction client - allows mocking for tests."""

    async def start(self) -> None: ...

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
    ) -> Response: ...
    async def cancel(self) -> Response: ...
    async def disable_pass(self, pass_name: str) -> Response: ...
    async def enable_pass(self, pass_name: str) -> Response: ...
    async def skip_current_pass(self) -> Response: ...
    async def close(self) -> None: ...

    @property
    def error_message(self) -> str | None: ...
    def get_progress_updates(self) -> AsyncGenerator[ProgressUpdate, None]: ...
    @property
    def is_completed(self) -> bool: ...


class StatsDisplay(Static):
    """Widget to display reduction statistics."""

    # Use prefixed names to avoid conflicts with textual's built-in properties
    current_status = reactive("Starting...")
    current_size = reactive(0)
    original_size = reactive(0)
    call_count = reactive(0)
    reduction_count = reactive(0)
    interesting_calls = reactive(0)
    wasted_calls = reactive(0)
    runtime = reactive(0.0)
    parallel_workers = reactive(0)
    average_parallelism = reactive(0.0)
    effective_parallelism = reactive(0.0)
    time_since_last_reduction = reactive(0.0)

    def update_stats(self, update: ProgressUpdate) -> None:
        self.current_status = update.status
        self.current_size = update.size
        self.original_size = update.original_size
        self.call_count = update.calls
        self.reduction_count = update.reductions
        self.interesting_calls = update.interesting_calls
        self.wasted_calls = update.wasted_calls
        self.runtime = update.runtime
        self.parallel_workers = update.parallel_workers
        self.average_parallelism = update.average_parallelism
        self.effective_parallelism = update.effective_parallelism
        self.time_since_last_reduction = update.time_since_last_reduction
        self.refresh(layout=True)

    def render(self) -> str:
        if self.original_size == 0:
            return "Waiting for reduction to start..."

        # Calculate stats
        reduction_pct = (1.0 - self.current_size / self.original_size) * 100
        deleted = self.original_size - self.current_size

        # Build stats display
        lines = []

        # Size and reduction info
        if self.reduction_count > 0 and self.runtime > 0:
            reduction_rate = deleted / self.runtime
            lines.append(
                f"Current test case size: {humanize.naturalsize(self.current_size)} "
                f"({reduction_pct:.2f}% reduction, {humanize.naturalsize(reduction_rate)} / second)"
            )
        else:
            lines.append(
                f"Current test case size: {humanize.naturalsize(self.current_size)}"
            )

        # Runtime
        if self.runtime > 0:
            runtime_delta = timedelta(seconds=self.runtime)
            lines.append(f"Total runtime: {humanize.precisedelta(runtime_delta)}")

        # Call statistics
        if self.call_count > 0:
            calls_per_sec = self.call_count / self.runtime if self.runtime > 0 else 0
            interesting_pct = (self.interesting_calls / self.call_count) * 100
            wasted_pct = (self.wasted_calls / self.call_count) * 100
            lines.append(
                f"Calls to interestingness test: {self.call_count} "
                f"({calls_per_sec:.2f} calls / second, "
                f"{interesting_pct:.2f}% interesting, "
                f"{wasted_pct:.2f}% wasted)"
            )
        else:
            lines.append("Not yet called interestingness test")

        # Time since last reduction
        if self.reduction_count > 0 and self.runtime > 0:
            reductions_per_sec = self.reduction_count / self.runtime
            lines.append(
                f"Time since last reduction: {self.time_since_last_reduction:.2f}s "
                f"({reductions_per_sec:.2f} reductions / second)"
            )
        else:
            lines.append("No reductions yet")

        lines.append("")
        lines.append(f"Reducer status: {self.current_status}")

        # Parallelism stats - always show
        lines.append(
            f"Current parallel workers: {self.parallel_workers} "
            f"(Average {self.average_parallelism:.2f}) "
            f"(effective parallelism: {self.effective_parallelism:.2f})"
        )

        return "\n".join(lines)


def _format_time_label(seconds: float) -> str:
    """Format a time value for axis labels."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes}m"
    else:
        hours = int(seconds / 3600)
        return f"{hours}h"


def _get_time_axis_bounds(current_time: float) -> tuple[float, list[float], list[str]]:
    """Get stable x-axis bounds and tick values.

    Returns (max_time, ticks, labels) where max_time is the stable right boundary,
    ticks are the tick positions, and labels are the formatted labels.

    The axis only rescales when current_time exceeds the current boundary.
    """
    if current_time <= 0:
        ticks = [0.0, 10.0, 20.0, 30.0]
        labels = [_format_time_label(t) for t in ticks]
        return (30.0, ticks, labels)

    # For the first 10 minutes, expand one minute at a time with 1-minute ticks
    if current_time < 600:
        # Round up to next minute
        minutes = int(current_time / 60) + 1
        max_time = float(minutes * 60)
        interval = 60.0
    else:
        # After 10 minutes, use larger boundaries
        # (boundary, tick_interval) - axis extends to boundary, ticks at interval
        boundaries = [
            (1800, 300),  # 30m with 5m ticks
            (3600, 600),  # 1h with 10m ticks
            (7200, 1200),  # 2h with 20m ticks
            (14400, 1800),  # 4h with 30m ticks
            (28800, 3600),  # 8h with 1h ticks
        ]

        # Find the first boundary that exceeds current_time
        max_time = 1800.0
        interval = 300.0
        for boundary, tick_interval in boundaries:
            if current_time < boundary:
                max_time = float(boundary)
                interval = float(tick_interval)
                break
        else:
            # Beyond 8h: extend in 4h increments with 1h ticks
            hours = int(current_time / 14400) + 1
            max_time = float(hours * 14400)
            interval = 3600.0

    # Generate ticks from 0 to max_time
    ticks = []
    t = 0.0
    while t <= max_time:
        ticks.append(t)
        t += interval

    labels = [_format_time_label(t) for t in ticks]
    return (max_time, ticks, labels)


def _get_percentage_axis_bounds(
    min_pct: float, max_pct: float
) -> tuple[float, list[float], list[str]]:
    """Get stable y-axis bounds for percentage values on log scale.

    Returns (min_pct_bound, ticks, labels) where min_pct_bound is the stable lower
    boundary, ticks are the log10 tick positions, and labels are percentage strings.

    The axis only rescales when min_pct gets close to the current lower boundary.
    """
    # Standard percentage boundaries (log scale friendly)
    boundaries = [100, 50, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

    # Find the appropriate lower bound - use the first boundary below min_pct * 0.5
    # This gives us some room before we need to rescale
    lower_bound = 0.01
    for b in boundaries:
        if b < min_pct * 0.5:
            lower_bound = b
            break

    # Find which percentage values to show as ticks (between lower_bound and 100%)
    # Since boundaries always includes 100 and lower_bound <= 100, this is never empty
    tick_pcts = [p for p in boundaries if p >= lower_bound and p <= 100]

    # Convert to log scale
    ticks = [math.log10(max(0.01, p)) for p in tick_pcts]
    labels = [f"{p}%" if p >= 1 else f"{p:.1f}%" for p in tick_pcts]

    return (lower_bound, ticks, labels)


class SizeGraph(PlotextPlot):
    """Widget to display test case size over time on a log scale."""

    _size_history: list[tuple[float, int]]
    _original_size: int
    _current_runtime: float

    def __init__(
        self,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._size_history = []
        self._original_size = 0
        self._current_runtime = 0.0

    def update_graph(
        self,
        new_entries: list[tuple[float, int]],
        original_size: int,
        current_runtime: float,
    ) -> None:
        """Update the graph with new data."""
        if new_entries:
            self._size_history.extend(new_entries)
        if original_size > 0:
            self._original_size = original_size
        self._current_runtime = current_runtime
        self._setup_plot()
        self.refresh()

    def on_mount(self) -> None:
        """Set up the plot on mount."""
        self._setup_plot()

    def on_resize(self) -> None:
        """Redraw when resized."""
        self._setup_plot()

    def _setup_plot(self) -> None:
        """Configure and draw the plot."""
        plt = self.plt
        plt.clear_figure()
        plt.theme("dark")

        if len(self._size_history) < 2 or self._original_size == 0:
            plt.xlabel("Time")
            plt.ylabel("% of original")
            return

        times = [t for t, _ in self._size_history]
        sizes = [s for _, s in self._size_history]

        # Calculate percentages of original size
        percentages = [(s / self._original_size) * 100 for s in sizes]

        # Use log scale for y-axis (percentages)
        log_percentages = [math.log10(max(0.01, p)) for p in percentages]

        plt.plot(times, log_percentages, marker="fhd")

        # Get stable x-axis bounds
        max_time, x_ticks, x_labels = _get_time_axis_bounds(self._current_runtime)
        plt.xticks(x_ticks, x_labels)
        plt.xlim(0, max_time)

        # Get stable y-axis bounds
        min_pct = min(percentages)
        lower_bound, y_ticks, y_labels = _get_percentage_axis_bounds(min_pct, 100)
        plt.yticks(y_ticks, y_labels)
        plt.ylim(math.log10(max(0.01, lower_bound)), math.log10(100))

        plt.xlabel("Time")
        plt.ylabel("% of original")

        # Build to apply the plot
        _ = plt.build()


class ContentPreview(Static):
    """Widget to display the current test case content preview."""

    preview_content = reactive("")
    hex_mode = reactive(False)
    _last_displayed_content: str = ""
    _last_display_time: float = 0.0
    _pending_content: str = ""
    _pending_hex_mode: bool = False

    def update_content(self, content: str, hex_mode: bool) -> None:
        # Store the pending content
        self._pending_content = content
        self._pending_hex_mode = hex_mode

        # Throttle updates to once per second
        now = time.time()
        if now - self._last_display_time < 1.0:
            return

        # Update the displayed content
        self._last_display_time = now

        # Track last displayed content for diffs
        if self.preview_content and self.preview_content != content:
            self._last_displayed_content = str(self.preview_content)

        self.preview_content = content
        self.hex_mode = hex_mode
        self.refresh(layout=True)

    def _get_available_lines(self) -> int:
        """Get the number of lines available for display based on container size."""
        try:
            # Try to get the parent container's size (the VerticalScroll viewport)
            parent = self.parent
            if parent and hasattr(parent, "size"):
                parent_size = parent.size  # type: ignore[union-attr]
                if parent_size.height > 0:
                    return max(10, parent_size.height - 2)
            # Fall back to app screen size
            if self.app and self.app.size.height > 0:
                # Estimate available space (screen minus header, footer, stats, etc.)
                return max(10, self.app.size.height - 15)
        except Exception:
            pass
        # Fallback based on common terminal height
        return 30

    def render(self) -> str:
        if not self.preview_content:
            return "Loading..."

        available_lines = self._get_available_lines()

        if self.hex_mode:
            return f"[Hex mode]\n{self.preview_content}"

        lines = self.preview_content.split("\n")

        # For small files that fit, show full content
        if len(lines) <= available_lines:
            return self.preview_content

        # For larger files, show diff if we have previous displayed content
        if (
            self._last_displayed_content
            and self._last_displayed_content != self.preview_content
        ):
            from difflib import unified_diff

            prev_lines = self._last_displayed_content.split("\n")
            curr_lines = self.preview_content.split("\n")
            diff = list(unified_diff(prev_lines, curr_lines, lineterm=""))
            if diff:
                # Show as much diff as fits
                return "\n".join(diff[:available_lines])

        # No diff available, show truncated content
        return (
            "\n".join(lines[:available_lines])
            + f"\n\n... ({len(lines) - available_lines} more lines)"
        )


class OutputPreview(Static):
    """Widget to display test output preview."""

    output_content = reactive("")
    active_test_id: reactive[int | None] = reactive(None)
    last_return_code: reactive[int | None] = reactive(None)
    _last_update_time: float = 0.0
    _last_seen_test_id: int | None = None  # Track last test ID for "completed" message
    # Pending updates that haven't been applied yet (due to throttling)
    _pending_content: str = ""
    _pending_test_id: int | None = None
    _pending_return_code: int | None = None
    # Track if we've ever seen any output (once true, never show "No test output yet...")
    _has_seen_output: bool = False

    def update_output(
        self, content: str, test_id: int | None, return_code: int | None = None
    ) -> None:
        # Only update pending content if there's actual content to show
        # This prevents switching to empty output when we have previous output
        if content:
            self._pending_content = content
            self._has_seen_output = True
        self._pending_test_id = test_id
        if return_code is not None:
            self._pending_return_code = return_code

        # Throttle display updates to every 200ms
        now = time.time()
        if now - self._last_update_time < 0.2:
            return

        self._last_update_time = now
        # Only update output_content if we have new content
        if self._pending_content:
            self.output_content = self._pending_content
        # Track the last test ID we've seen (for showing in "completed" message)
        if self._pending_test_id is not None:
            self._last_seen_test_id = self._pending_test_id
        self.active_test_id = self._pending_test_id
        self.last_return_code = self._pending_return_code
        self.refresh(layout=True)

    def _get_available_lines(self) -> int:
        """Get the number of lines available for display based on container size."""
        try:
            parent = self.parent
            if parent and hasattr(parent, "size"):
                parent_size = parent.size  # type: ignore[union-attr]
                if parent_size.height > 0:
                    return max(10, parent_size.height - 3)
            if self.app and self.app.size.height > 0:
                return max(10, self.app.size.height - 15)
        except Exception:
            pass
        return 30

    def render(self) -> str:
        # Header line
        if self.active_test_id is not None:
            header = f"[green]Test #{self.active_test_id} running...[/green]"
        elif self._last_seen_test_id is not None:
            if self.last_return_code is not None:
                header = f"[dim]Test #{self._last_seen_test_id} exited with code {self.last_return_code}[/dim]"
            else:
                header = f"[dim]Test #{self._last_seen_test_id} completed[/dim]"
        elif self._has_seen_output or self.output_content:
            # Have seen output before - show without header
            header = ""
        else:
            header = "[dim]No test output yet...[/dim]"

        if not self.output_content:
            return header

        available_lines = self._get_available_lines()
        lines = self.output_content.split("\n")

        # Build prefix (header + newline, or empty if no header)
        prefix = f"{header}\n" if header else ""

        # Show tail of output (most recent lines)
        if len(lines) <= available_lines:
            return f"{prefix}{self.output_content}"

        # Truncate from the beginning
        truncated_lines = lines[-(available_lines):]
        skipped = len(lines) - available_lines
        return f"{prefix}... ({skipped} earlier lines)\n" + "\n".join(truncated_lines)


class HelpScreen(ModalScreen[None]):
    """Modal screen showing keyboard shortcuts help."""

    CSS = """
    HelpScreen {
        align: center middle;
    }

    HelpScreen > Vertical {
        width: 60;
        height: auto;
        max-height: 80%;
        background: $panel;
        border: thick $primary;
        padding: 1 2;
    }

    HelpScreen #help-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    HelpScreen .help-section {
        margin-bottom: 1;
    }

    HelpScreen .help-key {
        color: $accent;
    }
    """

    BINDINGS = [
        ("escape,q,h", "dismiss", "Close"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Keyboard Shortcuts", id="help-title")
            yield Static("")
            yield Static("[bold]Main Screen[/bold]", classes="help-section")
            yield Static("  [green]h[/green]     Show this help")
            yield Static("  [green]p[/green]     Open pass statistics")
            yield Static("  [green]c[/green]     Skip current pass")
            yield Static("  [green]q[/green]     Quit application")
            yield Static("")
            yield Static("[bold]Pass Statistics Screen[/bold]", classes="help-section")
            yield Static("  [green]↑/↓[/green]   Navigate passes")
            yield Static("  [green]space[/green] Toggle pass enabled/disabled")
            yield Static("  [green]c[/green]     Skip current pass")
            yield Static("  [green]q[/green]     Close modal")
            yield Static("")
            yield Static("[dim]Press any key to close[/dim]")


class ExpandedBoxModal(ModalScreen[None]):
    """Modal screen showing an expanded view of a content box."""

    CSS = """
    ExpandedBoxModal {
        align: center middle;
    }

    ExpandedBoxModal > Vertical {
        width: 95%;
        height: 90%;
        background: $panel;
        border: thick $primary;
        padding: 0 1 1 1;
    }

    ExpandedBoxModal #expanded-title {
        text-align: center;
        text-style: bold;
        height: auto;
        width: 100%;
        border-bottom: solid $primary;
        padding: 0;
        margin-bottom: 1;
    }

    ExpandedBoxModal VerticalScroll {
        width: 100%;
        height: 1fr;
    }

    ExpandedBoxModal #expanded-content {
        width: 100%;
    }

    ExpandedBoxModal #expanded-graph {
        width: 100%;
        height: 1fr;
    }
    """

    BINDINGS = [
        ("escape,enter,q", "dismiss", "Close"),
    ]

    def __init__(
        self, title: str, content_widget_id: str, file_path: str | None = None
    ) -> None:
        super().__init__()
        self._title = title
        self._content_widget_id = content_widget_id
        self._file_path = file_path

    def _read_file_with_retry(self, file_path: str, max_retries: int = 3) -> str:
        """Read file content with retry for transient errors during reduction.

        The file may be briefly unavailable while the reducer is writing a new
        version. Retry a few times with a short delay to handle this.
        """
        last_error: OSError | None = None
        for attempt in range(max_retries):
            try:
                with open(file_path, "rb") as f:
                    raw_content = f.read()
                # Try to decode as text, fall back to hex if binary
                try:
                    return raw_content.decode("utf-8")
                except UnicodeDecodeError:
                    return "[Binary content - hex display]\n\n" + raw_content.hex()
            except OSError as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(0.05)  # 50ms delay between retries

        # All retries failed - raise the last error to be caught by caller
        if last_error is not None:
            raise last_error
        return "Could not read file"

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._title, id="expanded-title")
            if self._content_widget_id == "graph-container":
                # For graph, create a new SizeGraph widget
                yield SizeGraph(id="expanded-graph")
            else:
                # For other content, use a scrollable static
                with VerticalScroll():
                    yield Static("", id="expanded-content")

    def on_mount(self) -> None:
        """Populate content from the source widget."""
        app = self.app

        if self._content_widget_id == "graph-container":
            # Copy data from the main graph to the expanded graph
            try:
                main_graph = app.query_one("#size-graph", SizeGraph)
                expanded_graph = self.query_one("#expanded-graph", SizeGraph)
                # Copy the history data
                expanded_graph._size_history = main_graph._size_history.copy()
                expanded_graph._original_size = main_graph._original_size
                expanded_graph._current_runtime = main_graph._current_runtime
                expanded_graph._setup_plot()
            except Exception:
                pass
            return

        # For non-graph content, populate the static
        content = ""

        if self._content_widget_id == "stats-container":
            try:
                stats_display = app.query_one("#stats-display", StatsDisplay)
                content = stats_display.render()
            except Exception:
                content = "Statistics not available"
        elif self._content_widget_id == "content-container":
            # Read the full current test case file content
            try:
                if self._file_path:
                    content = self._read_file_with_retry(self._file_path)
                else:
                    content_preview = app.query_one("#content-preview", ContentPreview)
                    content = content_preview.preview_content
            except Exception:
                content = "Content preview not available"
        elif self._content_widget_id == "output-container":
            try:
                output_preview = app.query_one("#output-preview", OutputPreview)
                # Use pending values (most recent) rather than throttled values
                raw_content = output_preview._pending_content or output_preview.output_content
                test_id = output_preview._last_seen_test_id
                if output_preview._pending_test_id is not None:
                    test_id = output_preview._pending_test_id
                return_code = output_preview._pending_return_code
                if return_code is None:
                    return_code = output_preview.last_return_code
                active_test_id = output_preview._pending_test_id if output_preview._pending_test_id is not None else output_preview.active_test_id
                has_seen_output = output_preview._has_seen_output

                # Build header
                if active_test_id is not None:
                    header = f"[green]Test #{active_test_id} running...[/green]\n\n"
                elif test_id is not None:
                    if return_code is not None:
                        header = f"[dim]Test #{test_id} exited with code {return_code}[/dim]\n\n"
                    else:
                        header = f"[dim]Test #{test_id} completed[/dim]\n\n"
                else:
                    header = ""

                if raw_content:
                    content = header + raw_content
                elif has_seen_output or test_id is not None:
                    # We've seen output before - show header only (no "No test output" message)
                    content = header.rstrip("\n") if header else ""
                else:
                    content = "[dim]No test output yet...[/dim]"
            except Exception:
                content = "Output not available"

        try:
            expanded_content = self.query_one("#expanded-content", Static)
            expanded_content.update(content)
        except Exception:
            pass


class PassStatsScreen(ModalScreen[None]):
    """Modal screen showing pass statistics in a table."""

    CSS = """
    PassStatsScreen {
        align: center middle;
    }

    PassStatsScreen > Vertical {
        width: 90%;
        height: 85%;
        background: $panel;
        border: thick $primary;
    }

    PassStatsScreen DataTable {
        height: 1fr;
    }

    PassStatsScreen #stats-footer {
        dock: bottom;
        height: auto;
        padding: 1;
        background: $panel;
        text-align: center;
    }
    """

    BINDINGS = [
        ("escape,q,p", "dismiss", "Close"),
        ("space", "toggle_disable", "Toggle Enable"),
        ("c", "skip_current", "Skip Pass"),
        ("h", "show_help", "Help"),
    ]

    pass_stats: reactive[list[PassStatsData]] = reactive(list)
    current_pass_name: reactive[str] = reactive("")
    disabled_passes: reactive[set[str]] = reactive(set)

    def __init__(self, app: "ShrinkRayApp") -> None:
        super().__init__()
        self._app = app
        self.pass_stats = app._latest_pass_stats.copy()
        self.current_pass_name = app._current_pass_name
        self.disabled_passes = set(app._disabled_passes)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(
                "Pass Statistics - [space] toggle, [c] skip, [h] help, [q] close",
                id="stats-header",
            )
            yield DataTable(id="pass-stats-table")
            yield Static(
                f"Showing {len(self.pass_stats)} passes in run order",
                id="stats-footer",
            )

    def on_mount(self) -> None:
        table = self.query_one(DataTable)

        # Select entire rows, not individual cells
        table.cursor_type = "row"

        table.add_columns(
            "Enabled",
            "Pass Name",
            "Runs",
            "Bytes Deleted",
            "Tests",
            "Reductions",
            "Success %",
        )

        self._update_table_data()

        # Set up periodic refresh (every 500ms)
        self.set_interval(0.5, self._refresh_data)

    def _update_table_data(self) -> None:
        """Update the table with current pass stats."""
        table = self.query_one(DataTable)

        # Save cursor position and scroll position before clearing
        saved_cursor = table.cursor_coordinate
        saved_scroll_y = table.scroll_y

        table.clear()

        if not self.pass_stats:
            table.add_row("-", "No pass data yet", "-", "-", "-", "-", "-")
        else:
            for ps in self.pass_stats:
                is_current = ps.pass_name == self.current_pass_name
                is_disabled = ps.pass_name in self.disabled_passes
                bytes_str = humanize.naturalsize(ps.bytes_deleted, binary=True)

                # Checkbox for enabled/disabled
                if is_disabled:
                    checkbox = Text("[ ]", style="dim")
                else:
                    checkbox = Text("[✓]", style="green")

                # Determine styling: bold for current, dim for disabled
                if is_disabled:
                    style = "dim strike"
                elif is_current:
                    style = "bold"
                else:
                    style = ""

                # Apply styling
                if style:
                    name = Text(ps.pass_name, style=style)
                    runs = Text(str(ps.run_count), style=style)
                    bytes_del = Text(bytes_str, style=style)
                    tests = Text(f"{ps.test_evaluations:,}", style=style)
                    reductions = Text(str(ps.successful_reductions), style=style)
                    success = Text(f"{ps.success_rate:.1f}%", style=style)
                else:
                    name = ps.pass_name
                    runs = str(ps.run_count)
                    bytes_del = bytes_str
                    tests = f"{ps.test_evaluations:,}"
                    reductions = str(ps.successful_reductions)
                    success = f"{ps.success_rate:.1f}%"

                table.add_row(
                    checkbox, name, runs, bytes_del, tests, reductions, success
                )

        # Restore cursor and scroll position after rebuilding
        # Only restore if the saved position is still valid
        row_count = table.row_count
        if row_count > 0 and saved_cursor.row < row_count:
            table.cursor_coordinate = saved_cursor
        table.scroll_y = saved_scroll_y

    def _refresh_data(self) -> None:
        """Refresh data from the app.

        Note: We don't update disabled_passes from the worker because
        the local state is the source of truth. This avoids flicker when
        the user toggles a pass but the worker hasn't confirmed yet.
        """
        new_stats = self._app._latest_pass_stats.copy()
        new_current = self._app._current_pass_name
        if new_stats != self.pass_stats or new_current != self.current_pass_name:
            self.pass_stats = new_stats
            self.current_pass_name = new_current
            self._update_table_data()
            # Update footer with disabled count
            disabled_count = len(self.disabled_passes)
            if disabled_count > 0:
                footer_text = (
                    f"Showing {len(self.pass_stats)} passes ({disabled_count} disabled)"
                )
            else:
                footer_text = f"Showing {len(self.pass_stats)} passes in run order"
            footer = self.query_one("#stats-footer", Static)
            footer.update(footer_text)

    def _get_selected_pass_name(self) -> str | None:
        """Get the pass name from the currently selected row."""
        table = self.query_one(DataTable)
        if table.row_count == 0:
            return None
        cursor_row = table.cursor_coordinate.row
        if cursor_row >= len(self.pass_stats):
            return None
        return self.pass_stats[cursor_row].pass_name

    def action_toggle_disable(self) -> None:
        """Toggle the disabled state of the selected pass."""
        pass_name = self._get_selected_pass_name()
        if pass_name is None:
            return

        if pass_name in self.disabled_passes:
            # Enable the pass - update UI immediately, send command in background
            # Create new set to trigger reactive update
            self.disabled_passes = self.disabled_passes - {pass_name}
            self._update_table_data()
            self._app.run_worker(self._send_enable_pass(pass_name))
        else:
            # Disable the pass - update UI immediately, send command in background
            # Create new set to trigger reactive update
            self.disabled_passes = self.disabled_passes | {pass_name}
            self._update_table_data()
            self._app.run_worker(self._send_disable_pass(pass_name))

    async def _send_disable_pass(self, pass_name: str) -> None:
        """Send disable command to the subprocess (fire and forget)."""
        if self._app._client is not None:
            await self._app._client.disable_pass(pass_name)

    async def _send_enable_pass(self, pass_name: str) -> None:
        """Send enable command to the subprocess (fire and forget)."""
        if self._app._client is not None:
            await self._app._client.enable_pass(pass_name)

    def action_skip_current(self) -> None:
        """Skip the currently running pass."""
        self._app.run_worker(self._skip_pass())

    async def _skip_pass(self) -> None:
        """Skip the current pass via the client."""
        if self._app._client is not None:
            await self._app._client.skip_current_pass()

    def action_show_help(self) -> None:
        """Show the help screen."""
        self._app.push_screen(HelpScreen())


class ShrinkRayApp(App[None]):
    """Textual app for Shrink Ray."""

    CSS = """
    #main-container {
        height: 100%;
    }

    #status-label {
        text-style: bold;
        margin: 0 1;
    }

    #stats-area {
        height: 1fr;
    }

    #stats-container {
        border: solid $primary;
        margin: 0;
        padding: 1;
        width: 1fr;
        height: 100%;
    }

    #stats-container:focus {
        border: thick $primary;
    }

    #graph-container {
        border: solid $primary;
        margin: 0;
        padding: 1;
        width: 1fr;
        height: 100%;
    }

    #graph-container:focus {
        border: thick $primary;
    }

    #size-graph {
        width: 100%;
        height: 100%;
    }

    #content-area {
        height: 1fr;
    }

    #content-container {
        border: solid $primary;
        margin: 0;
        padding: 1;
        width: 1fr;
        height: 100%;
    }

    #content-container:focus {
        border: thick $primary;
    }

    #output-container {
        border: solid $primary;
        margin: 0;
        padding: 1;
        width: 1fr;
        height: 100%;
    }

    #output-container:focus {
        border: thick $primary;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("p", "show_pass_stats", "Pass Stats"),
        ("c", "skip_current_pass", "Skip Pass"),
        ("h", "show_help", "Help"),
        ("up", "focus_up", "Focus Up"),
        ("down", "focus_down", "Focus Down"),
        ("left", "focus_left", "Focus Left"),
        ("right", "focus_right", "Focus Right"),
        ("enter", "expand_box", "Expand"),
    ]

    ENABLE_COMMAND_PALETTE = False

    def __init__(
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
        exit_on_completion: bool = True,
        client: ReductionClientProtocol | None = None,
        theme: ThemeMode = "auto",
    ) -> None:
        super().__init__()
        self._file_path = file_path
        self._test = test
        self._parallelism = parallelism
        self._timeout = timeout
        self._seed = seed
        self._input_type = input_type
        self._in_place = in_place
        self._formatter = formatter
        self._volume = volume
        self._no_clang_delta = no_clang_delta
        self._clang_delta = clang_delta
        self._trivial_is_error = trivial_is_error
        self._exit_on_completion = exit_on_completion
        self._client: ReductionClientProtocol | None = client
        self._owns_client = client is None
        self._completed = False
        self._theme = theme
        self._latest_pass_stats: list[PassStatsData] = []
        self._current_pass_name: str = ""
        self._disabled_passes: list[str] = []

    # Box IDs in navigation order: [top-left, top-right, bottom-left, bottom-right]
    _BOX_IDS = ["stats-container", "graph-container", "content-container", "output-container"]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="main-container"):
            yield Label(
                "Shrink Ray - [h] help, [p] passes, [c] skip pass, [q] quit",
                id="status-label",
                markup=False,
            )
            with Horizontal(id="stats-area"):
                with VerticalScroll(id="stats-container") as stats_scroll:
                    stats_scroll.border_title = "Statistics"
                    stats_scroll.can_focus = True
                    yield StatsDisplay(id="stats-display")
                with Vertical(id="graph-container") as graph_container:
                    graph_container.border_title = "Size Over Time"
                    graph_container.can_focus = True
                    yield SizeGraph(id="size-graph")
            with Horizontal(id="content-area"):
                with VerticalScroll(id="content-container") as content_scroll:
                    content_scroll.border_title = "Recent Reductions"
                    content_scroll.can_focus = True
                    yield ContentPreview(id="content-preview")
                with VerticalScroll(id="output-container") as output_scroll:
                    output_scroll.border_title = "Test Output"
                    output_scroll.can_focus = True
                    yield OutputPreview(id="output-preview")
        yield Footer()

    async def on_mount(self) -> None:
        # Register and apply custom themes
        self.register_theme(SHRINKRAY_LIGHT_THEME)
        self.register_theme(SHRINKRAY_DARK_THEME)

        if self._theme == "dark":
            self.theme = "shrinkray-dark"
        elif self._theme == "light":
            self.theme = "shrinkray-light"
        else:  # auto
            self.theme = (
                "shrinkray-dark" if detect_terminal_theme() else "shrinkray-light"
            )

        self.title = "Shrink Ray"
        self.sub_title = self._file_path

        # Set initial focus to first box
        self.query_one("#stats-container").focus()

        self.run_reduction()

    def _get_focused_box_index(self) -> int:
        """Get the index of the currently focused box, or 0 if none."""
        for i, box_id in enumerate(self._BOX_IDS):
            try:
                if self.query_one(f"#{box_id}").has_focus:
                    return i
            except Exception:
                pass
        return 0

    def _focus_box(self, index: int) -> None:
        """Focus the box at the given index (with wrapping)."""
        index = index % len(self._BOX_IDS)
        box_id = self._BOX_IDS[index]
        self.query_one(f"#{box_id}").focus()

    def action_focus_up(self) -> None:
        """Move focus to the box above."""
        current = self._get_focused_box_index()
        # Grid is 2x2: top row is 0,1; bottom row is 2,3
        # Moving up: 2->0, 3->1, 0->2, 1->3 (wraps)
        if current >= 2:
            self._focus_box(current - 2)
        else:
            self._focus_box(current + 2)

    def action_focus_down(self) -> None:
        """Move focus to the box below."""
        current = self._get_focused_box_index()
        # Moving down: 0->2, 1->3, 2->0, 3->1 (wraps)
        if current < 2:
            self._focus_box(current + 2)
        else:
            self._focus_box(current - 2)

    def action_focus_left(self) -> None:
        """Move focus to the box on the left."""
        current = self._get_focused_box_index()
        # Moving left within row: 0->1, 1->0, 2->3, 3->2 (wraps within row)
        if current % 2 == 0:
            self._focus_box(current + 1)
        else:
            self._focus_box(current - 1)

    def action_focus_right(self) -> None:
        """Move focus to the box on the right."""
        current = self._get_focused_box_index()
        # Moving right within row: 0->1, 1->0, 2->3, 3->2 (wraps within row)
        if current % 2 == 0:
            self._focus_box(current + 1)
        else:
            self._focus_box(current - 1)

    def action_expand_box(self) -> None:
        """Expand the currently focused box to a modal."""
        current = self._get_focused_box_index()
        box_id = self._BOX_IDS[current]

        # Get the title from the container's border_title
        titles = {
            "stats-container": "Statistics",
            "graph-container": "Size Over Time",
            "content-container": "Current Test Case",
            "output-container": "Test Output",
        }
        title = titles.get(box_id, "Details")

        # Pass file_path for content-container to enable full file reading
        file_path = self._file_path if box_id == "content-container" else None
        self.push_screen(ExpandedBoxModal(title, box_id, file_path=file_path))

    @work(exclusive=True)
    async def run_reduction(self) -> None:
        """Start the reduction subprocess and monitor progress."""
        try:
            if self._client is None:
                # No client provided - start one and begin reduction
                debug_mode = self._volume == "debug"
                self._client = SubprocessClient(debug_mode=debug_mode)
                self._owns_client = True

                await self._client.start()

                # Start the reduction - validation was already done by main()
                response = await self._client.start_reduction(
                    file_path=self._file_path,
                    test=self._test,
                    parallelism=self._parallelism,
                    timeout=self._timeout,
                    seed=self._seed,
                    input_type=self._input_type,
                    in_place=self._in_place,
                    formatter=self._formatter,
                    volume=self._volume,
                    no_clang_delta=self._no_clang_delta,
                    clang_delta=self._clang_delta,
                    trivial_is_error=self._trivial_is_error,
                    skip_validation=True,
                )

                if response.error:
                    # Exit immediately on startup error
                    self.exit(return_code=1, message=f"Error: {response.error}")
                    return

            # Monitor progress (client is already started and reduction is running)
            stats_display = self.query_one("#stats-display", StatsDisplay)
            content_preview = self.query_one("#content-preview", ContentPreview)
            output_preview = self.query_one("#output-preview", OutputPreview)
            size_graph = self.query_one("#size-graph", SizeGraph)

            async with aclosing(self._client.get_progress_updates()) as updates:
                async for update in updates:
                    stats_display.update_stats(update)
                    content_preview.update_content(
                        update.content_preview, update.hex_mode
                    )
                    output_preview.update_output(
                        update.test_output_preview,
                        update.active_test_id,
                        update.last_test_return_code,
                    )
                    size_graph.update_graph(
                        update.new_size_history,
                        update.original_size,
                        update.runtime,
                    )
                    # Also update expanded modals if they exist
                    self._update_expanded_graph(
                        update.new_size_history,
                        update.original_size,
                        update.runtime,
                    )
                    self._update_expanded_stats()
                    self._latest_pass_stats = update.pass_stats
                    self._current_pass_name = update.current_pass_name
                    self._disabled_passes = update.disabled_passes

                    # Check if all passes are disabled
                    self._check_all_passes_disabled()

                    if self._client.is_completed:
                        break

            self._completed = True

            # Check if there was an error from the worker
            if self._client.error_message:
                # Exit immediately on error, printing the error message
                self.exit(
                    return_code=1,
                    message=f"Error: {self._client.error_message}",
                )
                return
            elif self._exit_on_completion:
                self.exit()
            else:
                self.update_status("Reduction completed! Press 'q' to exit.")

        except Exception as e:
            traceback.print_exc()
            self.exit(return_code=1, message=f"Error: {e}")
        finally:
            if self._owns_client and self._client:
                await self._client.close()

    def _check_all_passes_disabled(self) -> None:
        """Check if all passes are disabled and show a message if so."""
        if self._latest_pass_stats and self._disabled_passes:
            all_pass_names = {ps.pass_name for ps in self._latest_pass_stats}
            if all_pass_names and all_pass_names <= set(self._disabled_passes):
                self.update_status(
                    "Reduction paused (all passes disabled) - [p] to re-enable passes"
                )

    def update_status(self, message: str) -> None:
        """Update the status label."""
        try:
            self.query_one("#status-label", Label).update(message)
        except Exception:
            pass  # Widget not yet mounted

    def _update_expanded_graph(
        self,
        new_entries: list[tuple[float, int]],
        original_size: int,
        current_runtime: float,
    ) -> None:
        """Update the expanded graph if it exists in a modal screen."""
        # Check if there's an ExpandedBoxModal for the graph on the screen stack
        for screen in self.screen_stack:
            if isinstance(screen, ExpandedBoxModal):
                if screen._content_widget_id == "graph-container":
                    try:
                        expanded_graph = screen.query_one("#expanded-graph", SizeGraph)
                        expanded_graph.update_graph(
                            new_entries, original_size, current_runtime
                        )
                    except Exception:
                        pass
                    break

    def _update_expanded_stats(self) -> None:
        """Update the expanded stats if it exists in a modal screen."""
        for screen in self.screen_stack:
            if isinstance(screen, ExpandedBoxModal):
                if screen._content_widget_id == "stats-container":
                    try:
                        stats_display = self.query_one("#stats-display", StatsDisplay)
                        expanded_content = screen.query_one("#expanded-content", Static)
                        expanded_content.update(stats_display.render())
                    except Exception:
                        pass
                    break

    async def action_quit(self) -> None:
        """Quit the application with graceful cancellation."""
        if self._client and not self._completed:
            try:
                await self._client.cancel()
            except Exception:
                pass  # Process may have already exited
        self.exit()

    def action_show_pass_stats(self) -> None:
        """Show the pass statistics modal."""
        self.push_screen(PassStatsScreen(self))

    def action_show_help(self) -> None:
        """Show the help modal."""
        self.push_screen(HelpScreen())

    def action_skip_current_pass(self) -> None:
        """Skip the currently running pass."""
        if self._client and not self._completed:
            self.run_worker(self._skip_pass())

    async def _skip_pass(self) -> None:
        """Skip the current pass via the client."""
        if self._client is not None:
            await self._client.skip_current_pass()

    @property
    def is_completed(self) -> bool:
        """Check if reduction is completed."""
        return self._completed


def run_textual_ui(
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
    exit_on_completion: bool = True,
    theme: ThemeMode = "auto",
) -> None:
    """Run the textual TUI.

    Note: Validation must be done before calling this function.
    The caller (main()) is responsible for running run_validation() first.
    """
    import sys

    # Start the TUI app - validation has already been done by main()
    app = ShrinkRayApp(
        file_path=file_path,
        test=test,
        parallelism=parallelism,
        timeout=timeout,
        seed=seed,
        input_type=input_type,
        in_place=in_place,
        formatter=formatter,
        volume=volume,
        no_clang_delta=no_clang_delta,
        clang_delta=clang_delta,
        trivial_is_error=trivial_is_error,
        exit_on_completion=exit_on_completion,
        theme=theme,
    )
    app.run()
    if app.return_code:
        sys.exit(app.return_code)
