"""Textual-based TUI for Shrink Ray."""

import os
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Literal, Protocol

import humanize
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import DataTable, Footer, Header, Label, Static

from shrinkray.subprocess.client import SubprocessClient
from shrinkray.subprocess.protocol import PassStatsData, ProgressUpdate, Response


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
        timeout: float = 1.0,
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
    def get_progress_updates(self) -> AsyncIterator[ProgressUpdate]: ...
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


class ContentPreview(Static):
    """Widget to display the current test case content preview."""

    preview_content = reactive("")
    hex_mode = reactive(False)
    _last_displayed_content: str = ""
    _last_display_time: float = 0.0
    _pending_content: str = ""
    _pending_hex_mode: bool = False

    def update_content(self, content: str, hex_mode: bool) -> None:
        import time

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
        ("escape,q", "dismiss", "Close"),
        ("space", "toggle_disable", "Toggle Enable"),
        ("c", "skip_current", "Skip Pass"),
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
                "Pass Statistics - [space] toggle, [c] skip pass, [q] close",
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
                    runs = Text(str(ps.call_count), style=style)
                    bytes_del = Text(bytes_str, style=style)
                    tests = Text(f"{ps.test_evaluations:,}", style=style)
                    reductions = Text(str(ps.successful_reductions), style=style)
                    success = Text(f"{ps.success_rate:.1f}%", style=style)
                else:
                    name = ps.pass_name
                    runs = str(ps.call_count)
                    bytes_del = bytes_str
                    tests = f"{ps.test_evaluations:,}"
                    reductions = str(ps.successful_reductions)
                    success = f"{ps.success_rate:.1f}%"

                table.add_row(checkbox, name, runs, bytes_del, tests, reductions, success)

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
                footer_text = f"Showing {len(self.pass_stats)} passes ({disabled_count} disabled)"
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


class ShrinkRayApp(App[None]):
    """Textual app for Shrink Ray."""

    CSS = """
    #main-container {
        height: 100%;
    }

    #stats-container {
        height: auto;
        border: solid green;
        padding: 1;
        margin: 1;
    }

    #status-label {
        text-style: bold;
        margin: 0 1;
    }

    #content-container {
        border: solid blue;
        margin: 1;
        height: 1fr;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("p", "show_pass_stats", "Pass Stats"),
        ("c", "skip_current_pass", "Skip Pass"),
        ("h", "show_help", "Help"),
    ]

    ENABLE_COMMAND_PALETTE = False

    def __init__(
        self,
        file_path: str,
        test: list[str],
        parallelism: int | None = None,
        timeout: float = 1.0,
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

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="main-container"):
            yield Label(
                "Shrink Ray - [h] help, [p] passes, [c] skip pass, [q] quit",
                id="status-label",
            )
            with Vertical(id="stats-container"):
                yield StatsDisplay(id="stats-display")
            with VerticalScroll(id="content-container"):
                yield ContentPreview(id="content-preview")
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
        self.run_reduction()

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

                # Start the reduction
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
                )

                if response.error:
                    # Exit immediately on startup error
                    self.exit(return_code=1, message=f"Error: {response.error}")
                    return

            # Monitor progress (client is already started and reduction is running)
            stats_display = self.query_one("#stats-display", StatsDisplay)
            content_preview = self.query_one("#content-preview", ContentPreview)

            async for update in self._client.get_progress_updates():
                stats_display.update_stats(update)
                content_preview.update_content(update.content_preview, update.hex_mode)
                self._latest_pass_stats = update.pass_stats
                self._current_pass_name = update.current_pass_name
                self._disabled_passes = update.disabled_passes

                # Check if all passes are disabled
                if self._latest_pass_stats and self._disabled_passes:
                    all_pass_names = {ps.pass_name for ps in self._latest_pass_stats}
                    if all_pass_names and all_pass_names <= set(self._disabled_passes):
                        self.update_status(
                            "Reduction paused (all passes disabled) - "
                            "[p] to re-enable passes"
                        )

                if self._client.is_completed:
                    break

            self._completed = True

            # Check if there was an error from the worker
            if self._client.error_message:
                # Exit immediately on error, printing the error message
                self.exit(return_code=1, message=f"Error: {self._client.error_message}")
                return
            elif self._exit_on_completion:
                self.exit()
            else:
                self.update_status("Reduction completed! Press 'q' to exit.")

        except Exception as e:
            self.exit(return_code=1, message=f"Error: {e}")
        finally:
            if self._owns_client and self._client:
                await self._client.close()

    def update_status(self, message: str) -> None:
        """Update the status label."""
        try:
            self.query_one("#status-label", Label).update(message)
        except Exception:
            pass  # Widget not yet mounted

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


async def _validate_initial_example(
    file_path: str,
    test: list[str],
    parallelism: int | None,
    timeout: float,
    seed: int,
    input_type: str,
    in_place: bool,
    formatter: str,
    volume: str,
    no_clang_delta: bool,
    clang_delta: str,
    trivial_is_error: bool,
) -> str | None:
    """Validate initial example before showing TUI.

    Returns error_message if validation failed, None if it passed.
    """
    debug_mode = volume == "debug"
    client = SubprocessClient(debug_mode=debug_mode)
    try:
        await client.start()

        response = await client.start_reduction(
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
        )

        if response.error:
            return response.error

        # Validation passed - cancel this reduction since TUI will start fresh
        await client.cancel()
        return None
    finally:
        await client.close()


def run_textual_ui(
    file_path: str,
    test: list[str],
    parallelism: int | None = None,
    timeout: float = 1.0,
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
    """Run the textual TUI."""
    import asyncio
    import sys

    print("Validating initial example...", flush=True)

    # Validate initial example before showing TUI
    async def validate():
        return await _validate_initial_example(
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
        )

    try:
        error = asyncio.run(validate())
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    # Validation passed - now show the TUI which will start a fresh client
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
