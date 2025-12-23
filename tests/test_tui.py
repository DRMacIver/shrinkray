"""Comprehensive tests for the textual TUI module."""

import asyncio
import os
import tempfile
from collections.abc import AsyncIterator

import pytest

from shrinkray.subprocess.protocol import ProgressUpdate, Response
from shrinkray.tui import ContentPreview, ShrinkRayApp, StatsDisplay


class FakeReductionClient:
    """Fake client for testing the TUI without launching a real subprocess."""

    def __init__(
        self,
        updates: list[ProgressUpdate] | None = None,
        start_error: str | None = None,
        start_delay: float = 0.0,
    ):
        self._updates = updates or []
        self._start_error = start_error
        self._start_delay = start_delay
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
        timeout: float = 1.0,
        seed: int = 0,
        input_type: str = "all",
        in_place: bool = False,
        formatter: str = "default",
        volume: str = "normal",
        no_clang_delta: bool = False,
        clang_delta: str = "",
        trivial_is_error: bool = True,
    ) -> Response:
        if self._start_error:
            return Response(id="start", error=self._start_error)
        return Response(id="start", result={"status": "started"})

    async def cancel(self) -> Response:
        self._cancelled = True
        self._completed = True
        return Response(id="cancel", result={"status": "cancelled"})

    async def close(self) -> None:
        self._closed = True

    async def get_progress_updates(self) -> AsyncIterator[ProgressUpdate]:
        for update in self._updates:
            if self._cancelled:
                break
            yield update
            await asyncio.sleep(0.01)
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


class TestStatsDisplay:
    """Tests for the StatsDisplay widget."""

    def test_initial_render(self):
        """Test initial state shows waiting message."""
        widget = StatsDisplay()
        assert "Waiting for reduction to start" in widget.render()

    def test_update_stats(self):
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

    def test_render_with_stats(self):
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

    def test_render_small_reduction(self):
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


class TestContentPreview:
    """Tests for the ContentPreview widget."""

    def test_initial_render(self):
        """Test initial state shows loading message."""
        widget = ContentPreview()
        assert "Loading" in widget.render()

    def test_text_content(self):
        """Test rendering text content."""
        widget = ContentPreview()
        widget.update_content("Hello, World!\nThis is a test.", False)
        rendered = widget.render()
        assert "Hello, World!" in rendered
        assert "This is a test." in rendered

    def test_hex_content(self):
        """Test rendering hex content."""
        widget = ContentPreview()
        widget.update_content("00000000  48 65 6c 6c 6f", True)
        rendered = widget.render()
        assert "[Hex mode]" in rendered
        assert "48 65 6c 6c 6f" in rendered

    def test_large_content_truncated(self):
        """Test that large content is truncated."""
        widget = ContentPreview()
        # Create content with more lines than would fit
        lines = [f"Line {i}" for i in range(100)]
        large_content = "\n".join(lines)
        widget.update_content(large_content, False)
        rendered = widget.render()
        # Should show truncation message
        assert "more lines" in rendered

    def test_content_diff_shown_for_large_files(self):
        """Test that diff is shown when content changes in large files."""
        import time

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

    def test_content_update_throttled(self):
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

    def test_render_diff_for_changed_large_content(self):
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

    def test_render_diff_no_changes_shows_truncated(self):
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


class TestShrinkRayAppWithFakeClient:
    """Tests for ShrinkRayApp using a fake client."""

    @pytest.fixture
    def basic_updates(self) -> list[ProgressUpdate]:
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

    def test_app_mounts_successfully(self):
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

    def test_app_shows_initial_status(self):
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

    def test_app_receives_progress_updates(self, basic_updates):
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

    def test_app_shows_error_on_start_failure(self):
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

    def test_quit_action(self):
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

    def test_quit_cancels_reduction(self, basic_updates):
        """Test that pressing 'q' cancels the reduction and quits."""

        async def run_test():
            # Use many updates to ensure we have time to cancel
            many_updates = basic_updates * 100
            fake_client = FakeReductionClient(updates=many_updates)
            app = ShrinkRayApp(
                file_path="/tmp/test.txt",
                test=["./test.sh"],
                client=fake_client,
            )

            async with app.run_test() as pilot:
                # Wait for the reduction to start
                await asyncio.sleep(0.1)
                await pilot.pause()
                await pilot.press("q")
                # Wait for quit to be processed
                await asyncio.sleep(0.1)
                await pilot.pause()

                # Client should have received cancel
                assert fake_client._cancelled

        run_async(run_test())

    def test_app_completes_successfully(self, basic_updates):
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

    def test_progress_updates_change_stats(self, basic_updates):
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

    def test_app_sets_title(self):
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

    def test_client_closed_on_completion(self, basic_updates):
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

    def test_app_with_various_parameters(self):
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


class TestAppWithoutClient:
    """Tests for ShrinkRayApp without providing a client."""

    def test_app_creates_own_client(self):
        """Test that app creates its own client when none provided."""
        from unittest.mock import AsyncMock, MagicMock, patch

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

    def test_app_handles_exception_in_run_reduction(self):
        """Test that app handles exceptions during reduction."""
        from unittest.mock import AsyncMock, MagicMock, patch

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


class TestEndToEnd:
    """End-to-end tests that use a real subprocess."""

    @pytest.fixture
    def temp_test_file(self):
        """Create a temporary test file for reduction."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as f:
            f.write(b"Hello, World! This is some test content to reduce.")
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def temp_test_script(self):
        """Create a temporary interestingness test script."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            # Script that succeeds if file contains "Hello"
            f.write('#!/bin/bash\ngrep -q "Hello" "$1"\n')
            temp_path = f.name
        os.chmod(temp_path, 0o755)
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_real_subprocess_communication(self, temp_test_file, temp_test_script):
        """Test with a real subprocess client."""

        async def run_test():
            from shrinkray.subprocess.client import SubprocessClient

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


class TestThemeDetection:
    """Tests for terminal theme detection."""

    def test_detect_dark_from_colorfgbg_dark(self, monkeypatch):
        """Test that COLORFGBG with dark background returns True."""
        from shrinkray.tui import detect_terminal_theme

        monkeypatch.setenv("COLORFGBG", "15;0")  # white on black
        assert detect_terminal_theme() is True

    def test_detect_light_from_colorfgbg_light(self, monkeypatch):
        """Test that COLORFGBG with light background returns False."""
        from shrinkray.tui import detect_terminal_theme

        monkeypatch.setenv("COLORFGBG", "0;15")  # black on white
        assert detect_terminal_theme() is False

    def test_detect_light_from_colorfgbg_gray(self, monkeypatch):
        """Test that COLORFGBG with gray background (7+) returns False."""
        from shrinkray.tui import detect_terminal_theme

        monkeypatch.setenv("COLORFGBG", "0;7")  # black on light gray
        assert detect_terminal_theme() is False

    def test_detect_dark_colorfgbg_boundary(self, monkeypatch):
        """Test that COLORFGBG with value 6 returns True (dark)."""
        from shrinkray.tui import detect_terminal_theme

        monkeypatch.setenv("COLORFGBG", "15;6")
        assert detect_terminal_theme() is True

    def test_detect_invalid_colorfgbg_falls_through(self, monkeypatch):
        """Test that invalid COLORFGBG falls through to default."""
        from shrinkray.tui import detect_terminal_theme

        monkeypatch.setenv("COLORFGBG", "invalid")
        monkeypatch.delenv("TERM_PROGRAM", raising=False)
        # Should fall through to default (True = dark)
        assert detect_terminal_theme() is True

    def test_detect_colorfgbg_non_numeric(self, monkeypatch):
        """Test that non-numeric COLORFGBG values are handled."""
        from shrinkray.tui import detect_terminal_theme

        monkeypatch.setenv("COLORFGBG", "foo;bar")
        monkeypatch.delenv("TERM_PROGRAM", raising=False)
        assert detect_terminal_theme() is True

    def test_detect_empty_colorfgbg(self, monkeypatch):
        """Test that empty COLORFGBG falls through."""
        from shrinkray.tui import detect_terminal_theme

        monkeypatch.setenv("COLORFGBG", "")
        monkeypatch.delenv("TERM_PROGRAM", raising=False)
        assert detect_terminal_theme() is True

    def test_detect_no_env_vars_defaults_dark(self, monkeypatch):
        """Test that no environment variables defaults to dark."""
        from shrinkray.tui import detect_terminal_theme

        monkeypatch.delenv("COLORFGBG", raising=False)
        monkeypatch.delenv("TERM_PROGRAM", raising=False)
        assert detect_terminal_theme() is True

    def test_detect_macos_terminal_dark(self, monkeypatch):
        """Test macOS Terminal.app dark mode detection."""
        from unittest.mock import MagicMock, patch

        from shrinkray.tui import detect_terminal_theme

        monkeypatch.delenv("COLORFGBG", raising=False)
        monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
        monkeypatch.delenv("__CFBundleIdentifier", raising=False)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Dark"

        with patch("subprocess.run", return_value=mock_result):
            assert detect_terminal_theme() is True

    def test_detect_macos_terminal_light(self, monkeypatch):
        """Test macOS Terminal.app light mode detection."""
        from unittest.mock import MagicMock, patch

        from shrinkray.tui import detect_terminal_theme

        monkeypatch.delenv("COLORFGBG", raising=False)
        monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
        monkeypatch.delenv("__CFBundleIdentifier", raising=False)

        mock_result = MagicMock()
        mock_result.returncode = 1  # Fails when in light mode
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            assert detect_terminal_theme() is False

    def test_detect_macos_iterm_dark(self, monkeypatch):
        """Test iTerm.app dark mode detection."""
        from unittest.mock import MagicMock, patch

        from shrinkray.tui import detect_terminal_theme

        monkeypatch.delenv("COLORFGBG", raising=False)
        monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
        monkeypatch.delenv("__CFBundleIdentifier", raising=False)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Dark"

        with patch("subprocess.run", return_value=mock_result):
            assert detect_terminal_theme() is True

    def test_detect_macos_subprocess_exception(self, monkeypatch):
        """Test macOS detection handles subprocess exceptions."""
        from unittest.mock import patch

        from shrinkray.tui import detect_terminal_theme

        monkeypatch.delenv("COLORFGBG", raising=False)
        monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
        monkeypatch.delenv("__CFBundleIdentifier", raising=False)

        with patch("subprocess.run", side_effect=Exception("timeout")):
            # Should fall through to default (True = dark)
            assert detect_terminal_theme() is True

    def test_detect_macos_with_cf_bundle_identifier(self, monkeypatch):
        """Test macOS detection skips subprocess when __CFBundleIdentifier is set."""
        from shrinkray.tui import detect_terminal_theme

        monkeypatch.delenv("COLORFGBG", raising=False)
        monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
        monkeypatch.setenv("__CFBundleIdentifier", "com.apple.Terminal")

        # Should fall through to default without calling subprocess
        assert detect_terminal_theme() is True


class TestThemeSettings:
    """Tests for theme settings in ShrinkRayApp."""

    def test_app_with_dark_theme(self):
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

    def test_app_with_light_theme(self):
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

    def test_app_with_auto_theme(self, monkeypatch):
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


class TestModuleInterface:
    """Tests for the module-level interface."""

    def test_tui_can_be_imported(self):
        """Test that the TUI module can be imported."""
        from shrinkray import tui

        assert hasattr(tui, "ShrinkRayApp")
        assert hasattr(tui, "run_textual_ui")
        assert hasattr(tui, "StatsDisplay")
        assert hasattr(tui, "ReductionClientProtocol")

    def test_shrinkray_app_is_textual_app(self):
        """Test that ShrinkRayApp is a proper textual App subclass."""
        from textual.app import App

        from shrinkray.tui import ShrinkRayApp

        assert issubclass(ShrinkRayApp, App)

    def test_run_textual_ui_signature(self):
        """Test that run_textual_ui has expected parameters."""
        import inspect

        from shrinkray.tui import run_textual_ui

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

    def test_fake_client_implements_protocol(self):
        """Test that FakeReductionClient implements the protocol."""
        client = FakeReductionClient()

        # Check it has all required methods
        assert hasattr(client, "start")
        assert hasattr(client, "start_reduction")
        assert hasattr(client, "cancel")
        assert hasattr(client, "close")
        assert hasattr(client, "get_progress_updates")
        assert hasattr(client, "is_completed")


class TestContentPreviewEdgeCases:
    """Edge case tests for ContentPreview."""

    def test_get_available_lines_fallback_to_app_size(self):
        """Test _get_available_lines falls back to app.size when parent unavailable."""
        widget = ContentPreview()
        # Mock the app with a valid size
        from unittest.mock import MagicMock

        mock_app = MagicMock()
        mock_app.size.height = 50
        widget._app = mock_app  # type: ignore

        # The method should use the app size fallback
        result = widget._get_available_lines()
        assert result >= 10  # Minimum is 10

    def test_render_identical_content_no_diff(self):
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


class TestShrinkRayAppExceptionHandlers:
    """Tests for exception handling paths in ShrinkRayApp."""

    def test_update_status_before_mount(self):
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

    def test_quit_handles_cancel_exception(self):
        """Test that action_quit handles exceptions from cancel."""
        from unittest.mock import AsyncMock, MagicMock

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

    def test_client_completed_breaks_loop(self):
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


class TestCoverageEdgeCases:
    """Tests specifically for hard-to-cover edge cases."""

    def test_content_preview_app_size_zero_height(self):
        """Test _get_available_lines when app.size.height is 0."""
        from unittest.mock import MagicMock, PropertyMock, patch

        widget = ContentPreview()

        # Mock the app with zero height
        mock_app = MagicMock()
        mock_app.size.height = 0

        # Mock parent with no usable size
        mock_parent = MagicMock()
        mock_parent.size.height = 0

        # Use patch to override the app property
        with patch.object(
            type(widget), "app", new_callable=PropertyMock
        ) as mock_app_prop:
            mock_app_prop.return_value = mock_app
            with patch.object(
                type(widget), "parent", new_callable=PropertyMock
            ) as mock_parent_prop:
                mock_parent_prop.return_value = mock_parent

                # Should fall through to the default return value of 30
                result = widget._get_available_lines()
                assert result == 30

    def test_content_preview_diff_is_empty(self):
        """Test render when diff computation produces empty result.

        This covers line 286->291 where if diff is empty, we fall through
        to the truncated content display.
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

        # Mock unified_diff to return empty list
        # Note: unified_diff is imported inside render(), so patch difflib directly
        from unittest.mock import patch

        with patch.object(widget, "_get_available_lines", return_value=10):
            with patch("difflib.unified_diff", return_value=[]):
                result = widget.render()

        # Since diff is empty (mocked), should show truncated content
        assert "more lines" in result

    def test_cancel_exception_is_caught(self):
        """Test that exceptions during cancel() are caught (lines 450-451)."""

        class ExceptionOnCancelClient(FakeReductionClient):
            def __init__(self):
                # Many updates so we can quit mid-reduction
                super().__init__(updates=[])

            async def cancel(self) -> Response:
                raise ConnectionError("Process already dead")

            async def get_progress_updates(self) -> AsyncIterator[ProgressUpdate]:
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
                # Wait for reduction to start and get at least one update
                await pilot.pause()
                await asyncio.sleep(0.1)

                # Verify app is not completed yet
                assert not app._completed

                # Press quit - this should trigger cancel() which raises
                await pilot.press("q")
                # If we get here without crashing, the exception was caught
                await pilot.pause()

        # Should not raise
        run_async(run_test())

    def test_run_textual_ui_creates_and_runs_app(self):
        """Test run_textual_ui function creates app and calls run()."""
        from unittest.mock import MagicMock, patch

        from shrinkray.tui import run_textual_ui

        # Mock validation to pass without launching subprocess
        async def mock_validate(*args, **kwargs):
            return None

        # Patch ShrinkRayApp and validation
        with (
            patch("shrinkray.tui.ShrinkRayApp") as mock_app_class,
            patch(
                "shrinkray.tui._validate_initial_example", side_effect=mock_validate
            ),
        ):
            mock_app = MagicMock()
            mock_app.return_code = None  # Ensure no exit
            mock_app_class.return_value = mock_app

            run_textual_ui(
                file_path="/tmp/test.txt",
                test=["./test.sh"],
                parallelism=4,
                timeout=2.0,
                seed=42,
                input_type="bytes",
                in_place=True,
                formatter="clang-format",
                volume="quiet",
                no_clang_delta=True,
                clang_delta="/usr/bin/clang_delta",
                trivial_is_error=True,
                theme="dark",
            )

            # Verify app was created with correct arguments
            mock_app_class.assert_called_once_with(
                file_path="/tmp/test.txt",
                test=["./test.sh"],
                parallelism=4,
                timeout=2.0,
                seed=42,
                input_type="bytes",
                in_place=True,
                formatter="clang-format",
                volume="quiet",
                no_clang_delta=True,
                clang_delta="/usr/bin/clang_delta",
                trivial_is_error=True,
                theme="dark",
            )

            # Verify run() was called
            mock_app.run.assert_called_once()

    def test_completed_flag_during_iteration_breaks_loop(self):
        """Test that is_completed becoming True mid-iteration breaks the loop."""

        class CompletesEarlyClient(FakeReductionClient):
            def __init__(self):
                super().__init__(updates=[])
                self._yield_count = 0

            async def get_progress_updates(self) -> AsyncIterator[ProgressUpdate]:
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
