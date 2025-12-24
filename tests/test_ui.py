"""Tests for UI abstractions."""

import io
import sys
from unittest.mock import MagicMock

import trio

from shrinkray.ui import BasicUI


def test_shrinkray_ui_reducer_property():
    """Test ShrinkRayUI.reducer returns state's reducer."""
    mock_state = MagicMock()
    mock_reducer = MagicMock()
    mock_state.reducer = mock_reducer

    # Create a concrete subclass for testing
    ui = BasicUI(state=mock_state)

    assert ui.reducer is mock_reducer


def test_shrinkray_ui_problem_property():
    """Test ShrinkRayUI.problem returns reducer's target."""
    mock_state = MagicMock()
    mock_reducer = MagicMock()
    mock_target = MagicMock()
    mock_state.reducer = mock_reducer
    mock_reducer.target = mock_target

    ui = BasicUI(state=mock_state)

    assert ui.problem is mock_target


def test_basic_ui_can_be_created():
    """Test BasicUI can be instantiated."""
    mock_state = MagicMock()
    ui = BasicUI(state=mock_state)
    assert ui.state is mock_state


async def test_basic_ui_run_prints_reduction_message():
    """Test BasicUI.run() prints a message when reduction occurs.

    Exercises the reduction message printing in BasicUI.run().
    """
    # Create mock state with initial size > current size
    mock_state = MagicMock()
    mock_problem = MagicMock()

    # Initial is 100 bytes, current is 50 bytes = 50 byte reduction
    mock_state.initial = b"x" * 100
    mock_problem.current_test_case = b"x" * 50

    # size function just returns len
    mock_problem.size = len
    mock_state.problem = mock_problem

    ui = BasicUI(state=mock_state)

    # Capture stdout
    captured = io.StringIO()
    old_stdout = sys.stdout

    try:
        sys.stdout = captured

        # Run the UI briefly - it should print once then sleep
        # We use move_on_after to exit after the first iteration
        async with trio.open_nursery() as nursery:

            async def run_ui():
                await ui.run(nursery)

            nursery.start_soon(run_ui)
            # Wait a tiny bit for the first print to happen
            await trio.sleep(0.05)
            # Cancel after first iteration
            nursery.cancel_scope.cancel()
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()
    # Should contain the reduction message
    assert "Reduced" in output
    assert "deleted" in output
    assert "50" in output or "Bytes" in output  # Size should be mentioned
