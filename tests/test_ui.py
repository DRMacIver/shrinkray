"""Tests for UI abstractions."""

from unittest.mock import MagicMock

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
