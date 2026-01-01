import os
import warnings

import pytest
from hypothesis import settings

# Import from vendored pytest_textual_snapshot for syrupy 5.0 compatibility
# This registers the snap_compare fixture with the correct .svg extension
from tests import pytest_textual_snapshot as _textual_snapshot


@pytest.fixture(autouse=True)
def shrinkray_temp_directory(tmp_path):
    """Set SHRINKRAY_DIRECTORY to a temp directory for all tests.

    This prevents tests from creating .shrinkray directories in the project root.
    The fixture is autouse=True so it applies to all tests automatically.
    """
    old_value = os.environ.get("SHRINKRAY_DIRECTORY")
    os.environ["SHRINKRAY_DIRECTORY"] = str(tmp_path)
    yield
    if old_value is None:
        os.environ.pop("SHRINKRAY_DIRECTORY", None)
    else:
        os.environ["SHRINKRAY_DIRECTORY"] = old_value


# Re-export the hooks and fixture for pytest to pick up
pytest_addoption = _textual_snapshot.pytest_addoption  # used
pytest_sessionstart = _textual_snapshot.pytest_sessionstart  # used
pytest_terminal_summary = _textual_snapshot.pytest_terminal_summary  # used
snap_compare = _textual_snapshot.snap_compare  # used

settings.register_profile("default", settings(deadline=None))
warnings.filterwarnings("ignore", category=DeprecationWarning)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_sessionfinish(session, exitstatus):
    """Handle session finish for both snapshot report and validation."""
    # Generate the snapshot report
    _textual_snapshot.pytest_sessionfinish_snapshot_report(session, exitstatus)
