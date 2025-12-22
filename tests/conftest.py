import pytest
from hypothesis import settings
import warnings

settings.register_profile("default", settings(deadline=None))
warnings.filterwarnings("ignore", category=DeprecationWarning)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
