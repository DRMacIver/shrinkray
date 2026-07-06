"""Regression test: Hypothesis's deadline must be off in every test run.

The default 200ms deadline turns slow machines (e.g. loaded CI runners)
into flaky DeadlineExceeded/Flaky failures. On CI, hypothesis >= 6.116.0
disables it by auto-loading its "ci" profile; locally, conftest.py
disables it by re-registering the active profile, which works on
hypothesis >= 6.117.0. Our floor used to be 6.92.1, which supports
neither, so the lowest-direct CI jobs ran with deadlines on.
"""

from hypothesis import settings


def test_hypothesis_deadline_is_disabled():
    assert settings.default is not None
    assert settings.default.deadline is None
