"""Shared pytest configuration for the olmoearth2 test suite."""


def pytest_configure(config):
    """Register custom markers used by the suite."""
    config.addinivalue_line(
        "markers",
        "slow: heavy tests (full base model / checkpoint weight loading). "
        "Run with `pytest` (default) and excluded via `-m 'not slow'`.",
    )
