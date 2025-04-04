"""Configure pytest.

Based on https://docs.pytest.org/en/latest/example/simple.html.
"""

from __future__ import annotations

import sys

import pytest


def pytest_addoption(parser):
    """Add flag to run tests for extra MLIPs."""
    parser.addoption(
        "--run-extra-mlips",
        action="store_true",
        default=False,
        help="Test additional MLIPs",
    )


def pytest_configure(config):
    """Configure pytest to include marker for extra MLIPs."""
    config.addinivalue_line(
        "markers", "extra_mlips: mark test as containing extra MLIPs"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests if marker applied to unit tests."""
    if config.getoption("--run-extra-mlips"):
        # --run-extra-mlips given in cli: do not skip tests
        return
    skip_extra_mlips = pytest.mark.skip(reason="need --run-extra-mlips option to run")
    for item in items:
        if "extra_mlips" in item.keywords:
            item.add_marker(skip_extra_mlips)


@pytest.fixture(autouse=True)
def capture_wrap():
    """Block closure of stderr and stdout."""
    # Prevent `ValueError: I/O operation on closed file` during testing
    # See discussion in https://github.com/stfc/janus-core/pull/426
    sys.stderr.close = lambda *args, **kwargs: None
    sys.stdout.close = lambda *args, **kwargs: None
    yield
