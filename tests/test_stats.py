"""Test stats reader."""

from pathlib import Path

import pytest

from janus_core.stats import Stats

DATA_PATH = Path(__file__).parent / "data"


def test_stats():
    """Test readind md stats"""
    data_path = DATA_PATH / "md-stats.dat"

    s = Stats(data_path)

    assert s.rows == 100
    assert s.columns == 18
    assert s.units[0] == ""
    assert s.units[17] == "[K]"
    assert s.units[0] == "Step"
    assert s.units[17] == "T*"
