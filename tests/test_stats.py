"""Test stats reader."""

from pathlib import Path

from janus_core.stats import Stats

DATA_PATH = Path(__file__).parent / "data"


def test_stats():
    """Test readind md stats"""
    data_path = DATA_PATH / "md-stats.dat"

    stat_data = Stats(data_path)

    assert stat_data.rows == 100
    assert stat_data.columns == 18
    assert stat_data.units[0] == ""
    assert stat_data.units[17] == "[K]"
    assert stat_data.labels[0] == "Step"
    assert stat_data.labels[17] == "T*"
