"""Test stats reader."""

from pathlib import Path

from pytest import approx

from janus_core.stats import Stats

DATA_PATH = Path(__file__).parent / "data"


def test_stats(capsys):
    """Test reading md stats."""
    data_path = DATA_PATH / "md-stats.dat"

    stat_data = Stats(data_path)

    assert stat_data.rows == 100
    assert stat_data.columns == 18
    assert stat_data.data[99, 17] == approx(300.0)
    assert stat_data.units[0] == ""
    assert stat_data.units[17] == "K"
    assert stat_data.labels[0] == "# Step"
    assert stat_data.labels[17] == "Target T"

    print(stat_data)
    std_out_err = capsys.readouterr()
    assert std_out_err.err == ""
    assert "index label units" in std_out_err.out
    assert (
        f"contains {stat_data.columns} timeseries, each with {stat_data.rows} elements"
        in std_out_err.out
    )
