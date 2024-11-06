"""Test stats reader."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import approx

from janus_core.helpers.stats import Stats

DATA_PATH = Path(__file__).parent / "data"


class TestStats:
    """Tests for the stats type."""

    data = Stats(DATA_PATH / "md-stats.dat")

    @pytest.mark.parametrize(
        "attr,expected",
        (
            ("rows", 100),
            ("columns", 18),
        ),
        ids=(
            "get rows",
            "get_cols",
        ),
    )
    def test_props(self, attr, expected):
        """Test props are being set correctly."""
        assert getattr(self.data, attr) == expected

    @pytest.mark.parametrize(
        "attr,ind,expected",
        (
            ("data", (99, 17), approx(300.0)),
            ("units", 0, ""),
            ("units", 17, "K"),
            ("labels", 0, "# Step"),
            ("labels", 17, "Target T"),
        ),
        ids=(
            "data value",
            "Step units",
            "Target T units",
            "Step label",
            "Target T label",
        ),
    )
    def test_data_index(self, attr, ind, expected):
        """Test data indexing working correctly."""
        assert getattr(self.data, attr)[ind] == expected

    @pytest.mark.parametrize(
        "ind,expectedcol",
        (
            ("target t", 17),
            (17, 17),
            (slice(3, 7, 2), (3, 5)),
            ((1, 3), (1, 3)),
            (("target t", "step"), (17, 0)),
            (("target t", 0), (17, 0)),
        ),
        ids=(
            "str",
            "int",
            "slice",
            "tuple[int]",
            "tuple[str]",
            "tuple[mixed]",
        ),
    )
    def test_getitem(self, ind, expectedcol):
        """Test getitem indexing working correctly."""
        assert (self.data[ind] == self.data.data[:, expectedcol]).all()

    def test_repr(self, capsys):
        """Test repr working correctly."""
        print(self.data)
        std_out_err = capsys.readouterr()
        assert std_out_err.err == ""
        assert "index label units" in std_out_err.out
        assert (
            f"contains {self.data.columns} timeseries, "
            f"each with {self.data.rows} elements" in std_out_err.out
        )
