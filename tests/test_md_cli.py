"""Test md commandline interface."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from janus_core.cli import app

DATA_PATH = Path(__file__).parent / "data"

runner = CliRunner()


def test_md_help():
    """Test calling `janus md --help`."""
    result = runner.invoke(app, ["md", "--help"])
    assert result.exit_code == 0
    # Command is returned as "root"
    assert "Usage: root md [OPTIONS]" in result.stdout


test_data = [
    ("nvt"),
    ("nve"),
    ("npt"),
    ("nvt-nh"),
    ("nph"),
]


@pytest.mark.parametrize("ensemble", test_data)
def test_md(ensemble, tmp_path):
    """Test singlepoint calculation."""
    file_prefix = tmp_path / f"{ensemble}-T300"

    result = runner.invoke(
        app,
        [
            "md",
            ensemble,
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--temp",
            300,
            "--file-prefix",
            file_prefix,
        ],
    )

    # atoms = read_atoms(results_path)
    assert result.exit_code == 0
