"""Test md commandline interface."""

from pathlib import Path

from ase import Atoms
from ase.io import read
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
    """Test all MD simulations are able to run."""
    file_prefix = tmp_path / f"{ensemble}-T300"
    traj_path = tmp_path / f"{ensemble}-T300-traj.xyz"

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
            "--steps",
            2,
            "--traj-every",
            1,
        ],
    )

    assert result.exit_code == 0

    # Check at least one image has been saved in trajectory
    atoms = read(traj_path)
    assert isinstance(atoms, Atoms)


def test_md_log(tmp_path, caplog):
    """Test log correctly written for MD."""
    file_prefix = tmp_path / "nvt-T300"

    with caplog.at_level("INFO", logger="janus_core.md"):
        result = runner.invoke(
            app,
            [
                "md",
                "nvt",
                "--struct",
                DATA_PATH / "NaCl.cif",
                "--temp",
                300,
                "--file-prefix",
                file_prefix,
                "--steps",
                2,
            ],
        )
        assert result.exit_code == 0
        assert " Starting molecular dynamics simulation" in caplog.text
