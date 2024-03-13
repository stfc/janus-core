"""Test geomopt commandline interface."""

from pathlib import Path

from ase.io import read
from typer.testing import CliRunner

from janus_core.cli import app

DATA_PATH = Path(__file__).parent / "data"

runner = CliRunner()


def test_geomopt_help():
    """Test calling `janus geomopt --help`."""
    result = runner.invoke(app, ["geomopt", "--help"])
    assert result.exit_code == 0
    # Command is returned as "root"
    assert "Usage: root geomopt [OPTIONS]" in result.stdout


def test_geomopt(tmp_path):
    """Test geomopt calculation."""
    results_path = tmp_path / "NaCl-results.xyz"

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--write-kwargs",
            f"{{'filename': '{str(results_path)}'}}",
            "--max-force",
            "0.2",
        ],
    )
    assert result.exit_code == 0


def test_geomopt_log(tmp_path, caplog):
    """Test log correctly written for geomopt."""
    with caplog.at_level("INFO", logger="janus_core.geom_opt"):
        result = runner.invoke(
            app,
            [
                "geomopt",
                "--struct",
                DATA_PATH / "NaCl.cif",
                "--log",
                f"{tmp_path}/test.log",
            ],
        )
        assert result.exit_code == 0
        assert "Starting geometry optimization" in caplog.text


def test_geomopt_traj(tmp_path):
    """Test log correctly written for geomopt."""
    traj_path = f"{tmp_path}/test.xyz"
    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--traj",
            traj_path,
        ],
    )
    assert result.exit_code == 0
    atoms = read(traj_path)
    assert "forces" in atoms.arrays
