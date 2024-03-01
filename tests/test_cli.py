"""Test commandline interface."""

from pathlib import Path

from typer.testing import CliRunner

from janus_core.cli import app

DATA_PATH = Path(__file__).parent / "data"

runner = CliRunner()


def test_janus_help():
    """Test calling `janus --help`."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # Command is returned as "root"
    assert "Usage: root [OPTIONS] COMMAND [ARGS]..." in result.stdout


def test_singlepoint_help():
    """Test calling `janus singlepoint --help`."""
    result = runner.invoke(app, ["singlepoint", "--help"])
    assert result.exit_code == 0
    # Command is returned as "root"
    assert "Usage: root singlepoint [OPTIONS]" in result.stdout


def test_singlepoint():
    """Test singlepoint calculation."""
    result = runner.invoke(app, ["singlepoint", "--system", DATA_PATH / "NaCl.cif"])
    assert result.exit_code == 0
    assert "{'energy': -" in result.stdout
    assert "'forces': array" in result.stdout
    assert "'stress': array" in result.stdout


def test_singlepoint_properties():
    """Test properties for singlepoint calculation."""
    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--system",
            DATA_PATH / "NaCl.cif",
            "--property",
            "energy",
            "--property",
            "forces",
        ],
    )
    assert result.exit_code == 0
    assert "{'energy': -" in result.stdout
    assert "'forces': array" in result.stdout
    assert "'stress': array" not in result.stdout


def test_singlepoint_read_kwargs():
    """Test setting read_kwargs for singlepoint calculation."""
    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--system",
            DATA_PATH / "benzene-traj.xyz",
            "--read-kwargs",
            "{'index': ':'}",
            "--property",
            "energy",
        ],
    )
    assert result.exit_code == 0
    assert "'energy': [" in result.stdout


def test_singlepoint_calc_kwargs():
    """Test setting calc_kwargs for singlepoint calculation."""
    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--system",
            DATA_PATH / "NaCl.cif",
            "--calc-kwargs",
            "{'default_dtype': 'float32'}",
            "--property",
            "energy",
        ],
    )
    assert result.exit_code == 0
    assert "Using float32 for MACECalculator" in result.stdout
