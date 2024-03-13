"""Test singlepoint commandline interface."""

from pathlib import Path

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
