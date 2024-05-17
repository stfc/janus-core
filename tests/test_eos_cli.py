"""Test eis commandline interface."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from janus_core.cli.janus import app
from tests.utils import assert_log_contains

DATA_PATH = Path(__file__).parent / "data"

runner = CliRunner()


def test_help():
    """Test calling `janus eos --help`."""
    result = runner.invoke(app, ["eos", "--help"])
    assert result.exit_code == 0
    assert "Usage: janus eos [OPTIONS]" in result.stdout


def test_eos(tmp_path):
    """Test calculating the equation of state."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    eos_raw_path = tmp_path / "NaCl-eos-raw.dat"
    eos_fit_path = tmp_path / "NaCl-eos-fit.dat"
    result = runner.invoke(
        app,
        [
            "eos",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--file-prefix",
            tmp_path / "NaCl",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    assert eos_raw_path.exists()
    assert eos_fit_path.exists()

    # Check contents of raw data file
    with open(eos_raw_path, encoding="utf8") as eos_raw_file:
        lines = eos_raw_file.readlines()

    assert len(lines) == 8
    assert lines[0] == "#Lattice Scalar | Energy [eV] | Volume [Å^3] \n"
    assert lines[4].split()[0] == "1.0"
    assert float(lines[4].split()[1]) == pytest.approx(-27.046359959669214)
    assert float(lines[4].split()[2]) == pytest.approx(184.28932483484286)

    # Check contents of fitted data file
    with open(eos_fit_path, encoding="utf8") as eos_fit_file:
        lines = eos_fit_file.readlines()

    assert len(lines) == 2
    assert lines[0] == "#Bulk modulus [GPa] | Energy [eV] | Volume [Å^3] \n"
    assert float(lines[1].split()[0]) == pytest.approx(27.186555689697165)
    assert float(lines[1].split()[1]) == pytest.approx(-27.046361904823204)
    assert float(lines[1].split()[2]) == pytest.approx(184.22281215770133)

    # Check only initial structure is minimized
    assert_log_contains(
        log_path,
        includes=["Minimising initial structure"],
        excludes=["Minimising lattice scalar = 1.0"],
    )


def test_setting_lattice(tmp_path):
    """Test setting the lattice constants."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    eos_raw_path = tmp_path / "NaCl-eos-raw.dat"
    result = runner.invoke(
        app,
        [
            "eos",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--min-lattice",
            0.8,
            "--max-lattice",
            1.2,
            "--n-lattice",
            5,
            "--file-prefix",
            tmp_path / "NaCl",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    assert eos_raw_path.exists()

    # Check contents of raw data file
    with open(eos_raw_path, encoding="utf8") as eos_raw_file:
        lines = eos_raw_file.readlines()
    assert len(lines) == 6
    assert lines[3].split()[0] == "1.0"
    assert float(lines[1].split()[0]) == pytest.approx(0.8 ** (1 / 3))
    assert float(lines[5].split()[0]) == pytest.approx(1.2 ** (1 / 3))


test_data = [("--min-lattice", 1), ("--max-lattice", 0.9), ("--n-lattice", 0)]


@pytest.mark.parametrize("option, value", test_data)
def test_invalid_lattice(option, value, tmp_path):
    """Test setting the invalid lattice constants."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    result = runner.invoke(
        app,
        [
            "eos",
            "--struct",
            DATA_PATH / "NaCl.cif",
            option,
            value,
            "--file-prefix",
            tmp_path / "NaCl",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)


def test_minimising_all(tmp_path):
    """Test calculating the equation of state."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    result = runner.invoke(
        app,
        [
            "eos",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--minimize-all",
            "--file-prefix",
            tmp_path / "NaCl",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0

    # Check minimizes multiple structures
    assert_log_contains(
        log_path,
        includes=["Minimising initial structure", "Minimising lattice scalar = 1.0"],
    )
