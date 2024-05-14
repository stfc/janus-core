"""Test phonons commandline interface."""

from pathlib import Path

from typer.testing import CliRunner

from janus_core.cli.janus import app

DATA_PATH = Path(__file__).parent / "data"

runner = CliRunner()


def test_help():
    """Test calling `janus phonons --help`."""
    result = runner.invoke(app, ["phonons", "--help"])
    assert result.exit_code == 0
    assert "Usage: janus phonons [OPTIONS]" in result.stdout


def test_phonons(tmp_path):
    """Test calculating phonons."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    phonon_results = tmp_path / "NaCl-ase.yml"
    autoband_results = tmp_path / "NaCl-auto-band.yml"
    result = runner.invoke(
        app,
        [
            "phonons",
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
    assert phonon_results.exists()
    assert autoband_results.exists()


def test_thermal_props(tmp_path):
    """Test calculating thermal properties."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    thermal_results = tmp_path / "NaCl-cv.dat"
    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--thermal",
            "--file-prefix",
            tmp_path / "NaCl",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    assert thermal_results.exists()


def test_dos(tmp_path):
    """Test calculating the DOS."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    dos_results = tmp_path / "NaCl-dos.dat"
    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--dos",
            "--file-prefix",
            tmp_path / "NaCl",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    assert dos_results.exists()


def test_pdos(tmp_path):
    """Test calculating the PDOS."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    pdos_results = tmp_path / "NaCl-pdos.dat"
    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--pdos",
            "--file-prefix",
            tmp_path / "NaCl",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    assert pdos_results.exists()
