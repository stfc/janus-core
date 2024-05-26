"""Test phonons commandline interface."""

from pathlib import Path

import pytest
from typer.testing import CliRunner
import yaml

from janus_core.cli.janus import app

DATA_PATH = Path(__file__).parent / "data"

runner = CliRunner()


def test_help():
    """Test calling `janus phonons --help`."""
    result = runner.invoke(app, ["phonons", "--help"])
    assert result.exit_code == 0
    assert "Usage: janus phonons [OPTIONS]" in result.stdout


def test_bands(tmp_path):
    """Test calculating force constants and bands."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    phonon_results = tmp_path / "NaCl-phonopy.yml"
    autoband_results = tmp_path / "NaCl-auto_bands.yml"
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
            "--band",
        ],
    )
    assert result.exit_code == 0
    assert phonon_results.exists()
    assert autoband_results.exists()


def test_bands_simple(tmp_path):
    """Test calculating force constants and reduced bands information."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    autoband_results = tmp_path / "NaCl-auto_bands.yml"
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
            "--band",
            "--no-write-full",
        ],
    )
    assert result.exit_code == 0
    assert autoband_results.exists()
    with open(autoband_results, encoding="utf8") as file:
        bands = yaml.safe_load(file)
        assert "eigenvector" not in bands["phonon"][0]["band"][0].keys()


def test_hdf5(tmp_path):
    """Test calculating phonons."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    phonon_results = tmp_path / "NaCl-phonopy.yml"
    hdf5_results = tmp_path / "NaCl-force_constants.hdf5"
    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--file-prefix",
            tmp_path / "NaCl",
            "--hdf5",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    assert phonon_results.exists()
    assert hdf5_results.exists()


def test_thermal_props(tmp_path):
    """Test calculating thermal properties."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    thermal_results = tmp_path / "NaCl-thermal.dat"
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


def test_plot(tmp_path):
    """Test for ploting routines"""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    pdos_results = tmp_path / "NaCl-pdos.dat"
    dos_results = tmp_path / "NaCl-dos.dat"
    hdf5_results = tmp_path / "NaCl-force_constants.hdf5"
    autoband_results = tmp_path / "NaCl-auto_bands.yml"
    svgs = [
        tmp_path / "NaCl-dos.svg",
        tmp_path / "NaCl-pdos.svg",
        tmp_path / "NaCl-bs-dos.svg",
        tmp_path / "NaCl-auto_bands.svg",
    ]
    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--pdos",
            "--dos",
            "--band",
            "--hdf5",
            "--plot-to-file",
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
    assert dos_results.exists()
    assert hdf5_results.exists()
    assert autoband_results.exists()
    for svg in svgs:
        assert svg.exists()
    with open(autoband_results, encoding="utf8") as file:
        bands = yaml.safe_load(file)
        assert "eigenvector" in bands["phonon"][0]["band"][0].keys()


def test_supercell(tmp_path):
    """Test setting the supercell."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    param_file = tmp_path / "NaCl-phonopy.yml"
    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--supercell",
            "1x2x3",
            "--file-prefix",
            tmp_path / "NaCl",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0

    # Check parameters
    with open(param_file, encoding="utf8") as file:
        params = yaml.safe_load(file)

    assert "supercell_matrix" in params
    assert len(params["supercell_matrix"]) == 3
    assert params["supercell_matrix"] == [[1, 0, 0], [0, 2, 0], [0, 0, 3]]


test_data = ["2", "2.1x2.1x2.1", "2x2xa"]


@pytest.mark.parametrize("supercell", test_data)
def test_invalid_supercell(supercell, tmp_path):
    """Test errors are raise for invalid supercells."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--supercell",
            supercell,
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
