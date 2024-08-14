"""Test phonons commandline interface."""

from pathlib import Path

import pytest
from typer.testing import CliRunner
import yaml

from janus_core.cli.janus import app
from tests.utils import assert_log_contains, strip_ansi_codes

DATA_PATH = Path(__file__).parent / "data"

runner = CliRunner()


def test_help():
    """Test calling `janus phonons --help`."""
    result = runner.invoke(app, ["phonons", "--help"])
    assert result.exit_code == 0
    assert "Usage: janus phonons [OPTIONS]" in strip_ansi_codes(result.stdout)


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
            "--bands",
        ],
    )
    assert result.exit_code == 0
    assert phonon_results.exists()
    assert autoband_results.exists()

    # Read phonons summary file
    assert summary_path.exists()
    with open(summary_path, encoding="utf8") as file:
        phonon_summary = yaml.safe_load(file)

    assert "command" in phonon_summary
    assert "janus phonons" in phonon_summary["command"]
    assert "start_time" in phonon_summary
    assert "inputs" in phonon_summary
    assert "end_time" in phonon_summary

    assert "emissions" in phonon_summary
    assert phonon_summary["emissions"] > 0


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
            "--bands",
            "--no-write-full",
        ],
    )
    assert result.exit_code == 0

    assert autoband_results.exists()
    with open(autoband_results, encoding="utf8") as file:
        bands = yaml.safe_load(file)
    assert "eigenvector" not in bands["phonon"][0]["band"][0]

    # Read phonons summary file
    assert summary_path.exists()
    with open(summary_path, encoding="utf8") as file:
        phonon_summary = yaml.safe_load(file)

    assert "command" in phonon_summary
    assert "janus phonons" in phonon_summary["command"]
    assert "inputs" in phonon_summary
    assert "calcs" in phonon_summary["inputs"]
    assert phonon_summary["inputs"]["calcs"][0] == "bands"


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
    """Test for ploting routines."""
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
            "--bands",
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
    for svg in svgs:
        assert svg.exists()

    assert autoband_results.exists()
    with open(autoband_results, encoding="utf8") as file:
        bands = yaml.safe_load(file)
    assert "eigenvector" in bands["phonon"][0]["band"][0]
    assert "group_velocity" in bands["phonon"][0]["band"][0]

    # Read phonons summary file
    assert summary_path.exists()
    with open(summary_path, encoding="utf8") as file:
        phonon_summary = yaml.safe_load(file)
    assert phonon_summary["inputs"]["calcs"][0] == "bands"
    assert phonon_summary["inputs"]["calcs"][1] == "dos"
    assert phonon_summary["inputs"]["calcs"][2] == "pdos"


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


def test_minimize_kwargs(tmp_path):
    """Test setting optimizer function and writing optimized structure."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    file_prefix = tmp_path / "test"
    opt_path = tmp_path / "test-opt.extxyz"

    minimize_kwargs = "{'optimizer': 'FIRE', 'write_results': True}"

    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--minimize",
            "--minimize-kwargs",
            minimize_kwargs,
            "--file-prefix",
            file_prefix,
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0

    assert_log_contains(
        log_path,
        includes=["Starting geometry optimization", "Using optimizer: FIRE"],
    )
    assert opt_path.exists()


def test_minimize_filename(tmp_path):
    """Test minimize filename overwrites default."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    file_prefix = tmp_path / "test"
    opt_path = tmp_path / "geomopt-opt.extxyz"

    # write_results should be set automatically
    minimize_kwargs = f"{{'write_kwargs': {{'filename': '{str(opt_path)}'}}}}"

    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--minimize",
            "--minimize-kwargs",
            minimize_kwargs,
            "--file-prefix",
            file_prefix,
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    assert opt_path.exists()


@pytest.mark.parametrize("read_kwargs", ["{'index': 0}", "{}"])
def test_valid_traj_input(read_kwargs, tmp_path):
    """Test valid trajectory input structure handled."""
    phonon_results = tmp_path / "traj-phonopy.yml"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl-traj.xyz",
            "--read-kwargs",
            read_kwargs,
            "--file-prefix",
            tmp_path / "traj",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    assert phonon_results.exists()


def test_invalid_traj_input(tmp_path):
    """Test invalid trajectory input structure handled."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl-traj.xyz",
            "--read-kwargs",
            "{'index': ':'}",
            "--file-prefix",
            tmp_path / "traj",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
