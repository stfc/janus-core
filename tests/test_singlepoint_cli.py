"""Test singlepoint commandline interface."""

from pathlib import Path

from ase.io import read
from typer.testing import CliRunner
import yaml

from janus_core.cli.janus import app
from tests.utils import assert_log_contains, read_atoms

DATA_PATH = Path(__file__).parent / "data"

runner = CliRunner()

# Many pylint warnings now raised due to similar log/summary flags
# These depend on tmp_path, so not easily refactorisable
# pylint: disable=duplicate-code


def test_singlepoint_help():
    """Test calling `janus singlepoint --help`."""
    result = runner.invoke(app, ["singlepoint", "--help"])
    assert result.exit_code == 0
    assert "Usage: janus singlepoint [OPTIONS]" in result.stdout


def test_singlepoint(tmp_path):
    """Test singlepoint calculation."""
    results_path = Path("./NaCl-results.extxyz").absolute()
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )

    # Check atoms can read read, then delete file
    atoms = read_atoms(results_path)
    assert result.exit_code == 0
    assert "mace_mp_energy" in atoms.info
    assert "mace_mp_forces" in atoms.arrays


def test_properties(tmp_path):
    """Test properties for singlepoint calculation."""
    results_path_1 = tmp_path / "H2O-energy-results.extxyz"
    results_path_2 = tmp_path / "H2O-stress-results.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    # Check energy is can be calculated successfully
    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "H2O.cif",
            "--properties",
            "energy",
            "--out",
            results_path_1,
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0

    atoms = read(results_path_1)
    assert "mace_mp_energy" in atoms.info
    assert "mace_stress" not in atoms.info

    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "H2O.cif",
            "--properties",
            "stress",
            "--out",
            results_path_2,
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0

    atoms = read(results_path_2)
    assert "mace_mp_stress" in atoms.info
    assert "mace_mp_energy" not in atoms.info


def test_read_kwargs(tmp_path):
    """Test setting read_kwargs for singlepoint calculation."""
    results_path = tmp_path / "benzene-traj-results.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "benzene-traj.xyz",
            "--read-kwargs",
            "{'index': ':'}",
            "--out",
            results_path,
            "--properties",
            "energy",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0

    atoms = read(results_path, index=":")
    assert isinstance(atoms, list)


def test_calc_kwargs(tmp_path):
    """Test setting calc_kwargs for singlepoint calculation."""
    results_path = tmp_path / "NaCl-results.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--calc-kwargs",
            "{'default_dtype': 'float32'}",
            "--out",
            results_path,
            "--properties",
            "energy",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    assert "Using float32 for MACECalculator" in result.stdout


def test_log(tmp_path):
    """Test log correctly written for singlepoint."""
    results_path = tmp_path / "NaCl-results.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--out",
            results_path,
            "--properties",
            "energy",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0

    assert_log_contains(log_path, includes=["Starting single point calculation"])


def test_summary(tmp_path):
    """Test summary file can be read correctly."""
    results_path = tmp_path / "benzene-traj-results.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "benzene-traj.xyz",
            "--read-kwargs",
            "{'index': ':'}",
            "--out",
            results_path,
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )

    assert result.exit_code == 0

    # Read singlepoint summary file
    with open(summary_path, encoding="utf8") as file:
        sp_summary = yaml.safe_load(file)

    assert "command" in sp_summary
    assert "janus singlepoint" in sp_summary["command"]
    assert "start_time" in sp_summary
    assert "inputs" in sp_summary
    assert "end_time" in sp_summary

    assert "traj" in sp_summary["inputs"]
    assert "length" in sp_summary["inputs"]["traj"]
    assert "struct" in sp_summary["inputs"]["traj"]
    assert "n_atoms" in sp_summary["inputs"]["traj"]["struct"]


def test_config(tmp_path):
    """Test passing a config file with read kwargs, and values to be overwritten."""
    results_path = tmp_path / "benzene-traj-results.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "benzene-traj.xyz",
            "--out",
            results_path,
            "--log",
            log_path,
            "--summary",
            summary_path,
            "--config",
            DATA_PATH / "singlepoint_config.yml",
        ],
    )
    assert result.exit_code == 0
    atoms = read(results_path, index=":")
    assert len(atoms) == 1

    # Read singlepoint summary file
    with open(summary_path, encoding="utf8") as file:
        sp_summary = yaml.safe_load(file)

    assert "index" in sp_summary["inputs"]["calc"]["read_kwargs"]
    assert sp_summary["inputs"]["calc"]["read_kwargs"]["index"] == 0


def test_invalid_config():
    """Test passing a config file with an invalid option name."""
    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "benzene-traj.xyz",
            "--config",
            DATA_PATH / "invalid.yml",
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)


def test_write_kwargs(tmp_path):
    """Test setting invalidate_calc and write_results via write_kwargs."""
    results_path = tmp_path / "NaCl-results.extxyz"
    write_kwargs = "{'invalidate_calc': False, 'write_calc_results': True}"

    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--write-kwargs",
            write_kwargs,
            "--out",
            results_path,
        ],
    )
    assert result.exit_code == 0
    atoms = read(results_path)
    assert "mace_mp_energy" in atoms.info
    assert "mace_mp_forces" in atoms.arrays
    assert "energy" in atoms.calc.results
    assert "forces" in atoms.calc.results
