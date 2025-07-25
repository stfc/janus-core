"""Test singlepoint commandline interface."""

from __future__ import annotations

from pathlib import Path
import shutil

from ase import Atoms
from ase.io import read
from typer.testing import CliRunner
import yaml

from janus_core.cli.janus import app
from tests.utils import (
    assert_log_contains,
    check_output_files,
    clear_log_handlers,
    read_atoms,
    strip_ansi_codes,
)

DATA_PATH = Path(__file__).parent / "data"
MACE_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"

runner = CliRunner()


def test_singlepoint_help():
    """Test calling `janus singlepoint --help`."""
    result = runner.invoke(app, ["singlepoint", "--help"])
    assert result.exit_code == 0
    assert "Usage: janus singlepoint [OPTIONS]" in strip_ansi_codes(result.stdout)


def test_singlepoint():
    """Test singlepoint calculation."""
    results_dir = Path("./janus_results")
    results_path = results_dir / "NaCl-results.extxyz"
    log_path = results_dir / "NaCl-singlepoint-log.yml"
    summary_path = results_dir / "NaCl-singlepoint-summary.yml"

    assert not results_dir.exists()

    try:
        result = runner.invoke(
            app,
            [
                "singlepoint",
                "--struct",
                DATA_PATH / "NaCl.cif",
                "--arch",
                "mace_mp",
            ],
        )
        assert result.exit_code == 0

        assert results_path.exists()
        assert log_path.exists()
        assert summary_path.exists

        atoms = read_atoms(results_path)

        assert "mace_mp_energy" in atoms.info

        assert "arch" in atoms.info
        assert "model" in atoms.info
        assert atoms.info["arch"] == "mace_mp"
        assert atoms.info["model"] == "small"

        assert "mace_mp_forces" in atoms.arrays
        assert "system_name" in atoms.info
        assert atoms.info["system_name"] == "NaCl"

        expected_units = {"energy": "eV", "forces": "ev/Ang", "stress": "ev/Ang^3"}
        assert "units" in atoms.info
        for prop, units in expected_units.items():
            assert atoms.info["units"][prop] == units

    finally:
        # Ensure files deleted if command fails
        shutil.rmtree(results_dir, ignore_errors=True)

        clear_log_handlers()


def test_properties(tmp_path):
    """Test properties for singlepoint calculation in a new directory."""
    results_path_1 = tmp_path / "test" / "H2O-energy-results.extxyz"
    results_path_2 = tmp_path / "test" / "H2O-stress-results.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    # Check energy can be calculated successfully
    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "H2O.cif",
            "--arch",
            "mace_mp",
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
            "--arch",
            "mace_mp",
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
            "--arch",
            "mace_mp",
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
            "--arch",
            "mace_mp",
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
            "--arch",
            "mace_mp",
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
            "--arch",
            "mace_mp",
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
    assert "config" in sp_summary
    assert "info" in sp_summary
    assert "end_time" in sp_summary

    assert "properties" in sp_summary["config"]
    assert "traj" in sp_summary["info"]
    assert "length" in sp_summary["info"]["traj"]
    assert "struct" in sp_summary["info"]["traj"]
    assert "n_atoms" in sp_summary["info"]["traj"]["struct"]

    assert "emissions" in sp_summary
    assert sp_summary["emissions"] > 0

    output_files = {
        "results": results_path,
        "log": log_path,
        "summary": summary_path,
    }

    check_output_files(summary=sp_summary, output_files=output_files)


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
            "--arch",
            "mace_mp",
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

    assert "index" in sp_summary["config"]["read_kwargs"]
    assert sp_summary["config"]["read_kwargs"]["index"] == 0


def test_invalid_config():
    """Test passing a config file with an invalid option name."""
    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "benzene-traj.xyz",
            "--arch",
            "mace_mp",
            "--config",
            DATA_PATH / "invalid.yml",
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)


def test_write_kwargs(tmp_path):
    """Test setting invalidate_calc and write_results via write_kwargs."""
    results_path = tmp_path / "NaCl-results.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--write-kwargs",
            "{'invalidate_calc': False}",
            "--out",
            results_path,
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    atoms = read(results_path)
    assert "mace_mp_energy" in atoms.info
    assert "mace_mp_forces" in atoms.arrays
    assert "energy" in atoms.calc.results
    assert "forces" in atoms.calc.results


def test_write_cif(tmp_path):
    """Test writing out a cif file."""
    results_path = tmp_path / "NaCl-results.cif"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--write-kwargs",
            "{'invalidate_calc': False, 'write_results': True}",
            "--out",
            results_path,
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    atoms = read(results_path)
    assert isinstance(atoms, Atoms)


def test_hessian(tmp_path):
    """Test Hessian calculation."""
    results_path = tmp_path / "NaCl-results.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    # Check Hessian can be calculated successfully
    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--properties",
            "hessian",
            "--properties",
            "energy",
            "--out",
            results_path,
            "--calc-kwargs",
            "{'dispersion': True}",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0

    atoms = read(results_path)
    assert "mace_mp_d3_energy" in atoms.info
    assert "mace_mp_d3_hessian" in atoms.info
    assert "mace_mp_d3_stress" not in atoms.info
    assert atoms.info["mace_mp_d3_hessian"].shape == (24, 8, 3)
    assert atoms.info["units"]["hessian"] == "ev/Ang^2"


def test_no_carbon(tmp_path):
    """Test disabling carbon tracking."""
    results_path = tmp_path / "NaCl-results.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--properties",
            "energy",
            "--out",
            results_path,
            "--log",
            log_path,
            "--no-tracker",
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0

    # Read singlepoint summary file
    with open(summary_path, encoding="utf8") as file:
        sp_summary = yaml.safe_load(file)
    assert "emissions" not in sp_summary


def test_file_prefix(tmp_path):
    """Test file prefix creates directories and affects all files."""
    file_prefix = tmp_path / "test/test"
    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 0
    test_path = tmp_path / "test"
    assert list(tmp_path.iterdir()) == [test_path]
    assert set(test_path.iterdir()) == {
        test_path / "test-results.extxyz",
        test_path / "test-singlepoint-summary.yml",
        test_path / "test-singlepoint-log.yml",
    }


def test_model(tmp_path):
    """Test model passed correctly."""
    file_prefix = tmp_path / "NaCl"
    results_path = tmp_path / "NaCl-results.extxyz"
    log_path = tmp_path / "test.log"

    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--model",
            MACE_PATH,
            "--log",
            log_path,
            "--file-prefix",
            file_prefix,
            "--no-tracker",
        ],
    )
    assert result.exit_code == 0

    assert_log_contains(
        log_path, excludes=["FutureWarning: `model_path` has been deprecated."]
    )

    atoms = read(results_path)
    assert "model" in atoms.info
    assert atoms.info["model"] == str(MACE_PATH.as_posix())


def test_model_path_deprecated(tmp_path):
    """Test model_path sets model."""
    file_prefix = tmp_path / "NaCl"
    results_path = tmp_path / "NaCl-results.extxyz"
    log_path = tmp_path / "test.log"

    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--model-path",
            MACE_PATH,
            "--log",
            log_path,
            "--file-prefix",
            file_prefix,
            "--no-tracker",
        ],
    )
    assert result.exit_code == 0

    atoms = read(results_path)
    assert "model" in atoms.info
    assert atoms.info["model"] == str(MACE_PATH.as_posix())


def test_missing_arch(tmp_path):
    """Test no architecture specified."""
    file_prefix = tmp_path / "NaCl"

    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 2
    assert "Missing option" in result.stdout
