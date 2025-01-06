"""Test geomopt commandline interface."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import read
import pytest
from typer.testing import CliRunner
import yaml

from janus_core.cli.janus import app
from tests.utils import (
    assert_log_contains,
    clear_log_handlers,
    read_atoms,
    strip_ansi_codes,
)

DATA_PATH = Path(__file__).parent / "data"

runner = CliRunner()


def test_help():
    """Test calling `janus geomopt --help`."""
    result = runner.invoke(app, ["geomopt", "--help"])
    assert result.exit_code == 0
    assert "Usage: janus geomopt [OPTIONS]" in strip_ansi_codes(result.stdout)


def test_geomopt():
    """Test geomopt calculation."""
    results_path = Path("./NaCl-opt.extxyz").absolute()
    log_path = Path("./NaCl-geomopt-log.yml").absolute()
    summary_path = Path("./NaCl-geomopt-summary.yml").absolute()

    assert not results_path.exists()
    assert not log_path.exists()
    assert not summary_path.exists()

    try:
        result = runner.invoke(
            app,
            [
                "geomopt",
                "--struct",
                DATA_PATH / "NaCl.cif",
                "--fmax",
                "0.2",
            ],
        )
        assert result.exit_code == 0

        assert results_path.exists()
        assert log_path.exists()
        assert summary_path.exists()

    finally:
        # Ensure files deleted if command fails
        read_atoms(results_path)
        log_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
        clear_log_handlers()


def test_log(tmp_path):
    """Test log correctly written for geomopt."""
    results_path = tmp_path / "NaCl-opt.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--out",
            results_path,
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0

    # Only check reduced precision of energy and max force
    assert_log_contains(
        log_path,
        includes=[
            "Starting geometry optimization",
            "Final energy: -27.035127",
            "Max force: ",
        ],
        excludes="Using filter",
    )


def test_traj(tmp_path):
    """Test trajectory correctly written for geomopt."""
    results_path = tmp_path / "NaCl-opt.extxyz"
    traj_path = f"{tmp_path}/test.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--out",
            results_path,
            "--traj",
            traj_path,
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    atoms = read(traj_path)
    assert "mace_mp_forces" in atoms.arrays
    assert "system_name" in atoms.info
    assert atoms.info["system_name"] == "NaCl"


def test_opt_fully(tmp_path):
    """Test passing --opt-cell-fully without --opt-cell-lengths."""
    results_path = tmp_path / "NaCl-opt.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl-deformed.cif",
            "--out",
            results_path,
            "--opt-cell-fully",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0

    assert_log_contains(
        log_path,
        includes=["Using filter", "hydrostatic_strain: False", "scalar_pressure: 0.0"],
    )

    atoms = read(results_path)
    expected = [
        5.688268799219085,
        5.688750772505896,
        5.688822747326383,
        89.26002493790229,
        90.0,
        90.0,
    ]
    assert atoms.cell.cellpar() == pytest.approx(expected)
    assert "system_name" in atoms.info
    assert atoms.info["system_name"] == "NaCl-deformed"


def test_opt_fully_and_vectors(tmp_path):
    """Test passing --opt-cell-fully with --opt-cell-lengths."""
    results_path = tmp_path / "NaCl-opt.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl-deformed.cif",
            "--opt-cell-fully",
            "--opt-cell-lengths",
            "--out",
            results_path,
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0

    assert_log_contains(log_path, includes=["Using filter", "hydrostatic_strain: True"])

    atoms = read(results_path)
    expected = [
        5.687545288920282,
        5.687545288920282,
        5.687545288920282,
        89.0,
        90.0,
        90.0,
    ]
    assert atoms.cell.cellpar() == pytest.approx(expected)


def test_vectors_not_opt_fully(tmp_path):
    """Test passing --opt-cell-lengths without --opt-cell-fully."""
    results_path = tmp_path / "NaCl-opt.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--out",
            results_path,
            "--opt-cell-lengths",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0

    assert_log_contains(log_path, includes=["Using filter", "hydrostatic_strain: True"])


test_data = ["--opt-cell-lengths", "--opt-cell-fully"]


@pytest.mark.parametrize("option", test_data)
def test_scalar_pressure(option, tmp_path):
    """Test passing --pressure with --opt-cell-lengths."""
    results_path = tmp_path / "NaCl-opt.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl-deformed.cif",
            "--out",
            results_path,
            option,
            "--pressure",
            "0.01",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    assert_log_contains(log_path, includes=["scalar_pressure: 0.01 GPa"])


def test_duplicate_traj(tmp_path):
    """Test trajectory file cannot be not passed via traj_kwargs."""
    traj_path = tmp_path / "NaCl-traj.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    minimize_kwargs = f"{{'opt_kwargs': {{'trajectory' : '{str(traj_path)}'}}}}"

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--minimize-kwargs",
            minimize_kwargs,
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)


def test_restart(tmp_path):
    """Test restarting geometry optimization."""
    data_path = DATA_PATH / "NaCl-deformed.cif"
    restart_path = tmp_path / "NaCl-res.pkl"
    results_path = tmp_path / "NaCl-opt.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    minimize_kwargs = f"{{'opt_kwargs': {{'restart': '{str(restart_path)}'}}}}"

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            data_path,
            "--out",
            results_path,
            "--minimize-kwargs",
            minimize_kwargs,
            "--steps",
            2,
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    atoms = read(results_path)
    intermediate_energy = atoms.info["mace_mp_energy"]

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--out",
            results_path,
            "--minimize-kwargs",
            minimize_kwargs,
            "--steps",
            2,
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    atoms = read(results_path)
    final_energy = atoms.info["mace_mp_energy"]
    assert final_energy < intermediate_energy


def test_summary(tmp_path):
    """Test summary file can be read correctly."""
    results_path = tmp_path / "NaCl-results.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--out",
            results_path,
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0

    # Read geomopt summary file
    with open(summary_path, encoding="utf8") as file:
        geomopt_summary = yaml.safe_load(file)

    assert "command" in geomopt_summary
    assert "janus geomopt" in geomopt_summary["command"]
    assert "start_time" in geomopt_summary
    assert "end_time" in geomopt_summary

    assert "inputs" in geomopt_summary
    assert "opt_kwargs" in geomopt_summary["inputs"]
    assert "struct" in geomopt_summary["inputs"]
    assert "n_atoms" in geomopt_summary["inputs"]["struct"]

    assert "emissions" in geomopt_summary
    assert geomopt_summary["emissions"] > 0


def test_config(tmp_path):
    """Test passing a config file with opt_kwargs."""
    results_path = tmp_path / "NaCl-results.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--out",
            results_path,
            "--log",
            log_path,
            "--summary",
            summary_path,
            "--config",
            DATA_PATH / "geomopt_config.yml",
        ],
    )
    assert result.exit_code == 0

    # Read geomopt summary file
    with open(summary_path, encoding="utf8") as file:
        geomopt_summary = yaml.safe_load(file)

    assert "alpha" in geomopt_summary["inputs"]["opt_kwargs"]
    assert geomopt_summary["inputs"]["opt_kwargs"]["alpha"] == 100


def test_invalid_config():
    """Test passing a config file with an invalid option name."""
    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--config",
            DATA_PATH / "invalid.yml",
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)


def test_const_volume(tmp_path):
    """Test setting constant volume with --opt-cell-fully."""
    results_path = tmp_path / "NaCl-opt.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    minimize_kwargs = "{'filter_kwargs': {'constant_volume' : True}}"

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl-deformed.cif",
            "--out",
            results_path,
            "--opt-cell-fully",
            "--minimize-kwargs",
            minimize_kwargs,
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    assert_log_contains(log_path, includes=["constant_volume: True"])


def test_optimizer_str(tmp_path):
    """Test setting optimizer function."""
    results_path = tmp_path / "NaCl-opt.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl-deformed.cif",
            "--out",
            results_path,
            "--optimizer",
            "FIRE",
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


def test_filter_str(tmp_path):
    """Test setting filter function."""
    results_path = tmp_path / "NaCl-opt.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl-deformed.cif",
            "--out",
            results_path,
            "--opt-cell-fully",
            "--filter-func",
            "UnitCellFilter",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    assert_log_contains(
        log_path,
        includes=["Starting geometry optimization", "Using filter: UnitCellFilter"],
    )


def test_filter_str_error(tmp_path):
    """Test setting filter function without --opt-cell-fully or --opt-cell-lengths."""
    results_path = tmp_path / "NaCl-opt.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl-deformed.cif",
            "--out",
            results_path,
            "--filter-func",
            "UnitCellFilter",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)


@pytest.mark.parametrize("read_kwargs", ["{'index': 1}", "{}"])
def test_valid_traj_input(read_kwargs, tmp_path):
    """Test valid trajectory input structure handled."""
    results_path = tmp_path / "benezene-traj.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "benzene-traj.xyz",
            "--out",
            results_path,
            "--read-kwargs",
            read_kwargs,
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    atoms = read(results_path)
    assert isinstance(atoms, Atoms)


def test_invalid_traj_input(tmp_path):
    """Test invalid trajectory input structure handled."""
    results_path = tmp_path / "benezene-traj.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "benzene-traj.xyz",
            "--out",
            results_path,
            "--read-kwargs",
            "{'index': ':'}",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)


def test_reuse_output(tmp_path):
    """Test using geomopt output as new input."""
    results_path_1 = tmp_path / "test" / "NaCl-opt-1.extxyz"
    results_path_2 = tmp_path / "test" / "NaCl-opt-2.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl-deformed.cif",
            "--fmax",
            0.1,
            "--out",
            results_path_1,
            "--log",
            log_path,
            "--summary",
            summary_path,
            "--config",
            DATA_PATH / "geomopt_config.yml",
        ],
    )
    assert result.exit_code == 0
    results_1 = read(results_path_1)

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            results_path_1,
            "--fmax",
            0.01,
            "--out",
            results_path_2,
            "--log",
            log_path,
            "--summary",
            summary_path,
            "--config",
            DATA_PATH / "geomopt_config.yml",
        ],
    )
    assert result.exit_code == 0
    results_2 = read(results_path_2)
    assert results_1.positions != pytest.approx(results_2.positions)


def test_symmetrize(tmp_path):
    """Test symmetrizing final structure."""
    results_path_1 = tmp_path / "test" / "NaCl-opt-1.extxyz"
    results_path_2 = tmp_path / "test" / "NaCl-opt-2.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl-deformed.cif",
            "--fmax",
            0.001,
            "--out",
            results_path_1,
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    results_1 = read(results_path_1)

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            results_path_1,
            "--fmax",
            0.001,
            "--out",
            results_path_2,
            "--symmetrize",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    results_2 = read(results_path_2)
    assert results_1.positions != pytest.approx(results_2.positions)

    expected = [
        5.619999999999999,
        5.619999999999999,
        5.619999999999999,
        89.00000000000003,
        90.0,
        90.0,
    ]
    assert results_2.cell.cellpar() == pytest.approx(expected)


def test_no_carbon(tmp_path):
    """Test disabling carbon tracking."""
    results_path = tmp_path / "NaCl-results.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl.cif",
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

    # Read geomopt summary file
    with open(summary_path, encoding="utf8") as file:
        geomopt_summary = yaml.safe_load(file)
    assert "emissions" not in geomopt_summary


def test_units(tmp_path):
    """Test correct units are saved."""
    results_path = tmp_path / "NaCl-opt.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl.cif",
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
    expected_units = {"energy": "eV", "forces": "ev/Ang", "stress": "ev/Ang^3"}
    assert "units" in atoms.info
    for prop, units in expected_units.items():
        assert atoms.info["units"][prop] == units
