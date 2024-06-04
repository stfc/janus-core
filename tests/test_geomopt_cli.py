"""Test geomopt commandline interface."""

from pathlib import Path

from ase.io import read
import pytest
from typer.testing import CliRunner
import yaml

from janus_core.cli.janus import app
from tests.utils import assert_log_contains, read_atoms

DATA_PATH = Path(__file__).parent / "data"

runner = CliRunner()

# Many pylint now warnings raised due to similar log/summary flags
# These depend on tmp_path, so not easily refactorisable
# pylint: disable=duplicate-code


def test_help():
    """Test calling `janus geomopt --help`."""
    result = runner.invoke(app, ["geomopt", "--help"])
    assert result.exit_code == 0
    assert "Usage: janus geomopt [OPTIONS]" in result.stdout


def test_geomopt(tmp_path):
    """Test geomopt calculation."""
    results_path = Path("./NaCl-opt.xyz").absolute()
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    assert not results_path.exists()

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--fmax",
            "0.2",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    read_atoms(results_path)
    assert result.exit_code == 0


def test_log(tmp_path):
    """Test log correctly written for geomopt."""
    results_path = tmp_path / "NaCl-opt.xyz"
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

    assert_log_contains(
        log_path, includes="Starting geometry optimization", excludes="Using filter"
    )


def test_traj(tmp_path):
    """Test trajectory correctly written for geomopt."""
    results_path = tmp_path / "NaCl-opt.xyz"
    traj_path = f"{tmp_path}/test.xyz"
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
    assert "forces" in atoms.arrays


def test_fully_opt(tmp_path):
    """Test passing --fully-opt without --vectors-only"""
    results_path = tmp_path / "NaCl-opt.xyz"
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
            "--fully-opt",
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
    expected = [5.68834069, 5.68893345, 5.68932555, 89.75938298, 90.0, 90.0]
    assert atoms.cell.cellpar() == pytest.approx(expected)


def test_fully_opt_and_vectors(tmp_path):
    """Test passing --fully-opt with --vectors-only."""
    results_path = tmp_path / "NaCl-opt.xyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl-deformed.cif",
            "--fully-opt",
            "--vectors-only",
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
    expected = [5.69139709, 5.69139709, 5.69139709, 89.0, 90.0, 90.0]
    assert atoms.cell.cellpar() == pytest.approx(expected)


def test_vectors_not_fully_opt(tmp_path):
    """Test passing --vectors-only without --fully-opt."""
    results_path = tmp_path / "NaCl-opt.xyz"
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
            "--vectors-only",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0

    assert_log_contains(log_path, includes=["Using filter", "hydrostatic_strain: True"])


test_data = ["--vectors-only", "--fully-opt"]


@pytest.mark.parametrize("option", test_data)
def test_scalar_pressure(option, tmp_path):
    """Test passing --pressure with --vectors-only."""
    results_path = tmp_path / "NaCl-opt.xyz"
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
            "100",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    assert_log_contains(log_path, includes=["scalar_pressure: 6.24"])


def test_duplicate_traj(tmp_path):
    """Test trajectory file cannot be not passed via traj_kwargs."""
    traj_path = tmp_path / "NaCl-traj.xyz"
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
    results_path = tmp_path / "NaCl-opt.xyz"
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
    intermediate_energy = atoms.get_potential_energy()

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
    final_energy = atoms.get_potential_energy()
    assert final_energy < intermediate_energy


def test_summary(tmp_path):
    """Test summary file can be read correctly."""
    results_path = tmp_path / "NaCl-results.xyz"
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


def test_config(tmp_path):
    """Test passing a config file with opt_kwargs."""
    results_path = tmp_path / "NaCl-results.xyz"
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
    """Test setting constant volume with --fully-opt."""
    results_path = tmp_path / "NaCl-opt.xyz"
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
            "--fully-opt",
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
    results_path = tmp_path / "NaCl-opt.xyz"
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
    results_path = tmp_path / "NaCl-opt.xyz"
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
            "--fully-opt",
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
    """Test setting filter function without --fully-opt or --vectors-only."""
    results_path = tmp_path / "NaCl-opt.xyz"
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
