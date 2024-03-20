"""Test geomopt commandline interface."""

from pathlib import Path

from ase.io import read
import pytest
from typer.testing import CliRunner

from janus_core.cli import app
from tests.utils import read_atoms

DATA_PATH = Path(__file__).parent / "data"

runner = CliRunner()


def test_geomopt_help():
    """Test calling `janus geomopt --help`."""
    result = runner.invoke(app, ["geomopt", "--help"])
    assert result.exit_code == 0
    # Command is returned as "root"
    assert "Usage: root geomopt [OPTIONS]" in result.stdout


def test_geomopt():
    """Test geomopt calculation."""
    results_path = Path("./Cl4Na4-opt.xyz").absolute()
    assert not results_path.exists()

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--max-force",
            "0.2",
        ],
    )

    read_atoms(results_path)
    assert result.exit_code == 0


def test_geomopt_log(tmp_path, caplog):
    """Test log correctly written for geomopt."""
    results_path = tmp_path / "NaCl-opt.xyz"

    with caplog.at_level("INFO", logger="janus_core.geom_opt"):
        result = runner.invoke(
            app,
            [
                "geomopt",
                "--struct",
                DATA_PATH / "NaCl.cif",
                "--write-kwargs",
                f"{{'filename': '{str(results_path)}'}}",
                "--log",
                f"{tmp_path}/test.log",
            ],
        )
        assert result.exit_code == 0
        assert "Starting geometry optimization" in caplog.text
        assert "Using filter" not in caplog.text


def test_geomopt_traj(tmp_path):
    """Test trajectory correctly written for geomopt."""
    results_path = tmp_path / "NaCl-opt.xyz"
    traj_path = f"{tmp_path}/test.xyz"

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--write-kwargs",
            f"{{'filename': '{str(results_path)}'}}",
            "--traj",
            traj_path,
        ],
    )
    assert result.exit_code == 0
    atoms = read(traj_path)
    assert "forces" in atoms.arrays


def test_fully_opt(tmp_path, caplog):
    """Test passing --fully-opt without --vectors-only"""
    results_path = tmp_path / "NaCl-opt.xyz"

    with caplog.at_level("INFO", logger="janus_core.geom_opt"):
        result = runner.invoke(
            app,
            [
                "geomopt",
                "--struct",
                DATA_PATH / "NaCl-deformed.cif",
                "--write-kwargs",
                f"{{'filename': '{str(results_path)}'}}",
                "--fully-opt",
            ],
        )
        assert result.exit_code == 0
        assert "Using filter" in caplog.text

        atoms = read(results_path)
        expected = [5.68834069, 5.68893345, 5.68932555, 89.75938298, 90.0, 90.0]
        assert atoms.cell.cellpar() == pytest.approx(expected)


def test_fully_opt_and_vectors(tmp_path, caplog):
    """Test passing --fully-opt with --vectors-only."""
    results_path = tmp_path / "NaCl-opt.xyz"
    with caplog.at_level("INFO", logger="janus_core.geom_opt"):
        result = runner.invoke(
            app,
            [
                "geomopt",
                "--struct",
                DATA_PATH / "NaCl-deformed.cif",
                "--fully-opt",
                "--vectors-only",
                "--write-kwargs",
                f"{{'filename': '{str(results_path)}'}}",
                "--log",
                f"{tmp_path}/test.log",
            ],
        )
        assert result.exit_code == 0
        assert "Using filter" in caplog.text

        atoms = read(results_path)
        expected = [5.69139709, 5.69139709, 5.69139709, 89.0, 90.0, 90.0]
        assert atoms.cell.cellpar() == pytest.approx(expected)


def test_vectors_not_fully_opt(tmp_path):
    """Test passing --vectors-only without --fully-opt."""
    results_path = tmp_path / "NaCl-opt.xyz"
    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--write-kwargs",
            f"{{'filename': '{str(results_path)}'}}",
            "--vectors-only",
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)


def duplicate_traj(tmp_path):
    """Test trajectory file cannot be not passed via traj_kwargs."""
    traj_path = tmp_path / "NaCl-traj.xyz"
    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--opt-kwargs",
            f"{{'trajectory': '{str(traj_path)}'}}",
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)


def test_restart(tmp_path):
    """Test restarting geometry optimization."""
    data_path = DATA_PATH / "NaCl-deformed.cif"
    restart_path = tmp_path / "NaCl-res.pkl"
    results_path = tmp_path / "NaCl-opt.xyz"

    result = runner.invoke(
        app,
        [
            "geomopt",
            "--struct",
            data_path,
            "--write-kwargs",
            f"{{'filename': '{str(results_path)}'}}",
            "--opt-kwargs",
            f"{{'restart': '{str(restart_path)}'}}",
            "--steps",
            2,
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
            "--write-kwargs",
            f"{{'filename': '{str(results_path)}'}}",
            "--opt-kwargs",
            f"{{'restart': '{str(restart_path)}'}}",
            "--steps",
            2,
        ],
    )
    assert result.exit_code == 0
    atoms = read(results_path)
    final_energy = atoms.get_potential_energy()
    assert final_energy < intermediate_energy
