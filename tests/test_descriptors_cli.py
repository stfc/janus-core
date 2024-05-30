"""Test descriptors commandline interface."""

from pathlib import Path

from ase.io import read
from typer.testing import CliRunner

from janus_core.cli.janus import app
from tests.utils import assert_log_contains

DATA_PATH = Path(__file__).parent / "data"

runner = CliRunner()


def test_help():
    """Test calling `janus descriptors --help`."""
    result = runner.invoke(app, ["descriptors", "--help"])
    assert result.exit_code == 0
    assert "Usage: janus descriptors [OPTIONS]" in result.stdout


def test_descriptors(tmp_path):
    """Test calculating MLIP descriptors."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    out_path = tmp_path / "NaCl-descriptors.xyz"
    result = runner.invoke(
        app,
        [
            "descriptors",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--out",
            out_path,
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    assert out_path.exists()

    atoms = read(out_path)
    assert "descriptor" in atoms.info
    assert "Na_descriptor" not in atoms.info
    assert "Cl_descriptor" not in atoms.info

    # Check only initial structure is minimized
    assert_log_contains(
        log_path,
        includes=[
            "Starting descriptors calculation",
            "invariants_only: True",
            "calc_per_element: False",
        ],
    )


def test_calc_per_element(tmp_path):
    """Test calculating MLIP descriptors for each element."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    out_path = tmp_path / "NaCl-descriptors.xyz"
    result = runner.invoke(
        app,
        [
            "descriptors",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--out",
            out_path,
            "--calc-per-element",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    assert out_path.exists()

    atoms = read(out_path)
    assert "descriptor" in atoms.info
    assert "Na_descriptor" in atoms.info
    assert "Cl_descriptor" in atoms.info

    # Check only initial structure is minimized
    assert_log_contains(
        log_path,
        includes=[
            "Starting descriptors calculation",
            "invariants_only: True",
            "calc_per_element: True",
        ],
    )


def test_invariant(tmp_path):
    """Test setting invariant_only to false."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    out_path = tmp_path / "NaCl-descriptors.xyz"
    result = runner.invoke(
        app,
        [
            "descriptors",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--out",
            out_path,
            "--no-invariants-only",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    assert out_path.exists()

    atoms = read(out_path)
    assert "descriptor" in atoms.info
    assert "Na_descriptor" not in atoms.info
    assert "Cl_descriptor" not in atoms.info

    # Check only initial structure is minimized
    assert_log_contains(
        log_path,
        includes=[
            "Starting descriptors calculation",
            "invariants_only: False",
            "calc_per_element: False",
        ],
    )


def test_traj(tmp_path):
    """Test calculating descriptors for a trajectory."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    out_path = tmp_path / "benzene-descriptors.xyz"
    result = runner.invoke(
        app,
        [
            "descriptors",
            "--struct",
            DATA_PATH / "benzene-traj.xyz",
            "--out",
            out_path,
            "--read-kwargs",
            "{'index' : ':'}",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    assert out_path.exists()

    atoms = read(out_path, index=":")
    assert len(atoms) == 2
    assert "descriptor" in atoms[0].info
    assert "descriptor" in atoms[1].info
