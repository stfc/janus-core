"""Test descriptors commandline interface."""

from pathlib import Path

from ase.io import read
import pytest
from typer.testing import CliRunner
import yaml

from janus_core.cli.janus import app
from tests.utils import assert_log_contains, strip_ansi_codes

DATA_PATH = Path(__file__).parent / "data"

runner = CliRunner()


def test_help():
    """Test calling `janus descriptors --help`."""
    result = runner.invoke(app, ["descriptors", "--help"])
    assert result.exit_code == 0
    assert "Usage: janus descriptors [OPTIONS]" in strip_ansi_codes(result.stdout)


def test_descriptors(tmp_path):
    """Test calculating MLIP descriptors."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    out_path = tmp_path / "NaCl-descriptors.extxyz"
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
    assert "mace_mp_descriptor" in atoms.info
    assert "mace_mp_Na_descriptor" not in atoms.info
    assert "mace_mp_Cl_descriptor" not in atoms.info
    assert "mace_mp_descriptors" not in atoms.arrays

    # Check only initial structure is minimized
    assert_log_contains(
        log_path,
        includes=[
            "Starting descriptors calculation",
            "invariants_only: True",
            "calc_per_element: False",
        ],
    )

    # Read descriptors summary file
    assert summary_path.exists()
    with open(summary_path, encoding="utf8") as file:
        descriptors_summary = yaml.safe_load(file)

    assert "command" in descriptors_summary
    assert "janus descriptors" in descriptors_summary["command"]
    assert "start_time" in descriptors_summary
    assert "inputs" in descriptors_summary
    assert "end_time" in descriptors_summary

    assert "emissions" in descriptors_summary
    assert descriptors_summary["emissions"] > 0


def test_calc_per_element(tmp_path):
    """Test calculating MLIP descriptors for each element."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    out_path = tmp_path / "NaCl-descriptors.extxyz"
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
    assert "mace_mp_Na_descriptor" in atoms.info
    assert "mace_mp_Cl_descriptor" in atoms.info

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
    out_path = tmp_path / "NaCl-descriptors.extxyz"
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
    assert "mace_mp_descriptor" in atoms.info
    assert "mace_mp_Na_descriptor" not in atoms.info
    assert "mace_mp_Cl_descriptor" not in atoms.info

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
    out_path = tmp_path / "benzene-descriptors.extxyz"
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
    assert "mace_mp_descriptor" in atoms[0].info
    assert "mace_mp_descriptor" in atoms[1].info


def test_per_atom(tmp_path):
    """Test calculating descriptors for each atom."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    out_path = tmp_path / "NaCl-descriptors.extxyz"
    result = runner.invoke(
        app,
        [
            "descriptors",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--out",
            out_path,
            "--calc-per-atom",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    assert out_path.exists()

    atoms = read(out_path)
    assert "mace_mp_descriptor" in atoms.info
    assert "mace_mp_descriptors" in atoms.arrays
    assert len(atoms.arrays["mace_mp_descriptors"]) == 8
    assert atoms.arrays["mace_mp_descriptors"][0] == pytest.approx(-0.00203750)
