"""Test descriptors commandline interface."""

from __future__ import annotations

from pathlib import Path
import shutil

from ase.io import read
import pytest
from typer.testing import CliRunner
import yaml

from janus_core.cli.janus import app
from tests.utils import (
    assert_log_contains,
    check_output_files,
    clear_log_handlers,
    strip_ansi_codes,
)

DATA_PATH = Path(__file__).parent / "data"
MACE_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"

runner = CliRunner()


def test_help():
    """Test calling `janus descriptors --help`."""
    result = runner.invoke(app, ["descriptors", "--help"])
    assert result.exit_code == 0
    assert "Usage: janus descriptors [OPTIONS]" in strip_ansi_codes(result.stdout)


def test_descriptors():
    """Test calculating MLIP descriptors."""
    results_dir = Path("./janus_results")
    out_path = results_dir / "NaCl-descriptors.extxyz"
    log_path = results_dir / "NaCl-descriptors-log.yml"
    summary_path = results_dir / "NaCl-descriptors-summary.yml"

    assert not results_dir.exists()

    try:
        result = runner.invoke(
            app,
            [
                "descriptors",
                "--struct",
                DATA_PATH / "NaCl.cif",
            ],
        )
        assert result.exit_code == 0

        assert out_path.exists()
        assert log_path.exists()
        assert summary_path.exists()

        atoms = read(out_path)
        assert "mace_mp_descriptor" in atoms.info
        assert "mace_mp_Na_descriptor" not in atoms.info
        assert "mace_mp_Cl_descriptor" not in atoms.info
        assert "mace_mp_descriptors" not in atoms.arrays

        assert "system_name" in atoms.info
        assert atoms.info["system_name"] == "NaCl"

        # Check only initial structure is minimized
        assert_log_contains(
            log_path,
            includes=[
                "Starting descriptors calculation",
                "invariants_only: True",
                "calc_per_element: False",
                "calc_per_atom: False",
            ],
        )

        # Read descriptors summary file
        assert summary_path.exists()
        with open(summary_path, encoding="utf8") as file:
            descriptors_summary = yaml.safe_load(file)

        assert "command" in descriptors_summary
        assert "janus descriptors" in descriptors_summary["command"]
        assert "start_time" in descriptors_summary
        assert "config" in descriptors_summary
        assert "info" in descriptors_summary
        assert "end_time" in descriptors_summary

        assert "emissions" in descriptors_summary
        assert descriptors_summary["emissions"] > 0

        output_files = {
            "results": out_path,
            "log": log_path,
            "summary": summary_path,
        }
        check_output_files(descriptors_summary, output_files)

    finally:
        shutil.rmtree(results_dir, ignore_errors=True)
        clear_log_handlers()


def test_calc_per_element(tmp_path):
    """Test calculating MLIP descriptors for each element."""
    out_path = tmp_path / "test" / "NaCl-descriptors.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

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
            "calc_per_atom: False",
        ],
    )


def test_invariant(tmp_path):
    """Test setting invariant_only to false."""
    out_path = tmp_path / "NaCl-descriptors.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

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
    out_path = tmp_path / "benzene-descriptors.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

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
    out_path = tmp_path / "NaCl-descriptors.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

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

    # Check only initial structure is minimized
    assert_log_contains(
        log_path,
        includes=[
            "Starting descriptors calculation",
            "invariants_only: True",
            "calc_per_element: False",
            "calc_per_atom: True",
        ],
    )


def test_no_carbon(tmp_path):
    """Test disabling carbon tracking."""
    out_path = tmp_path / "test" / "NaCl-descriptors.extxyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

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
            "--no-tracker",
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0

    # Read descriptors summary file
    with open(summary_path, encoding="utf8") as file:
        descriptors_summary = yaml.safe_load(file)
    assert "emissions" not in descriptors_summary


def test_file_prefix(tmp_path):
    """Test file prefix creates directories and affects all files."""
    file_prefix = tmp_path / "test/test"
    result = runner.invoke(
        app,
        [
            "descriptors",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 0
    test_path = tmp_path / "test"
    assert list(tmp_path.iterdir()) == [test_path]
    assert set(test_path.iterdir()) == {
        test_path / "test-descriptors.extxyz",
        test_path / "test-descriptors-summary.yml",
        test_path / "test-descriptors-log.yml",
    }


def test_model(tmp_path):
    """Test model passed correctly."""
    file_prefix = tmp_path / "NaCl"
    results_path = tmp_path / "NaCl-descriptors.extxyz"
    log_path = tmp_path / "test.log"

    result = runner.invoke(
        app,
        [
            "descriptors",
            "--struct",
            DATA_PATH / "NaCl.cif",
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
    assert atoms.info["model"] == str(MACE_PATH)


def test_model_path_deprecated(tmp_path):
    """Test model_path sets model."""
    file_prefix = tmp_path / "NaCl"
    results_path = tmp_path / "NaCl-descriptors.extxyz"
    log_path = tmp_path / "test.log"

    result = runner.invoke(
        app,
        [
            "descriptors",
            "--struct",
            DATA_PATH / "NaCl.cif",
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
    assert atoms.info["model"] == str(MACE_PATH)
