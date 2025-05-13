"""Test eos commandline interface."""

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
    """Test calling `janus eos --help`."""
    result = runner.invoke(app, ["eos", "--help"])
    assert result.exit_code == 0
    assert "Usage: janus eos [OPTIONS]" in strip_ansi_codes(result.stdout)


def test_eos():
    """Test calculating the equation of state."""
    results_dir = Path("./janus_results")
    eos_raw_path = results_dir / "NaCl-eos-raw.dat"
    eos_fit_path = results_dir / "NaCl-eos-fit.dat"
    log_path = results_dir / "NaCl-eos-log.yml"
    summary_path = results_dir / "NaCl-eos-summary.yml"

    assert not results_dir.exists()

    try:
        result = runner.invoke(
            app,
            [
                "eos",
                "--struct",
                DATA_PATH / "NaCl.cif",
                "--arch",
                "mace_mp",
            ],
        )
        assert result.exit_code == 0

        assert eos_raw_path.exists()
        assert eos_fit_path.exists()
        assert log_path.exists()
        assert summary_path.exists()

        # Check contents of raw data file
        with open(eos_raw_path, encoding="utf8") as eos_raw_file:
            lines = eos_raw_file.readlines()

        assert len(lines) == 8
        assert lines[0] == "#Lattice Scalar | Energy [eV] | Volume [Å^3] \n"
        assert lines[4].split()[0] == "1.0"
        assert float(lines[4].split()[1]) == pytest.approx(-27.046359959669214)
        assert float(lines[4].split()[2]) == pytest.approx(184.05884033013012)

        # Check contents of fitted data file
        with open(eos_fit_path, encoding="utf8") as eos_fit_file:
            lines = eos_fit_file.readlines()

        assert len(lines) == 2
        assert lines[0] == "#Bulk modulus [GPa] | Energy [eV] | Volume [Å^3] \n"
        assert float(lines[1].split()[0]) == pytest.approx(27.186555689697165)
        assert float(lines[1].split()[1]) == pytest.approx(-27.046361904823204)
        assert float(lines[1].split()[2]) == pytest.approx(184.22281215770133)

        # Check only initial structure is minimized
        assert_log_contains(
            log_path,
            includes=["Minimising initial structure"],
            excludes=["Minimising lattice scalar = 1.0"],
        )

        # Read eos summary file
        with open(summary_path, encoding="utf8") as file:
            eos_summary = yaml.safe_load(file)

        assert "command" in eos_summary
        assert "janus eos" in eos_summary["command"]
        assert "start_time" in eos_summary
        assert "config" in eos_summary
        assert "info" in eos_summary
        assert "end_time" in eos_summary

        assert "emissions" in eos_summary
        assert eos_summary["emissions"] > 0

        output_files = {
            "raw": eos_raw_path,
            "plot": None,
            "generated_structures": None,
            "fit": eos_fit_path,
            "log": log_path,
            "summary": summary_path,
        }
        check_output_files(eos_summary, output_files)

    finally:
        shutil.rmtree(results_dir, ignore_errors=True)
        clear_log_handlers()


def test_setting_lattice(tmp_path):
    """Test setting the lattice constants."""
    file_prefix = tmp_path / "example"
    eos_raw_path = tmp_path / "example-eos-raw.dat"

    result = runner.invoke(
        app,
        [
            "eos",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--min-volume",
            0.8,
            "--max-volume",
            1.2,
            "--n-volumes",
            5,
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 0
    assert eos_raw_path.exists()

    # Check contents of raw data file
    with open(eos_raw_path, encoding="utf8") as eos_raw_file:
        lines = eos_raw_file.readlines()
    assert len(lines) == 6
    assert lines[3].split()[0] == "1.0"
    assert float(lines[1].split()[0]) == pytest.approx(0.8 ** (1 / 3))
    assert float(lines[5].split()[0]) == pytest.approx(1.2 ** (1 / 3))


@pytest.mark.parametrize(
    "option, value", [("--min-volume", 1), ("--max-volume", 0.9), ("--n-volumes", 0)]
)
def test_invalid_lattice(option, value, tmp_path):
    """Test setting the invalid lattice constants."""
    file_prefix = tmp_path / "NaCl"

    result = runner.invoke(
        app,
        [
            "eos",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            option,
            value,
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)


def test_minimising_all(tmp_path):
    """Test minimising structures with different lattice constants."""
    file_prefix = tmp_path / "NaCl"
    log_path = tmp_path / "NaCl-eos-log.yml"

    result = runner.invoke(
        app,
        [
            "eos",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--n-volumes",
            4,
            "--minimize-all",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 0

    # Check minimizes multiple structures
    assert_log_contains(
        log_path,
        includes=[
            "Minimising initial structure",
            "Minimising lattice scalar = 1.0",
            "constant_volume: True",
        ],
    )


def test_writing_structs(tmp_path):
    """Test writing out generated structures."""
    file_prefix = tmp_path / "test" / "example"
    raw_path = tmp_path / "test" / "example-eos-raw.dat"
    fit_path = tmp_path / "test" / "example-eos-fit.dat"
    generated_path = tmp_path / "test" / "example-generated.extxyz"
    log_path = tmp_path / "test" / "example-eos-log.yml"
    summary_path = tmp_path / "test" / "example-eos-summary.yml"

    result = runner.invoke(
        app,
        [
            "eos",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--n-volumes",
            4,
            "--file-prefix",
            file_prefix,
            "--write-structures",
        ],
    )
    assert result.exit_code == 0
    assert generated_path.exists()
    atoms = read(generated_path, index=":")
    assert len(atoms) == 5

    # Read eos summary file
    with open(summary_path, encoding="utf8") as file:
        eos_summary = yaml.safe_load(file)

    output_files = {
        "raw": raw_path,
        "plot": None,
        "generated_structures": generated_path,
        "fit": fit_path,
        "log": log_path,
        "summary": summary_path,
    }
    check_output_files(eos_summary, output_files)


def test_error_write_geomopt(tmp_path):
    """Test an error is raised if trying to write via geomopt."""
    file_prefix = tmp_path / "example"

    minimize_kwargs = "{'write_results': True}"

    result = runner.invoke(
        app,
        [
            "eos",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--n-volumes",
            4,
            "--file-prefix",
            file_prefix,
            "--minimize",
            "--minimize-kwargs",
            minimize_kwargs,
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)


@pytest.mark.parametrize("read_kwargs", ["{'index': 1}", "{}"])
def test_valid_traj_input(read_kwargs, tmp_path):
    """Test valid trajectory input structure handled."""
    file_prefix = tmp_path / "traj"
    eos_raw_path = tmp_path / "traj-eos-raw.dat"
    eos_fit_path = tmp_path / "traj-eos-fit.dat"

    result = runner.invoke(
        app,
        [
            "eos",
            "--struct",
            DATA_PATH / "NaCl-traj.xyz",
            "--arch",
            "mace_mp",
            "--read-kwargs",
            read_kwargs,
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 0
    assert eos_raw_path.exists()
    assert eos_fit_path.exists()


def test_invalid_traj_input(tmp_path):
    """Test invalid trajectory input structure handled."""
    file_prefix = tmp_path / "traj"

    result = runner.invoke(
        app,
        [
            "eos",
            "--struct",
            DATA_PATH / "NaCl-traj.xyz",
            "--arch",
            "mace_mp",
            "--read-kwargs",
            "{'index': ':'}",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)


def test_plot(tmp_path):
    """Test plotting equation of state."""
    file_prefix = tmp_path / "NaCl"
    raw_path = tmp_path / "NaCl-eos-raw.dat"
    fit_path = tmp_path / "NaCl-eos-fit.dat"
    plot_path = tmp_path / "NaCl-eos-plot.svg"
    log_path = tmp_path / "NaCl-eos-log.yml"
    summary_path = tmp_path / "NaCl-eos-summary.yml"

    result = runner.invoke(
        app,
        [
            "eos",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--n-volumes",
            4,
            "--plot-to-file",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 0

    # Read eos summary file
    with open(summary_path, encoding="utf8") as file:
        eos_summary = yaml.safe_load(file)

    output_files = {
        "raw": raw_path,
        "plot": plot_path,
        "generated_structures": None,
        "fit": fit_path,
        "log": log_path,
        "summary": summary_path,
    }
    check_output_files(eos_summary, output_files)


def test_no_carbon(tmp_path):
    """Test disabling carbon tracking."""
    file_prefix = tmp_path / "NaCl"
    summary_path = tmp_path / "NaCl-eos-summary.yml"

    result = runner.invoke(
        app,
        [
            "eos",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--n-volumes",
            4,
            "--no-tracker",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 0

    # Read eos summary file
    with open(summary_path, encoding="utf8") as file:
        eos_summary = yaml.safe_load(file)
    assert "emissions" not in eos_summary


def test_model(tmp_path):
    """Test model passed correctly."""
    file_prefix = tmp_path / "NaCl"
    generated_path = tmp_path / "NaCl-generated.extxyz"
    log_path = tmp_path / "test.log"

    result = runner.invoke(
        app,
        [
            "eos",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--model",
            MACE_PATH,
            "--n-volumes",
            5,
            "--no-minimize",
            "--log",
            log_path,
            "--file-prefix",
            file_prefix,
            "--no-tracker",
            "--write-structures",
        ],
    )
    assert result.exit_code == 0

    assert_log_contains(
        log_path,
        excludes=[
            "Minimising initial structure",
            "FutureWarning: `model_path` has been deprecated.",
        ],
    )

    atoms = read(generated_path)
    assert "model" in atoms.info
    assert atoms.info["model"] == str(MACE_PATH.as_posix())


def test_model_path_deprecated(tmp_path):
    """Test model_path sets model."""
    file_prefix = tmp_path / "NaCl"
    generated_path = tmp_path / "NaCl-generated.extxyz"
    log_path = tmp_path / "test.log"

    result = runner.invoke(
        app,
        [
            "eos",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--model-path",
            MACE_PATH,
            "--n-volumes",
            5,
            "--no-minimize",
            "--log",
            log_path,
            "--file-prefix",
            file_prefix,
            "--no-tracker",
            "--write-structures",
        ],
    )
    assert result.exit_code == 0

    atoms = read(generated_path)
    assert "model" in atoms.info
    assert atoms.info["model"] == str(MACE_PATH.as_posix())


def test_missing_arch(tmp_path):
    """Test no architecture specified."""
    file_prefix = tmp_path / "NaCl"

    result = runner.invoke(
        app,
        [
            "eos",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 2
    assert "Missing option" in result.stdout


def test_info(tmp_path):
    """Test info written to generated structures."""
    file_prefix = tmp_path / "NaCl"
    generated_path = tmp_path / "NaCl-generated.extxyz"

    result = runner.invoke(
        app,
        [
            "eos",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--n-volumes",
            4,
            "--file-prefix",
            file_prefix,
            "--write-structures",
            "--no-minimize",
            "--no-tracker",
        ],
    )
    assert result.exit_code == 0

    assert generated_path.exists()

    atoms = read(generated_path, index=":")
    for struct in atoms:
        assert "system_name" in struct.info
        assert struct.info["system_name"] == "NaCl"
        assert "config_type" in struct.info
        assert struct.info["config_type"] == "eos"


def test_info_min(tmp_path):
    """Test info written to generated structures after minimisation."""
    file_prefix = tmp_path / "NaCl"
    generated_path = tmp_path / "NaCl-generated.extxyz"

    result = runner.invoke(
        app,
        [
            "eos",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--n-volumes",
            4,
            "--file-prefix",
            file_prefix,
            "--write-structures",
            "--no-tracker",
        ],
    )
    assert result.exit_code == 0

    assert generated_path.exists()

    atoms = read(generated_path, index=":")
    for struct in atoms:
        assert "system_name" in struct.info
        assert struct.info["system_name"] == "NaCl"
        assert "config_type" in struct.info
        assert struct.info["config_type"] == "geom_opt"
