"""Test neb commandline interface."""

from __future__ import annotations

from pathlib import Path
import shutil

from ase import Atoms
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
    """Test calling `janus neb --help`."""
    result = runner.invoke(app, ["neb", "--help"])
    assert result.exit_code == 0
    assert "Usage: janus neb [OPTIONS]" in strip_ansi_codes(result.stdout)


def test_neb():
    """Test calculating force constants and band structure."""
    results_dir = Path("./janus_results")
    results_path = results_dir / "LiFePO4_start-neb-results.dat"
    band_path = results_dir / "LiFePO4_start-neb-band.extxyz"
    plot_path = results_dir / "LiFePO4_start-neb-plot.svg"
    log_path = results_dir / "LiFePO4_start-neb-log.yml"
    summary_path = results_dir / "LiFePO4_start-neb-summary.yml"

    assert not results_dir.exists()

    try:
        result = runner.invoke(
            app,
            [
                "neb",
                "--arch",
                "mace_mp",
                "--init-struct",
                DATA_PATH / "LiFePO4_start.cif",
                "--final-struct",
                DATA_PATH / "LiFePO4_end.cif",
                "--interpolator",
                "pymatgen",
                "--fmax",
                5,
                "--n-images",
                5,
                "--plot-band",
                "--write-band",
            ],
        )
        assert result.exit_code == 0

        assert results_path.exists()
        assert band_path.exists()
        assert plot_path.exists()
        assert log_path.exists()
        assert summary_path.exists()

        # Check contents of results file
        with open(results_path, encoding="utf8") as results_file:
            lines = results_file.readlines()

        assert len(lines) == 2
        assert lines[0] == "#Barrier [eV] | delta E [eV] | Max force [eV/Å] \n"
        results = [float(result) for result in lines[1].split()]
        assert results == pytest.approx(
            [0.8497755543465928, -3.0149328722473e-07, 4.802233744475505]
        )

        # Read NEB summary file
        with open(summary_path, encoding="utf8") as file:
            neb_summary = yaml.safe_load(file)

        assert "command" in neb_summary
        assert "janus neb" in neb_summary["command"]
        assert "start_time" in neb_summary
        assert "config" in neb_summary
        assert "info" in neb_summary
        assert "end_time" in neb_summary

        assert "emissions" in neb_summary
        assert neb_summary["emissions"] > 0

        output_files = {
            "results": results_path,
            "plot": plot_path,
            "band": band_path,
            "minimized_initial_structure": None,
            "minimized_final_structure": None,
            "log": log_path,
            "summary": summary_path,
        }
        check_output_files(summary=neb_summary, output_files=output_files)

        band = read(band_path, index=":")
        assert isinstance(band[0], Atoms)
        assert len(band) == 7

    finally:
        shutil.rmtree(results_dir, ignore_errors=True)
        clear_log_handlers()


def test_minimize(tmp_path):
    """Test minimizing structures before interpolation and optimization."""
    file_prefix = tmp_path / "LFPO"
    results_path = tmp_path / "LFPO-neb-results.dat"
    min_init_path = tmp_path / "LFPO-init-opt.extxyz"
    min_final_path = tmp_path / "LFPO-final-opt.extxyz"
    summary_path = tmp_path / "LFPO-neb-summary.yml"
    log_path = tmp_path / "LFPO-neb-log.yml"

    result = runner.invoke(
        app,
        [
            "neb",
            "--arch",
            "mace_mp",
            "--init-struct",
            DATA_PATH / "LiFePO4_start.cif",
            "--final-struct",
            DATA_PATH / "LiFePO4_end.cif",
            "--fmax",
            4,
            "--n-images",
            5,
            "--interpolator",
            "pymatgen",
            "--minimize",
            "--minimize-kwargs",
            "{'fmax': 1.0, 'write_results': True}",
            "--no-tracker",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 0

    # Read NEB summary file
    with open(summary_path, encoding="utf8") as file:
        neb_summary = yaml.safe_load(file)
        assert "emissions" not in neb_summary

    output_files = {
        "results": results_path,
        "plot": None,
        "band": None,
        "minimized_initial_structure": min_init_path,
        "minimized_final_structure": min_final_path,
        "log": log_path,
        "summary": summary_path,
    }
    check_output_files(summary=neb_summary, output_files=output_files)

    # Check contents of results file
    with open(results_path, encoding="utf8") as results_file:
        lines = results_file.readlines()

    assert len(lines) == 2
    assert lines[0] == "#Barrier [eV] | delta E [eV] | Max force [eV/Å] \n"
    results = [float(result) for result in lines[1].split()]
    assert results == pytest.approx(
        [0.8551249637134779, -1.4654790447821142e-07, 2.174366063099385]
    )

    assert_log_contains(
        log_path,
        includes=[
            "Starting geometry optimization",
            "Max force: 0.77",
            "Using pymatgen interpolator",
        ],
    )


def test_bands(tmp_path):
    """Test using band that has already been generated."""
    file_prefix = tmp_path / "LFPO"
    log_path = tmp_path / "LFPO-neb-log.yml"

    result = runner.invoke(
        app,
        [
            "neb",
            "--arch",
            "mace_mp",
            "--neb-structs",
            DATA_PATH / "LiFePO4-neb-band.xyz",
            "--steps",
            2,
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 0
    assert_log_contains(
        log_path, includes=["Skipping interpolation", "Optimization steps: 1"]
    )


def test_invalid_interpolator(tmp_path):
    """Test passing invalid interpolator."""
    file_prefix = tmp_path / "LFPO"

    result = runner.invoke(
        app,
        [
            "neb",
            "--arch",
            "mace_mp",
            "--init-struct",
            DATA_PATH / "LiFePO4_start.cif",
            "--final-struct",
            DATA_PATH / "LiFePO4_end.cif",
            "--interpolator",
            "test",
            "--no-tracker",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 2
    assert "'test' is not one of 'ase', 'pymatgen'." in result.stdout


def test_interpolate_and_band(tmp_path):
    """Test passing initial/final structures and band."""
    file_prefix = tmp_path / "LFPO"

    result = runner.invoke(
        app,
        [
            "neb",
            "--arch",
            "mace_mp",
            "--init-struct",
            DATA_PATH / "LiFePO4_start.cif",
            "--final-struct",
            DATA_PATH / "LiFePO4_end.cif",
            "--neb-structs",
            DATA_PATH / "LiFePO4-neb-band.xyz",
            "--no-tracker",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)


def test_invalid_n_images(tmp_path):
    """Test invalid number of images."""
    file_prefix = tmp_path / "LFPO"

    result = runner.invoke(
        app,
        [
            "neb",
            "--arch",
            "mace_mp",
            "--init-struct",
            DATA_PATH / "LiFePO4_start.cif",
            "--final-struct",
            DATA_PATH / "LiFePO4_end.cif",
            "--n-images",
            0,
            "--no-tracker",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)


def test_neb_class(tmp_path):
    """Test passing neb_class and invalid neb_kwargs."""
    file_prefix = tmp_path / "LFPO"
    log_path = tmp_path / "LFPO-neb-log.yml"

    result = runner.invoke(
        app,
        [
            "neb",
            "--arch",
            "mace_mp",
            "--init-struct",
            DATA_PATH / "LiFePO4_start.cif",
            "--final-struct",
            DATA_PATH / "LiFePO4_end.cif",
            "--steps",
            1,
            "--n-images",
            5,
            "--interpolator",
            "pymatgen",
            "--neb-class",
            "DyNEB",
            "--neb-kwargs",
            "{'dynamic_relaxation': False, 'scale_fmax': 1.0}",  # Invalid combination
            "--no-tracker",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    assert_log_contains(log_path, includes="Using NEB class: DyNEB")


def test_interpolator(tmp_path):
    """Test passing interpolator_kwargs."""
    file_prefix = tmp_path / "LFPO"
    results_path = tmp_path / "LFPO-neb-results.dat"

    result = runner.invoke(
        app,
        [
            "neb",
            "--arch",
            "mace_mp",
            "--init-struct",
            DATA_PATH / "LiFePO4_start.cif",
            "--final-struct",
            DATA_PATH / "LiFePO4_end.cif",
            "--steps",
            1,
            "--n-images",
            5,
            "--interpolator",
            "pymatgen",
            "--interpolator-kwargs",
            "{'autosort_tol': 0.0}",
            "--no-tracker",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 0

    # Check contents of results file
    with open(results_path, encoding="utf8") as results_file:
        lines = results_file.readlines()
    results = [float(result) for result in lines[1].split()]
    assert results == pytest.approx(
        [336164.9211653024, -3.0149328722473e-07, 59797105.48085273]
    )


def test_optimzer(tmp_path):
    """Test passing optimizer and optimizer_kwargs."""
    file_prefix = tmp_path / "LFPO"
    results_path = tmp_path / "LFPO-neb-results.dat"
    log_path = tmp_path / "LFPO-neb-log.yml"

    result = runner.invoke(
        app,
        [
            "neb",
            "--arch",
            "mace_mp",
            "--init-struct",
            DATA_PATH / "LiFePO4_start.cif",
            "--final-struct",
            DATA_PATH / "LiFePO4_end.cif",
            "--steps",
            0,
            "--n-images",
            5,
            "--interpolator",
            "pymatgen",
            "--optimizer",
            "FIRE",
            "--no-tracker",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 0
    assert_log_contains(log_path, includes="Using optimizer: FIRE")

    # Check contents of results file
    with open(results_path, encoding="utf8") as results_file:
        lines = results_file.readlines()
    results = [float(result) for result in lines[1].split()]
    assert results == pytest.approx(
        [1.705643002186438, -3.0149328722473e-07, 6.3809101701189075]
    )


def test_invalid_neb(tmp_path):
    """Test passing invalid neb_class."""
    file_prefix = tmp_path / "LFPO"

    result = runner.invoke(
        app,
        [
            "neb",
            "--arch",
            "mace_mp",
            "--init-struct",
            DATA_PATH / "LiFePO4_start.cif",
            "--final-struct",
            DATA_PATH / "LiFePO4_end.cif",
            "--steps",
            1,
            "--n-images",
            5,
            "--interpolator",
            "pymatgen",
            "--neb-class",
            "neb_class",
            "--no-tracker",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, AttributeError)


def test_invalid_opt(tmp_path):
    """Test passing invalid optimizer."""
    file_prefix = tmp_path / "LFPO"

    result = runner.invoke(
        app,
        [
            "neb",
            "--arch",
            "mace_mp",
            "--init-struct",
            DATA_PATH / "LiFePO4_start.cif",
            "--final-struct",
            DATA_PATH / "LiFePO4_end.cif",
            "--steps",
            1,
            "--n-images",
            5,
            "--interpolator",
            "pymatgen",
            "--optimizer",
            "optimizer",
            "--no-tracker",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, AttributeError)


def test_model(tmp_path):
    """Test model passed correctly."""
    file_prefix = tmp_path / "NaCl"
    results_path = tmp_path / "NaCl-neb-band.extxyz"
    log_path = tmp_path / "test.log"

    result = runner.invoke(
        app,
        [
            "neb",
            "--init-struct",
            DATA_PATH / "LiFePO4_start.cif",
            "--final-struct",
            DATA_PATH / "LiFePO4_end.cif",
            "--interpolator",
            "pymatgen",
            "--fmax",
            5,
            "--n-images",
            5,
            "--write-band",
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
    results_path = tmp_path / "NaCl-neb-band.extxyz"
    log_path = tmp_path / "test.log"

    result = runner.invoke(
        app,
        [
            "neb",
            "--init-struct",
            DATA_PATH / "LiFePO4_start.cif",
            "--final-struct",
            DATA_PATH / "LiFePO4_end.cif",
            "--interpolator",
            "pymatgen",
            "--fmax",
            5,
            "--n-images",
            5,
            "--write-band",
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
            "neb",
            "--init-struct",
            DATA_PATH / "LiFePO4_start.cif",
            "--final-struct",
            DATA_PATH / "LiFePO4_end.cif",
            "--interpolator",
            "pymatgen",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 2
    assert "Missing option" in result.stdout


def test_info(tmp_path):
    """Test info written to output structures."""
    file_prefix = tmp_path / "LiFePO4"
    neb_path = tmp_path / "LiFePO4-neb-band.extxyz"

    result = runner.invoke(
        app,
        [
            "neb",
            "--init-struct",
            DATA_PATH / "LiFePO4_start.cif",
            "--final-struct",
            DATA_PATH / "LiFePO4_end.cif",
            "--interpolator",
            "pymatgen",
            "--fmax",
            5,
            "--n-images",
            5,
            "--write-band",
            "--arch",
            "mace_mp",
            "--model-path",
            MACE_PATH,
            "--file-prefix",
            file_prefix,
            "--no-tracker",
        ],
    )
    assert result.exit_code == 0

    assert neb_path.exists()

    atoms = read(neb_path, index=":")
    for struct in atoms:
        assert "system_name" in struct.info
        assert struct.info["system_name"] == "LiFePO4_start"
        assert "config_type" in struct.info
        assert struct.info["config_type"] == "neb"
