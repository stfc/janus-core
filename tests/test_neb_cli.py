"""Test neb commandline interface."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import read
import pytest
from typer.testing import CliRunner
import yaml

from janus_core.cli.janus import app
from tests.utils import assert_log_contains, clear_log_handlers, strip_ansi_codes

DATA_PATH = Path(__file__).parent / "data"

runner = CliRunner()


def test_help():
    """Test calling `janus neb --help`."""
    result = runner.invoke(app, ["neb", "--help"])
    assert result.exit_code == 0
    assert "Usage: janus neb [OPTIONS]" in strip_ansi_codes(result.stdout)


def test_neb():
    """Test calculating force constants and band structure."""
    results_path = Path("./LiFePO4_start-neb-results.dat").absolute()
    band_path = Path("./LiFePO4_start-neb-band.extxyz").absolute()
    plot_path = Path("./LiFePO4_start-neb-plot.svg").absolute()
    log_path = Path("./LiFePO4_start-neb-log.yml").absolute()
    summary_path = Path("./LiFePO4_start-neb-summary.yml").absolute()

    assert not results_path.exists()
    assert not band_path.exists()
    assert not plot_path.exists()
    assert not log_path.exists()
    assert not summary_path.exists()

    try:
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
                "--plot-band",
                "--write-band",
            ],
        )
        assert result.exit_code == 0

        assert results_path.exists()
        assert log_path.exists()
        assert summary_path.exists()

        # Read NEB summary file
        with open(summary_path, encoding="utf8") as file:
            neb_summary = yaml.safe_load(file)

        # Check contents of results file
        with open(results_path, encoding="utf8") as results_file:
            lines = results_file.readlines()

        assert len(lines) == 2
        assert lines[0] == "#Barrier [eV] | delta E [eV] | Max force [eV/Å] \n"
        results = [float(result) for result in lines[1].split()]
        assert results == pytest.approx(
            [0.8984807983308647, 5.634287845168728e-07, 4.802233744475505]
        )
        assert "command" in neb_summary
        assert "janus neb" in neb_summary["command"]
        assert "start_time" in neb_summary
        assert "inputs" in neb_summary
        assert "end_time" in neb_summary

        assert "emissions" in neb_summary
        assert neb_summary["emissions"] > 0

        band = read(band_path, index=":")
        assert isinstance(band[0], Atoms)
        assert len(band) == 7

    finally:
        results_path.unlink(missing_ok=True)
        band_path.unlink(missing_ok=True)
        plot_path.unlink(missing_ok=True)
        log_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
        clear_log_handlers()


def test_minimize(tmp_path):
    """Test minimizing structures before interpolation and optimization."""
    file_prefix = tmp_path / "LFPO"
    results_path = tmp_path / "LFPO-neb-results.dat"
    summary_path = tmp_path / "LFPO-neb-summary.yml"
    log_path = tmp_path / "LFPO-neb-log.yml"

    result = runner.invoke(
        app,
        [
            "neb",
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
            "{'fmax': 1.0}",
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

    # Check contents of results file
    with open(results_path, encoding="utf8") as results_file:
        lines = results_file.readlines()

    assert len(lines) == 2
    assert lines[0] == "#Barrier [eV] | delta E [eV] | Max force [eV/Å] \n"
    results = [float(result) for result in lines[1].split()]
    assert results == pytest.approx(
        [0.8296793283384434, 5.145591330801835e-07, 2.174366063099385]
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
            "--band-structs",
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
            "--init-struct",
            DATA_PATH / "LiFePO4_start.cif",
            "--final-struct",
            DATA_PATH / "LiFePO4_end.cif",
            "--band-structs",
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
        [336131.8396612201, -3.792642177184348, 59797105.48085273]
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
        [1.59760263850103, 3.3379069463990163e-07, 6.3809101701189075]
    )


def test_invalid_neb(tmp_path):
    """Test passing invalid neb_class."""
    file_prefix = tmp_path / "LFPO"

    result = runner.invoke(
        app,
        [
            "neb",
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
