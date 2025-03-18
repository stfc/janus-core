"""Test phonons commandline interface."""

from __future__ import annotations

import lzma
from pathlib import Path
import shutil

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

runner = CliRunner()


def test_help():
    """Test calling `janus phonons --help`."""
    result = runner.invoke(app, ["phonons", "--help"])
    assert result.exit_code == 0
    assert "Usage: janus phonons [OPTIONS]" in strip_ansi_codes(result.stdout)


def test_phonons():
    """Test calculating force constants and band structure."""
    results_dir = Path("./janus_results")
    phonopy_path = results_dir / "NaCl-phonopy.yml"
    bands_path = results_dir / "NaCl-auto_bands.yml.xz"
    log_path = results_dir / "NaCl-phonons-log.yml"
    summary_path = results_dir / "NaCl-phonons-summary.yml"

    assert not results_dir.exists()

    try:
        result = runner.invoke(
            app,
            [
                "phonons",
                "--struct",
                DATA_PATH / "NaCl.cif",
                "--no-hdf5",
                "--bands",
            ],
        )
        assert result.exit_code == 0

        assert phonopy_path.exists()
        assert bands_path.exists()
        assert log_path.exists()
        assert summary_path.exists()

        # Read phonons summary file
        with open(summary_path, encoding="utf8") as file:
            phonon_summary = yaml.safe_load(file)

        has_eigenvectors = False
        has_velocity = False
        with lzma.open(bands_path, mode="rt") as file:
            for line in file:
                if "eigenvector" in line:
                    has_eigenvectors = True
                if "group_velocity" in line:
                    has_velocity = True
                if has_eigenvectors and has_velocity:
                    break
        assert has_eigenvectors and has_velocity

        assert "command" in phonon_summary
        assert "janus phonons" in phonon_summary["command"]
        assert "start_time" in phonon_summary
        assert "config" in phonon_summary
        assert "info" in phonon_summary
        assert "end_time" in phonon_summary

        assert "emissions" in phonon_summary
        assert phonon_summary["emissions"] > 0

        output_files = {
            "params": phonopy_path,
            "force_constants": None,
            "bands": bands_path,
            "bands_plot": None,
            "dos": None,
            "dos_plot": None,
            "band_dos_plot": None,
            "pdos": None,
            "pdos_plot": None,
            "thermal": None,
            "minimized_initial_structure": None,
            "log": log_path,
            "summary": summary_path,
        }
        check_output_files(summary=phonon_summary, output_files=output_files)

    finally:
        shutil.rmtree(results_dir, ignore_errors=True)
        clear_log_handlers()


def test_bands_simple(tmp_path):
    """Test calculating force constants and reduced bands information."""
    file_prefix = tmp_path / "NaCl"
    autoband_results = tmp_path / "NaCl-auto_bands.yml.xz"
    summary_path = tmp_path / "NaCl-phonons-summary.yml"

    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--bands",
            "--n-qpoints",
            21,
            "--no-write-full",
            "--no-hdf5",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 0

    assert autoband_results.exists()
    with lzma.open(autoband_results, mode="rb") as file:
        bands = yaml.safe_load(file)
    assert "eigenvector" not in bands["phonon"][0]["band"][0]
    assert bands["nqpoint"] == 126

    # Read phonons summary file
    assert summary_path.exists()
    with open(summary_path, encoding="utf8") as file:
        phonon_summary = yaml.safe_load(file)

    assert "command" in phonon_summary
    assert "janus phonons" in phonon_summary["command"]
    assert "config" in phonon_summary
    assert phonon_summary["config"]["bands"]


def test_hdf5(tmp_path):
    """Test saving force constants to HDF5 in new directory."""
    file_prefix = tmp_path / "test" / "NaCl"
    phonon_results = tmp_path / "test" / "NaCl-phonopy.yml"
    hdf5_results = tmp_path / "test" / "NaCl-force_constants.hdf5"
    summary_path = tmp_path / "test" / "NaCl-phonons-summary.yml"
    log_path = tmp_path / "test" / "NaCl-phonons-log.yml"

    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--file-prefix",
            file_prefix,
            "--hdf5",
        ],
    )
    assert result.exit_code == 0
    assert phonon_results.exists()
    assert hdf5_results.exists()

    # Read phonons summary file
    with open(summary_path, encoding="utf8") as file:
        phonon_summary = yaml.safe_load(file)

    output_files = {
        "params": phonon_results,
        "force_constants": hdf5_results,
        "bands": None,
        "bands_plot": None,
        "dos": None,
        "dos_plot": None,
        "band_dos_plot": None,
        "pdos": None,
        "pdos_plot": None,
        "thermal": None,
        "minimized_initial_structure": None,
        "log": log_path,
        "summary": summary_path,
    }
    check_output_files(summary=phonon_summary, output_files=output_files)


def test_thermal_props(tmp_path):
    """Test calculating thermal properties."""
    file_prefix = tmp_path / "test" / "NaCl"
    thermal_results = tmp_path / "test" / "NaCl-thermal.yml"
    summary_path = tmp_path / "test" / "NaCl-phonons-summary.yml"

    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--thermal",
            "--no-hdf5",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 0
    assert thermal_results.exists()

    # Read phonons summary file
    with open(summary_path, encoding="utf8") as file:
        phonon_summary = yaml.safe_load(file)

    output_files = {
        "thermal": thermal_results,
    }
    check_output_files(summary=phonon_summary, output_files=output_files)


def test_dos(tmp_path):
    """Test calculating the DOS."""
    file_prefix = tmp_path / "test" / "NaCl"
    dos_results = tmp_path / "test" / "NaCl-dos.dat"

    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--dos",
            "--dos-kwargs",
            "{'freq_min': -1, 'freq_max': 0}",
            "--no-hdf5",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 0
    assert dos_results.exists()
    lines = dos_results.read_text().splitlines()
    assert lines[1].split()[0] == "-1.0000000000"
    assert lines[-1].split()[0] == "0.0000000000"


def test_pdos(tmp_path):
    """Test calculating the PDOS."""
    file_prefix = tmp_path / "test" / "NaCl"
    pdos_results = tmp_path / "test" / "NaCl-pdos.dat"

    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--pdos",
            "--pdos-kwargs",
            "{'freq_min': -1, 'freq_max': 0, 'xyz_projection': True}",
            "--no-hdf5",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 0
    assert pdos_results.exists()
    with open(pdos_results, encoding="utf8") as file:
        lines = file.readlines()
    assert lines[1].split()[0] == "-1.0000000000"
    assert lines[-1].split()[0] == "0.0000000000"


def test_plot(tmp_path):
    """Test for plotting routines."""
    file_prefix = tmp_path / "NaCl"
    phonon_results = tmp_path / "NaCl-phonopy.yml"
    bands_path = tmp_path / "NaCl-auto_bands.yml.xz"
    pdos_results = tmp_path / "NaCl-pdos.dat"
    dos_results = tmp_path / "NaCl-dos.dat"
    autoband_results = tmp_path / "NaCl-auto_bands.yml.xz"
    summary_path = tmp_path / "NaCl-phonons-summary.yml"
    log_path = tmp_path / "NaCl-phonons-log.yml"
    svgs = [
        tmp_path / "NaCl-dos.svg",
        tmp_path / "NaCl-pdos.svg",
        tmp_path / "NaCl-bs-dos.svg",
        tmp_path / "NaCl-bands.svg",
    ]

    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--supercell",
            "1 1 1",
            "--pdos",
            "--dos",
            "--bands",
            "--no-hdf5",
            "--plot-to-file",
            "--no-write-full",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 0
    assert pdos_results.exists()
    assert dos_results.exists()
    for svg in svgs:
        assert svg.exists()

    assert autoband_results.exists()

    # Read phonons summary file
    assert summary_path.exists()
    with open(summary_path, encoding="utf8") as file:
        phonon_summary = yaml.safe_load(file)
    assert phonon_summary["config"]["bands"]
    assert phonon_summary["config"]["dos"]
    assert phonon_summary["config"]["pdos"]

    # Read phonons summary file
    with open(summary_path, encoding="utf8") as file:
        phonon_summary = yaml.safe_load(file)

    output_files = {
        "params": phonon_results,
        "force_constants": None,
        "bands": bands_path,
        "bands_plot": svgs[3],
        "dos": dos_results,
        "dos_plot": svgs[0],
        "band_dos_plot": [svgs[2]],
        "pdos": pdos_results,
        "pdos_plot": svgs[1],
        "thermal": None,
        "minimized_initial_structure": None,
        "log": log_path,
        "summary": summary_path,
    }
    check_output_files(summary=phonon_summary, output_files=output_files)


test_data = [
    ("1", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ("1 2 3", [[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
    ("1 1 0 -1 1 0 0 0 2", [[1, 1, 0], [-1, 1, 0], [0, 0, 2]]),
]


@pytest.mark.parametrize("supercell,supercell_matrix", test_data)
def test_supercell(supercell, supercell_matrix, tmp_path):
    """Test setting the supercell."""
    file_prefix = tmp_path / "NaCl"
    param_file = tmp_path / "NaCl-phonopy.yml"

    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--supercell",
            supercell,
            "--no-hdf5",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 0

    # Check parameters
    with open(param_file, encoding="utf8") as file:
        params = yaml.safe_load(file)

    assert "supercell_matrix" in params
    assert len(params["supercell_matrix"]) == 3
    assert params["supercell_matrix"] == supercell_matrix


test_data = ["2x2x2", "2.1 2.1 2.1", "2 2 a", "2 2", "2 2 2 2 2 2"]


@pytest.mark.parametrize("supercell", test_data)
def test_invalid_supercell(supercell, tmp_path):
    """Test errors are raise for invalid supercells."""
    file_prefix = tmp_path / "test"

    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--supercell",
            supercell,
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 1 or result.exit_code == 2


def test_minimize_kwargs(tmp_path):
    """Test setting optimizer function and writing optimized structure."""
    file_prefix = tmp_path / "test"
    opt_path = tmp_path / "test-opt.extxyz"
    log_path = tmp_path / "test-phonons-log.yml"

    minimize_kwargs = "{'optimizer': 'FIRE', 'write_results': True}"

    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--minimize",
            "--minimize-kwargs",
            minimize_kwargs,
            "--no-hdf5",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 0

    assert_log_contains(
        log_path,
        includes=["Starting geometry optimization", "Using optimizer: FIRE"],
    )
    assert opt_path.exists()


def test_minimize_filename(tmp_path):
    """Test minimize filename overwrites default."""
    file_prefix = tmp_path / "test" / "test"
    opt_path = tmp_path / "test" / "geomopt-opt.extxyz"
    summary_path = tmp_path / "test" / "test-phonons-summary.yml"

    # write_results should be set automatically
    minimize_kwargs = f"{{'write_kwargs': {{'filename': '{str(opt_path)}'}}}}"

    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--minimize",
            "--minimize-kwargs",
            minimize_kwargs,
            "--no-hdf5",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 0
    assert opt_path.exists()

    # Read phonons summary file
    with open(summary_path, encoding="utf8") as file:
        phonon_summary = yaml.safe_load(file)

    output_files = {
        "minimized_initial_structure": opt_path,
    }
    check_output_files(summary=phonon_summary, output_files=output_files)


@pytest.mark.parametrize("read_kwargs", ["{'index': 0}", "{}"])
def test_valid_traj_input(read_kwargs, tmp_path):
    """Test valid trajectory input structure handled."""
    file_prefix = tmp_path / "traj"
    phonon_results = tmp_path / "traj-phonopy.yml"

    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl-traj.xyz",
            "--supercell",
            "1 1 1",
            "--read-kwargs",
            read_kwargs,
            "--no-hdf5",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 0
    assert phonon_results.exists()


def test_invalid_traj_input(tmp_path):
    """Test invalid trajectory input structure handled."""
    file_prefix = tmp_path / "NaCl"

    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl-traj.xyz",
            "--read-kwargs",
            "{'index': ':'}",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)


def test_no_carbon(tmp_path):
    """Test disabling carbon tracking."""
    file_prefix = tmp_path / "NaCl"
    summary_path = tmp_path / "NaCl-phonons-summary.yml"

    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--no-hdf5",
            "--no-tracker",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 0

    # Read phonons summary file
    with open(summary_path, encoding="utf8") as file:
        phonon_summary = yaml.safe_load(file)
        assert "emissions" not in phonon_summary


def test_displacement_kwargs(tmp_path):
    """Test displacement_kwargs can be set."""
    file_prefix_1 = tmp_path / "NaCl_1"
    file_prefix_2 = tmp_path / "NaCl_2"
    displacement_file_1 = tmp_path / "NaCl_1-phonopy.yml"
    displacement_file_2 = tmp_path / "NaCl_2-phonopy.yml"

    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--no-hdf5",
            "--displacement-kwargs",
            "{'is_plusminus': True}",
            "--file-prefix",
            file_prefix_1,
        ],
    )
    assert result.exit_code == 0

    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--no-hdf5",
            "--displacement-kwargs",
            "{'is_plusminus': False}",
            "--file-prefix",
            file_prefix_2,
        ],
    )
    assert result.exit_code == 0

    # Check parameters
    with open(displacement_file_1, encoding="utf8") as file:
        params = yaml.safe_load(file)
        n_displacements_1 = len(params["displacements"])

    assert n_displacements_1 == 4

    with open(displacement_file_2, encoding="utf8") as file:
        params = yaml.safe_load(file)
        n_displacements_2 = len(params["displacements"])

    assert n_displacements_2 == 2


def test_paths(tmp_path):
    """Test displacement_kwargs can be set."""
    file_prefix = tmp_path / "NaCl"
    qpoint_file = DATA_PATH / "paths.yml"
    band_results = tmp_path / "NaCl-bands.yml.xz"

    result = runner.invoke(
        app,
        [
            "phonons",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--no-hdf5",
            "--bands",
            "--qpoint-file",
            qpoint_file,
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 0

    assert band_results.exists()
    with lzma.open(band_results, mode="rb") as file:
        bands = yaml.safe_load(file)
    assert bands["nqpoint"] == 11
    assert bands["npath"] == 1
