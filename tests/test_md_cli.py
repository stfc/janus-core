"""Test md commandline interface."""

from __future__ import annotations

from pathlib import Path
import shutil

from ase import Atoms
from ase.io import read
import ase.md.nose_hoover_chain
import numpy as np
import pytest
from typer.testing import CliRunner
import yaml

from janus_core.cli.janus import app
from janus_core.helpers.stats import Stats
from tests.utils import (
    assert_log_contains,
    check_output_files,
    clear_log_handlers,
    strip_ansi_codes,
)

if hasattr(ase.md.nose_hoover_chain, "IsotropicMTKNPT"):
    MTK_IMPORT_FAILED = False
else:
    MTK_IMPORT_FAILED = True

DATA_PATH = Path(__file__).parent / "data"
MACE_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"

runner = CliRunner()


def test_md_help():
    """Test calling `janus md --help`."""
    result = runner.invoke(app, ["md", "--help"])
    assert result.exit_code == 0
    assert "Usage: janus md [OPTIONS]" in strip_ansi_codes(result.stdout)


test_data = [
    ("nvt"),
    ("nve"),
    ("npt"),
    ("nvt-nh"),
    ("nph"),
    ("nvt-csvr"),
    pytest.param(
        "npt-mtk",
        marks=pytest.mark.skipif(
            MTK_IMPORT_FAILED, reason="Requires updated version of ASE"
        ),
    ),
]


@pytest.mark.parametrize("ensemble", test_data)
def test_md(ensemble):
    """Test all MD simulations are able to run."""
    # Expected default file prefix for each ensemble
    file_prefix = {
        "nvt": "NaCl-nvt-T300.0-",
        "nve": "NaCl-nve-T300.0-",
        "npt": "NaCl-npt-T300.0-p0.0-",
        "nvt-nh": "NaCl-nvt-nh-T300.0-",
        "nph": "NaCl-nph-T300.0-p0.0-",
        "nvt-csvr": "NaCl-nvt-csvr-T300.0-",
        "npt-mtk": "NaCl-npt-mtk-T300.0-p0.0-",
    }

    results_dir = Path("./janus_results")
    output_files = {
        key: results_dir / (file_prefix[ensemble] + name)
        for key, name in [
            ("final_structure", "final.extxyz"),
            ("restarts", "res-2.extxyz"),
            ("stats", "stats.dat"),
            ("trajectory", "traj.extxyz"),
            ("rdfs", "rdf.dat"),
            ("vafs", "vaf.dat"),
            ("correlations", "cor.dat"),
            ("log", "md-log.yml"),
            ("summary", "md-summary.yml"),
        ]
    }

    assert not results_dir.exists()

    try:
        result = runner.invoke(
            app,
            [
                "md",
                "--ensemble",
                ensemble,
                "--struct",
                DATA_PATH / "NaCl.cif",
                "--arch",
                "mace_mp",
                "--steps",
                2,
                "--traj-every",
                2,
                "--stats-every",
                2,
                "--restart-every",
                2,
                "--post-process-kwargs",
                "{'rdf_compute': True, 'vaf_compute': True}",
                "--correlation-kwargs",
                (
                    "{'vaf': {'a': 'Velocity', 'points': 10},"
                    " 'vaf_x': {'a': 'Velocity', 'points': 10,"
                    "'a_kwargs': {'components': ['x']}, 'b_kwargs': '.'}}"
                ),
            ],
        )

        assert result.exit_code == 0

        for file in output_files.values():
            assert file.exists()

        # Check at least one image has been saved in trajectory
        atoms = read(output_files["trajectory"])
        assert isinstance(atoms, Atoms)
        assert "energy" in atoms.calc.results
        assert "mace_mp_energy" in atoms.info
        assert "forces" in atoms.calc.results
        assert "mace_mp_forces" in atoms.arrays
        assert "momenta" in atoms.arrays
        assert "masses" in atoms.arrays

        expected_units = {
            "time": "fs",
            "real_time": "s",
            "energy": "eV",
            "forces": "ev/Ang",
            "stress": "ev/Ang^3",
            "temperature": "K",
            "density": "g/cm^3",
            "momenta": "(eV*u)^0.5",
        }
        if ensemble in ("nvt", "nvt-nh"):
            expected_units["pressure"] = "GPa"

        assert "units" in atoms.info
        for prop, units in expected_units.items():
            assert atoms.info["units"][prop] == units

        # Read summary
        with open(output_files["summary"], encoding="utf8") as file:
            summary = yaml.safe_load(file)

        for path in ("restarts", "rdfs", "vafs"):
            output_files[path] = [output_files[path]]
        output_files["minimized_initial_structure"] = None

        check_output_files(summary=summary, output_files=output_files)

        assert "2/2" in result.output

    finally:
        shutil.rmtree(results_dir, ignore_errors=True)
        clear_log_handlers()


def test_log(tmp_path):
    """Test log correctly written for MD."""
    file_prefix = tmp_path / "NaCl"
    stats_path = tmp_path / "NaCl-stats.dat"
    log_path = tmp_path / "NaCl-md-log.yml"

    result = runner.invoke(
        app,
        [
            "md",
            "--ensemble",
            "nvt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--steps",
            20,
            "--stats-every",
            1,
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 0

    assert_log_contains(log_path, includes=["Starting molecular dynamics simulation"])

    with open(stats_path, encoding="utf8") as stats_file:
        lines = stats_file.readlines()

        # Includes step 0
        assert len(lines) == 22

        # Test constant volume
        assert lines[0].split(" | ")[8] == "Volume [Ang^3]"
        init_volume = float(lines[1].split()[8])
        final_volume = float(lines[-1].split()[8])
        assert init_volume == 179.406144
        assert init_volume == pytest.approx(final_volume)

        # Test constant temperature
        assert lines[0].split(" | ")[16] == "Target_T [K]\n"
        init_temp = float(lines[1].split()[16])
        final_temp = float(lines[-1].split()[16])
        assert init_temp == 300.0
        assert final_temp == pytest.approx(final_temp)


def test_seed(tmp_path):
    """Test seed enables reproducable results for NVT."""
    file_prefix = tmp_path / "nvt-T300"
    stats_path = tmp_path / "nvt-T300-stats.dat"

    result_1 = runner.invoke(
        app,
        [
            "md",
            "--ensemble",
            "nvt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--file-prefix",
            file_prefix,
            "--steps",
            20,
            "--stats-every",
            20,
            "--seed",
            42,
        ],
    )
    assert result_1.exit_code == 0

    with open(stats_path, encoding="utf8") as stats_file:
        lines = stats_file.readlines()
        # Includes step 0
        assert len(lines) == 3

        final_stats_1 = lines[2].split()

    stats_path.unlink()

    result_2 = runner.invoke(
        app,
        [
            "md",
            "--ensemble",
            "nvt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--file-prefix",
            file_prefix,
            "--steps",
            20,
            "--stats-every",
            20,
            "--seed",
            42,
        ],
    )
    assert result_2.exit_code == 0

    with open(stats_path, encoding="utf8") as stats_file:
        lines = stats_file.readlines()
        # Includes step 0
        assert len(lines) == 3

        final_stats_2 = lines[2].split()

    for i, (stats_1, stats_2) in enumerate(
        zip(final_stats_1, final_stats_2, strict=True)
    ):
        if i != 1:
            assert stats_1 == stats_2


def test_summary(tmp_path):
    """Test summary file can be read correctly."""
    file_prefix = tmp_path / "nve"
    stats_path = tmp_path / "nve-stats.dat"
    traj_path = tmp_path / "nve-traj.extxyz"
    final_path = tmp_path / "nve-final.extxyz"
    log_path = tmp_path / "nve-md-log.yml"
    summary_path = tmp_path / "nve-md-summary.yml"

    result = runner.invoke(
        app,
        [
            "md",
            "--ensemble",
            "nve",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--file-prefix",
            file_prefix,
            "--steps",
            2,
            "--traj-every",
            1,
        ],
    )

    assert result.exit_code == 0

    # Read summary
    with open(summary_path, encoding="utf8") as file:
        summary = yaml.safe_load(file)

    assert "command" in summary
    assert "janus md" in summary["command"]
    assert "start_time" in summary
    assert "config" in summary
    assert "info" in summary
    assert "end_time" in summary

    assert "ensemble" in summary["config"]
    assert "struct" in summary["info"]
    assert "n_atoms" in summary["info"]["struct"]

    assert "emissions" in summary
    assert summary["emissions"] > 0

    output_files = {
        "stats": stats_path,
        "trajectory": traj_path,
        "final_structure": final_path,
        "restarts": None,
        "minimized_initial_structure": None,
        "rdfs": None,
        "vafs": None,
        "correlations": None,
        "log": log_path,
        "summary": summary_path,
    }
    check_output_files(summary=summary, output_files=output_files)


def test_config(tmp_path):
    """Test passing a config file."""
    file_prefix = tmp_path / "nvt"
    log_path = tmp_path / "nvt-md-log.yml"
    summary_path = tmp_path / "nvt-md-summary.yml"

    result = runner.invoke(
        app,
        [
            "md",
            "--ensemble",
            "nve",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--file-prefix",
            file_prefix,
            "--steps",
            2,
            "--minimize",
            "--config",
            DATA_PATH / "md_config.yml",
        ],
    )
    assert result.exit_code == 0

    # Read md summary file
    with open(summary_path, encoding="utf8") as file:
        md_summary = yaml.safe_load(file)

    # Check temperature is passed correctly
    assert md_summary["config"]["temp"] == 200
    # Check explicit option overwrites config
    assert md_summary["config"]["ensemble"] == "nve"
    # Check nested dictionary
    assert (
        md_summary["config"]["minimize_kwargs"]["filter_kwargs"]["hydrostatic_strain"]
        is True
    )

    # Check hydrostatic strain passed correctly
    assert_log_contains(log_path, includes=["hydrostatic_strain: True"])


@pytest.mark.parametrize("ensemble", test_data)
def test_heating(tmp_path, ensemble):
    """Test heating before MD."""
    file_prefix = tmp_path / "nvt-T300"

    result = runner.invoke(
        app,
        [
            "md",
            "--ensemble",
            ensemble,
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--file-prefix",
            file_prefix,
            "--stats-every",
            1,
            "--steps",
            5,
            "--temp-start",
            10,
            "--temp-end",
            20,
            "--temp-step",
            10,
            "--temp-time",
            2,
        ],
    )
    if ensemble in ("nve", "nph"):
        assert result.exit_code != 0
    else:
        assert result.exit_code == 0


def test_invalid_config():
    """Test passing a config file with an invalid option name."""
    result = runner.invoke(
        app,
        [
            "md",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--ensemble",
            "nvt",
            "--config",
            DATA_PATH / "invalid.yml",
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)


def test_ensemble_kwargs(tmp_path):
    """Test passing ensemble-kwargs to NPT."""
    struct_path = DATA_PATH / "NaCl.cif"
    file_prefix = tmp_path / "test" / "md"
    final_path = tmp_path / "test" / "md-final.extxyz"
    stats_path = tmp_path / "test" / "md-stats.dat"

    ensemble_kwargs = "{'mask' : (0, 1, 0)}"

    result = runner.invoke(
        app,
        [
            "md",
            "--ensemble",
            "npt",
            "--struct",
            struct_path,
            "--arch",
            "mace_mp",
            "--file-prefix",
            file_prefix,
            "--steps",
            2,
            "--ensemble-kwargs",
            ensemble_kwargs,
            "--stats-every",
            1,
        ],
    )

    assert result.exit_code == 0
    assert final_path.exists()
    assert stats_path.exists()

    with open(stats_path, encoding="utf8") as stats_file:
        lines = stats_file.readlines()
        # Includes step 0
        assert len(lines) == 4

    init_atoms = read(struct_path)
    final_atoms = read(final_path)

    assert np.array_equal(init_atoms.cell[0], final_atoms.cell[0])
    assert not np.array_equal(init_atoms.cell[1], final_atoms.cell[1])
    assert np.array_equal(init_atoms.cell[2], final_atoms.cell[2])


def test_invalid_ensemble_kwargs(tmp_path):
    """Test passing invalid key to ensemble-kwargs."""
    file_prefix = tmp_path / "npt-T300"

    # Not an option for NVT
    ensemble_kwargs = "{'mask': (0, 1, 0)}"

    result = runner.invoke(
        app,
        [
            "md",
            "--ensemble",
            "nvt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--file-prefix",
            file_prefix,
            "--steps",
            2,
            "--ensemble-kwargs",
            ensemble_kwargs,
            "--traj-every",
            1,
        ],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, TypeError)


def test_final_name(tmp_path):
    """Test specifying the final file name."""
    file_prefix = tmp_path / "npt"
    stats_path = tmp_path / "npt-stats.dat"
    traj_path = tmp_path / "npt-traj.extxyz"
    final_path = tmp_path / "example.extxyz"

    result = runner.invoke(
        app,
        [
            "md",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--ensemble",
            "nvt",
            "--steps",
            "2",
            "--stats-every",
            1,
            "--traj-every",
            1,
            "--file-prefix",
            file_prefix,
            "--final-file",
            final_path,
        ],
    )
    assert result.exit_code == 0
    assert traj_path.exists()
    assert stats_path.exists()
    assert final_path.exists()


def test_write_kwargs(tmp_path):
    """Test passing write-kwargs."""
    struct_path = DATA_PATH / "NaCl.cif"
    file_prefix = tmp_path / "md"
    final_path = tmp_path / "md-final.extxyz"
    traj_path = tmp_path / "md-traj.extxyz"

    write_kwargs = (
        "{'invalidate_calc': True, 'columns': ['symbols', 'positions', 'masses']}"
    )

    result = runner.invoke(
        app,
        [
            "md",
            "--ensemble",
            "npt",
            "--struct",
            struct_path,
            "--arch",
            "mace_mp",
            "--file-prefix",
            file_prefix,
            "--steps",
            2,
            "--write-kwargs",
            write_kwargs,
            "--traj-every",
            1,
        ],
    )

    assert result.exit_code == 0
    assert final_path.exists()
    assert traj_path.exists()
    final_atoms = read(final_path)
    traj = read(traj_path, index=":")

    # Check columns has been set
    assert not final_atoms.has("momenta")
    assert not traj[0].has("momenta")

    # Check results saved with arch label, but calc is not attached
    assert final_atoms.calc is None
    assert traj[0].calc is None
    assert "mace_mp_energy" in traj[0].info
    assert "mace_mp_energy" in final_atoms.info

    assert "system_name" in final_atoms.info
    assert final_atoms.info["system_name"] == "NaCl"
    assert "system_name" in traj[0].info
    assert traj[0].info["system_name"] == "NaCl"


@pytest.mark.parametrize("read_kwargs", ["{'index': 1}", "{}"])
def test_valid_traj_input(read_kwargs, tmp_path):
    """Test valid trajectory input structure handled."""
    file_prefix = tmp_path / "traj"
    final_path = tmp_path / "traj-final.extxyz"

    result = runner.invoke(
        app,
        [
            "md",
            "--ensemble",
            "nvt",
            "--struct",
            DATA_PATH / "benzene-traj.xyz",
            "--arch",
            "mace_mp",
            "--file-prefix",
            file_prefix,
            "--steps",
            2,
            "--read-kwargs",
            read_kwargs,
        ],
    )
    assert result.exit_code == 0
    atoms = read(final_path)
    assert isinstance(atoms, Atoms)


def test_invalid_traj_input(tmp_path):
    """Test invalid trajectory input structure handled."""
    file_prefix = tmp_path / "traj"

    result = runner.invoke(
        app,
        [
            "md",
            "--ensemble",
            "nvt",
            "--struct",
            DATA_PATH / "benzene-traj.xyz",
            "--arch",
            "mace_mp",
            "--file-prefix",
            file_prefix,
            "--steps",
            2,
            "--read-kwargs",
            "{'index': ':'}",
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)


def test_minimize_kwargs_filename(tmp_path):
    """Test passing filename via minimize kwargs to MD."""
    file_prefix = tmp_path / "test" / "md"
    opt_path = tmp_path / "test" / "test.extxyz"
    traj_path = tmp_path / "test" / "md-traj.extxyz"
    stats_path = tmp_path / "test" / "md-stats.dat"
    final_path = tmp_path / "test" / "md-final.extxyz"
    log_path = tmp_path / "test" / "md-md-log.yml"
    summary_path = tmp_path / "test" / "md-md-summary.yml"

    result = runner.invoke(
        app,
        [
            "md",
            "--ensemble",
            "nvt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--file-prefix",
            file_prefix,
            "--steps",
            2,
            "--traj-every",
            1,
            "--stats-every",
            1,
            "--minimize",
            "--minimize-kwargs",
            f"{{'write_kwargs': {{'filename': '{opt_path.as_posix()}'}}}}",
        ],
    )
    assert result.exit_code == 0

    assert opt_path.exists()
    assert traj_path.exists()
    assert stats_path.exists()
    assert final_path.exists()

    atoms = read(opt_path)
    assert isinstance(atoms, Atoms)

    # Read summary
    with open(summary_path, encoding="utf8") as file:
        summary = yaml.safe_load(file)

    output_files = {
        "stats": stats_path,
        "trajectory": traj_path,
        "final_structure": final_path,
        "restarts": None,
        "minimized_initial_structure": opt_path,
        "rdfs": None,
        "vafs": None,
        "correlations": None,
        "log": log_path,
        "summary": summary_path,
    }
    check_output_files(summary=summary, output_files=output_files)


def test_minimize_kwargs_write_results(tmp_path):
    """Test passing write_results via minimize kwargs to MD."""
    file_prefix = tmp_path / "test" / "md"
    opt_path = tmp_path / "test" / "md-opt.extxyz"
    traj_path = tmp_path / "test" / "md-traj.extxyz"
    stats_path = tmp_path / "test" / "md-stats.dat"
    final_path = tmp_path / "test" / "md-final.extxyz"

    result = runner.invoke(
        app,
        [
            "md",
            "--ensemble",
            "nvt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--file-prefix",
            file_prefix,
            "--steps",
            2,
            "--traj-every",
            1,
            "--stats-every",
            1,
            "--minimize",
            "--minimize-kwargs",
            "{'write_results': True}",
        ],
    )
    assert result.exit_code == 0

    assert opt_path.exists()
    assert traj_path.exists()
    assert stats_path.exists()
    assert final_path.exists()

    atoms = read(opt_path)
    assert isinstance(atoms, Atoms)


def test_auto_restart(tmp_path):
    """Test auto restart with file_prefix."""
    file_prefix = tmp_path / "md"
    traj_path = tmp_path / "md-traj.extxyz"
    stats_path = tmp_path / "md-stats.dat"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    # Predicted path
    restart_path = tmp_path / "md-res-4.extxyz"

    result = runner.invoke(
        app,
        [
            "md",
            "--ensemble",
            "nvt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--steps",
            4,
            "--traj-every",
            1,
            "--stats-every",
            3,
            "--restart-every",
            4,
            "--file-prefix",
            file_prefix,
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0

    assert traj_path.exists()
    assert stats_path.exists()
    assert restart_path.exists()

    traj = read(traj_path, index=":")
    assert len(traj) == 5
    for i, struct in enumerate(traj):
        assert struct.info["step"] == i

    with open(stats_path, encoding="utf8") as stats_file:
        lines = stats_file.readlines()
    # Includes header and steps 0, 3
    assert len(lines) == 3
    assert int(lines[-1].split()[0]) == 3

    result = runner.invoke(
        app,
        [
            "md",
            "--ensemble",
            "nvt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--steps",
            7,
            "--traj-every",
            1,
            "--stats-every",
            1,
            "--file-prefix",
            file_prefix,
            "--restart",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0

    assert_log_contains(log_path, includes="Auto restart successful")

    traj = read(traj_path, index=":")
    assert len(traj) == 8
    for i, struct in enumerate(traj):
        assert struct.info["step"] == i

    with open(stats_path, encoding="utf8") as stats_file:
        lines = stats_file.readlines()
    # Includes header and steps 0, 3, and steps 5, 6, 7
    assert len(lines) == 6
    assert int(lines[-1].split()[0]) == 7

    # Check progress bar counted restart steps correctly
    assert "7/7" in result.stdout


def test_no_carbon(tmp_path):
    """Test disabling carbon tracking."""
    file_prefix = tmp_path / "nvt"
    summary_path = tmp_path / "nvt-md-summary.yml"

    result = runner.invoke(
        app,
        [
            "md",
            "--ensemble",
            "nvt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--no-tracker",
            "--file-prefix",
            file_prefix,
            "--steps",
            1,
        ],
    )

    assert result.exit_code == 0

    # Read summary
    with open(summary_path, encoding="utf8") as file:
        summary = yaml.safe_load(file)
    assert "emissions" not in summary


@pytest.mark.parametrize("ensemble", ("nvt", "npt", "nvt-csvr"))
@pytest.mark.parametrize("output_every", (1, 2))
@pytest.mark.parametrize("heating", (True, False))
def test_consistent_stats_traj(tmp_path, ensemble, output_every, heating):
    """Test data saved to statistics is consistent with trajectory info."""
    file_prefix = tmp_path / ensemble
    stats_path = tmp_path / f"{ensemble}-stats.dat"
    traj_path = tmp_path / f"{ensemble}-traj.extxyz"

    inputs = [
        "md",
        "--ensemble",
        "nvt",
        "--struct",
        DATA_PATH / "NaCl.cif",
        "--arch",
        "mace_mp",
        "--no-tracker",
        "--file-prefix",
        file_prefix,
        "--steps",
        4,
        "--stats-every",
        output_every,
        "--traj-every",
        output_every,
    ]

    if heating:
        inputs = inputs + [
            "--temp-start",
            10,
            "--temp-end",
            20,
            "--temp-step",
            10,
            "--temp-time",
            2,
        ]

    result = runner.invoke(app, inputs)
    assert result.exit_code == 0

    atoms = read(traj_path, index=":")
    temps = [a.info["temperature"] for a in atoms]

    data = Stats(stats_path)
    temp_col = data.labels.index("T")
    stats_temps = data.data[:, temp_col]

    assert len(temps) == len(stats_temps)
    assert temps == pytest.approx(stats_temps, rel=1e5)

    if ensemble in ("nvt", "npt"):
        target_temps = [a.info["target_temperature"] for a in atoms]

        target_temp_col = data.labels.index("Target_T")
        stats_target_temps = data.data[:, target_temp_col]

        assert len(target_temps) == len(stats_target_temps)
        assert target_temps == pytest.approx(stats_target_temps, rel=1e5)


def test_no_progress(tmp_path):
    """Test disabling progress bar."""
    file_prefix = tmp_path / "nvt"

    result = runner.invoke(
        app,
        [
            "md",
            "--ensemble",
            "nvt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--no-tracker",
            "--file-prefix",
            file_prefix,
            "--steps",
            2,
            "--no-progress-bar",
        ],
    )

    assert result.exit_code == 0

    assert "2/2" not in result.output


def test_model(tmp_path):
    """Test model passed correctly."""
    file_prefix = tmp_path / "NaCl"
    results_path = tmp_path / "NaCl-final.extxyz"
    log_path = tmp_path / "test.log"

    result = runner.invoke(
        app,
        [
            "md",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--model",
            MACE_PATH,
            "--ensemble",
            "nvt",
            "--steps",
            1,
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
    results_path = tmp_path / "NaCl-final.extxyz"
    log_path = tmp_path / "test.log"

    result = runner.invoke(
        app,
        [
            "md",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--ensemble",
            "nvt",
            "--steps",
            1,
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
            "md",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--ensemble",
            "nvt",
            "--file-prefix",
            file_prefix,
        ],
    )
    assert result.exit_code == 2
    assert "Missing option" in result.stdout
