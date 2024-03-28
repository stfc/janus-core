"""Test singlepoint commandline interface."""

from pathlib import Path

from ase.io import read
from typer.testing import CliRunner
import yaml

from janus_core.cli import app
from tests.utils import read_atoms

DATA_PATH = Path(__file__).parent / "data"

runner = CliRunner()

# Many pylint warnings now raised due to similar log/summary flags
# These depend on tmp_path, so not easily refactorisable
# pylint: disable=duplicate-code


def test_janus_help():
    """Test calling `janus --help`."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # Command is returned as "root"
    assert "Usage: root [OPTIONS] COMMAND [ARGS]..." in result.stdout


def test_singlepoint_help():
    """Test calling `janus singlepoint --help`."""
    result = runner.invoke(app, ["singlepoint", "--help"])
    assert result.exit_code == 0
    # Command is returned as "root"
    assert "Usage: root singlepoint [OPTIONS]" in result.stdout


def test_singlepoint(tmp_path):
    """Test singlepoint calculation."""
    results_path = Path("./NaCl-results.xyz").absolute()
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )

    # Check atoms can read read, then delete file
    atoms = read_atoms(results_path)
    assert result.exit_code == 0
    assert atoms.get_potential_energy() is not None
    assert "forces" in atoms.arrays


def test_properties(tmp_path):
    """Test properties for singlepoint calculation."""
    results_path_1 = tmp_path / "H2O-energy-results.xyz"
    results_path_2 = tmp_path / "H2O-stress-results.xyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    # Check energy is can be calculated successfully
    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "H2O.cif",
            "--property",
            "energy",
            "--out",
            results_path_1,
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0

    atoms = read(results_path_1)
    assert atoms.get_potential_energy() is not None

    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "H2O.cif",
            "--property",
            "stress",
            "--out",
            results_path_2,
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 1
    assert not results_path_2.is_file()
    assert isinstance(result.exception, ValueError)


def test_read_kwargs(tmp_path):
    """Test setting read_kwargs for singlepoint calculation."""
    results_path = tmp_path / "benzene-traj-results.xyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "benzene-traj.xyz",
            "--read-kwargs",
            "{'index': ':'}",
            "--out",
            results_path,
            "--property",
            "energy",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0

    atoms = read(results_path, index=":")
    assert isinstance(atoms, list)


def test_calc_kwargs(tmp_path):
    """Test setting calc_kwargs for singlepoint calculation."""
    results_path = tmp_path / "NaCl-results.xyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--calc-kwargs",
            "{'default_dtype': 'float32'}",
            "--out",
            results_path,
            "--property",
            "energy",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 0
    assert "Using float32 for MACECalculator" in result.stdout


def test_log(tmp_path, caplog):
    """Test log correctly written for singlepoint."""
    results_path = tmp_path / "NaCl-results.xyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    with caplog.at_level("INFO", logger="janus_core.single_point"):
        result = runner.invoke(
            app,
            [
                "singlepoint",
                "--struct",
                DATA_PATH / "NaCl.cif",
                "--out",
                results_path,
                "--property",
                "energy",
                "--log",
                log_path,
                "--summary",
                summary_path,
            ],
        )
        assert "Starting single point calculation" in caplog.text
        assert result.exit_code == 0


def test_summary(tmp_path):
    """Test summary file can be read correctly."""
    results_path = tmp_path / "benzene-traj-results.xyz"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "benzene-traj.xyz",
            "--read-kwargs",
            "{'index': ':'}",
            "--out",
            results_path,
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )

    assert result.exit_code == 0

    # Read singlepoint summary file
    with open(summary_path, encoding="utf8") as file:
        sp_summary = yaml.safe_load(file)

    assert "command" in sp_summary[0]
    assert "janus singlepoint" in sp_summary[0]["command"]
    assert "start_time" in sp_summary[1]
    assert "inputs" in sp_summary[2]
    assert "end_time" in sp_summary[3]

    assert "traj" in sp_summary[2]["inputs"]
    assert "length" in sp_summary[2]["inputs"]["traj"]
    assert "struct" in sp_summary[2]["inputs"]["traj"]
    assert "n_atoms" in sp_summary[2]["inputs"]["traj"]["struct"]
