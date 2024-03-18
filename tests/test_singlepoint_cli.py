"""Test singlepoint commandline interface."""

from pathlib import Path

from ase.io import read
from typer.testing import CliRunner

from janus_core.cli import app
from tests.utils import read_atoms

DATA_PATH = Path(__file__).parent / "data"

runner = CliRunner()


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


def test_singlepoint():
    """Test singlepoint calculation."""
    results_path = Path("./NaCl-results.xyz").absolute()

    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "NaCl.cif",
        ],
    )

    atoms = read_atoms(results_path)
    assert result.exit_code == 0
    assert atoms.get_potential_energy() is not None
    assert "forces" in atoms.arrays


def test_singlepoint_properties(tmp_path):
    """Test properties for singlepoint calculation."""
    results_path_1 = tmp_path / "H2O-energy-results.xyz"
    results_path_2 = tmp_path / "H2O-stress-results.xyz"

    # Check energy is can be calculated successfully
    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "H2O.cif",
            "--property",
            "energy",
            "--write-kwargs",
            f"{{'filename': '{str(results_path_1)}'}}",
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
            "--write-kwargs",
            f"{{'filename': '{str(results_path_2)}'}}",
        ],
    )
    assert result.exit_code == 1
    assert not results_path_2.is_file()
    assert isinstance(result.exception, ValueError)


def test_singlepoint_read_kwargs(tmp_path):
    """Test setting read_kwargs for singlepoint calculation."""
    results_path = tmp_path / "benzene-traj-results.xyz"

    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "benzene-traj.xyz",
            "--read-kwargs",
            "{'index': ':'}",
            "--write-kwargs",
            f"{{'filename': '{str(results_path)}'}}",
            "--property",
            "energy",
        ],
    )
    assert result.exit_code == 0

    atoms = read(results_path, index=":")
    assert isinstance(atoms, list)


def test_singlepoint_calc_kwargs(tmp_path):
    """Test setting calc_kwargs for singlepoint calculation."""
    results_path = tmp_path / "NaCl-results.xyz"

    result = runner.invoke(
        app,
        [
            "singlepoint",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--calc-kwargs",
            "{'default_dtype': 'float32'}",
            "--write-kwargs",
            f"{{'filename': '{str(results_path)}'}}",
            "--property",
            "energy",
        ],
    )
    assert result.exit_code == 0
    assert "Using float32 for MACECalculator" in result.stdout


def test_singlepoint_log(tmp_path, caplog):
    """Test log correctly written for singlepoint."""
    results_path = tmp_path / "NaCl-results.xyz"
    with caplog.at_level("INFO", logger="janus_core.single_point"):
        result = runner.invoke(
            app,
            [
                "singlepoint",
                "--struct",
                DATA_PATH / "NaCl.cif",
                "--property",
                "energy",
                "--write-kwargs",
                f"{{'filename': '{str(results_path)}'}}",
                "--log",
                f"{tmp_path}/test.log",
            ],
        )
        assert "Starting single point calculation" in caplog.text
        assert result.exit_code == 0
