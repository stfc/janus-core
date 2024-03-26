"""Test md commandline interface."""

from pathlib import Path

from ase import Atoms
from ase.io import read
import pytest
from typer.testing import CliRunner

from janus_core.cli import app

DATA_PATH = Path(__file__).parent / "data"

runner = CliRunner()


def test_md_help():
    """Test calling `janus md --help`."""
    result = runner.invoke(app, ["md", "--help"])
    assert result.exit_code == 0
    # Command is returned as "root"
    assert "Usage: root md [OPTIONS]" in result.stdout


test_data = [
    ("nvt"),
    ("nve"),
    ("npt"),
    ("nvt-nh"),
    ("nph"),
]


@pytest.mark.parametrize("ensemble", test_data)
def test_md(ensemble, tmp_path):
    """Test all MD simulations are able to run."""
    file_prefix = tmp_path / f"{ensemble}-T300"
    traj_path = tmp_path / f"{ensemble}-T300-traj.xyz"

    result = runner.invoke(
        app,
        [
            "md",
            "--ensemble",
            ensemble,
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--temp",
            300,
            "--file-prefix",
            file_prefix,
            "--steps",
            2,
            "--traj-every",
            1,
        ],
    )

    assert result.exit_code == 0

    # Check at least one image has been saved in trajectory
    atoms = read(traj_path)
    assert isinstance(atoms, Atoms)


def test_md_log(tmp_path, caplog):
    """Test log correctly written for MD."""
    file_prefix = tmp_path / "nvt-T300"
    stats_path = tmp_path / "nvt-T300-stats.dat"

    with caplog.at_level("INFO", logger="janus_core.md"):
        result = runner.invoke(
            app,
            [
                "md",
                "--ensemble",
                "nvt",
                "--struct",
                DATA_PATH / "NaCl.cif",
                "--temp",
                300,
                "--file-prefix",
                file_prefix,
                "--steps",
                20,
                "--stats-every",
                1,
            ],
        )
        assert result.exit_code == 0
        assert " Starting molecular dynamics simulation" in caplog.text

        with open(stats_path, encoding="utf8") as stats_file:
            lines = stats_file.readlines()
            # Includes step 0
            assert len(lines) == 22

            # Test constant volume
            assert lines[0].split(" | ")[8] == "Volume [A^3]"
            init_volume = float(lines[1].split()[8])
            final_volume = float(lines[-1].split()[8])
            assert init_volume == 179.4
            assert init_volume == pytest.approx(final_volume)

            # Test constant temperature
            assert lines[0].split(" | ")[16] == "T [K]\n"
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
            "--temp",
            300,
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
            "--temp",
            300,
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

    for i, (stats_1, stats_2) in enumerate(zip(final_stats_1, final_stats_2)):
        if i != 1:
            assert stats_1 == stats_2
