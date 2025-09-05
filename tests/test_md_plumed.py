"""Test integration of PLUMED with molecular dynamics."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from ase.calculators.lj import LennardJones
from ase.constraints import FixedPlane
from ase.io import read
import pytest
from typer.testing import CliRunner

from janus_core.calculations.md import NPH, NPT, NPT_MTK, NVE, NVT, NVT_CSVR, NVT_NH
from janus_core.cli.janus import app
from tests.utils import assert_log_contains, chdir

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"

runner = CliRunner()


@contextlib.contextmanager
def set_env(**environ):
    """Set the environment variables temporarily."""
    old_environ = dict(os.environ)
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


@pytest.fixture
def plumed_input_file(tmp_path):
    """Create a simple PLUMED input file for testing."""
    plumed_input = """
    UNITS LENGTH=A TIME=fs ENERGY=eV
    d: DISTANCE ATOMS=1,2
    PRINT ARG=d FILE=COLVAR STRIDE=1
    """
    plumed_file = tmp_path / "plumed.dat"
    plumed_file.write_text(plumed_input)
    return plumed_file


@pytest.fixture(autouse=True)
def test_plumed_import():
    """Skip tests if PLUMED has not been set up."""
    try:
        from plumed import Plumed

        # Raises runtime error if PLUMED_KERNEL environment variable not set
        Plumed()
    except ImportError:
        pytest.skip("PLUMED not installed")
    except RuntimeError as err:
        if any("PLUMED not available" in arg for arg in err.args):
            pytest.skip("PLUMED not configured")


@pytest.mark.parametrize("ensemble", (NVT, NVE, NVT_NH, NVT_CSVR))
def test_md_plumed(tmp_path, plumed_input_file, ensemble):
    """Test NVT with PLUMED."""
    file_prefix = tmp_path / "NaCl"
    plumed_log = tmp_path / "test" / "NaCl-plumed.log"
    colvar_file = tmp_path / "COLVAR"

    with chdir(tmp_path):
        md = ensemble(
            struct=DATA_PATH / "NaCl.cif",
            arch="mace_mp",
            model=MODEL_PATH,
            steps=5,
            timestep=0.5,
            temp=300.0,
            plumed_input=plumed_input_file,
            plumed_log=plumed_log,
            file_prefix=file_prefix,
        )

        md.run()

        assert plumed_log.exists()
        assert colvar_file.exists()
        assert md.struct.calc.istep == 6

        with open(colvar_file) as f:
            lines = f.readlines()
            assert all(field in lines[0] for field in ("time", "d"))
            assert len(lines) > 0


def test_no_plumed_input():
    """Test MD without PLUMED input."""
    nvt = NVT(
        struct=DATA_PATH / "NaCl.cif",
        arch="mace_mp",
        model=MODEL_PATH,
        steps=5,
        plumed_input=None,
    )
    assert nvt.output_files["plumed_log"] is None


def test_no_plumed_env(tmp_path):
    """Test environment variable not set."""
    with set_env(PLUMED_KERNEL=""):
        with pytest.raises(RuntimeError):
            NVT(
                struct=DATA_PATH / "NaCl.cif",
                arch="mace_mp",
                model=MODEL_PATH,
                steps=5,
                timestep=0.5,
                temp=300.0,
                plumed_input=plumed_input_file,
                plumed_log="",
                file_prefix=tmp_path / "NaCl",
            )


def test_atoms_struct(tmp_path):
    """
    Test passing a structure with an attached calculator.

    Based on tutorial: https://github.com/Sucerquia/ASE-PLUMED_tutorial.
    """
    file_prefix = tmp_path / "NaCl"
    plumed_log = tmp_path / "plumed.log"
    colvar_file = tmp_path / "COLVAR"

    struct = read(DATA_PATH / "isomer.xyz")
    cons = [FixedPlane(i, [0, 0, 1]) for i in range(7)]
    struct.set_constraint(cons)
    struct.set_masses([1] * 7)
    struct.calc = LennardJones(rc=2.5, r0=3.0)

    with chdir(tmp_path):
        nvt = NVT(
            struct=struct,
            steps=5,
            timestep=0.5,
            temp=300.0,
            plumed_input=DATA_PATH / "plumed.dat",
            plumed_log=plumed_log,
            file_prefix=file_prefix,
        )

        nvt.run()

        assert plumed_log.exists()
        assert colvar_file.exists()

        lines = colvar_file.read_text().splitlines()
        assert len(lines) > 1


def test_cli(tmp_path, plumed_input_file):
    """Test plumed via CLI."""
    file_prefix = tmp_path / "NaCl"
    log_path = tmp_path / "test.log"
    plumed_log = tmp_path / "plumed.log"

    with chdir(tmp_path):
        result = runner.invoke(
            app,
            [
                "md",
                "--struct",
                DATA_PATH / "NaCl.cif",
                "--arch",
                "mace_mp",
                "--model",
                MODEL_PATH,
                "--ensemble",
                "nvt",
                "--steps",
                1,
                "--log",
                log_path,
                "--file-prefix",
                file_prefix,
                "--no-tracker",
                "--plumed-input",
                plumed_input_file,
                "--plumed-log",
                plumed_log,
            ],
        )

    assert result.exit_code == 0

    assert_log_contains(log_path, includes=["Plumed calculator configured"])


def test_restart(tmp_path, plumed_input_file):
    """Test restarting plumed simulation."""
    file_prefix = tmp_path / "NaCl"
    log_path = tmp_path / "test.log"
    stats_path = tmp_path / "NaCl-stats.dat"

    with chdir(tmp_path):
        nvt = NVT(
            struct=DATA_PATH / "NaCl.cif",
            arch="mace",
            model=MODEL_PATH,
            temp=300.0,
            steps=4,
            restart_every=4,
            stats_every=1,
            file_prefix=file_prefix,
            plumed_input=plumed_input_file,
        )
        nvt.run()

        assert nvt.dyn.nsteps == 4

        nvt_restart = NVT(
            struct=DATA_PATH / "NaCl.cif",
            arch="mace",
            model=MODEL_PATH,
            temp=300.0,
            steps=8,
            stats_every=1,
            restart=True,
            restart_auto=True,
            file_prefix=file_prefix,
            plumed_input=plumed_input_file,
            log_kwargs={"filename": log_path},
            track_carbon=False,
        )
        nvt_restart.run()

    assert nvt_restart.offset == 4
    assert nvt_restart.struct.calc.istep == 9

    assert_log_contains(log_path, includes=["Plumed calculator configured"])

    assert stats_path.exists()

    with open(stats_path, encoding="utf8") as stats_file:
        lines = stats_file.readlines()
    # Header, steps 0 - 4, 5 - 8 in stats
    assert len(lines) == 10


def test_plumed_minimise(tmp_path, plumed_input_file):
    """Test PLUMED with minimisation before."""
    file_prefix = tmp_path / "NaCl"
    plumed_log = tmp_path / "NaCl-plumed.log"
    colvar_file = tmp_path / "COLVAR"

    with chdir(tmp_path):
        nvt = NVT(
            struct=DATA_PATH / "NaCl.cif",
            arch="mace_mp",
            model=MODEL_PATH,
            minimize=True,
            minimize_kwargs={"filter_class": None},
            steps=3,
            timestep=0.5,
            temp=300.0,
            plumed_input=plumed_input_file,
            file_prefix=file_prefix,
        )

        nvt.run()

        assert plumed_log.exists()
        assert colvar_file.exists()
        assert nvt.struct.calc.istep == 4

        with open(colvar_file) as f:
            lines = f.readlines()
            assert all(field in lines[0] for field in ("time", "d"))
            assert len(lines) > 0


@pytest.mark.parametrize("ensemble", (NPT, NPH, NPT_MTK))
def test_plumed_invalid_ensemble(tmp_path, plumed_input_file, ensemble):
    """Test error raised for incompatible ensembles."""
    file_prefix = tmp_path / "NaCl"
    plumed_log = tmp_path / "test" / "NaCl-plumed.log"

    with chdir(tmp_path):
        with pytest.raises(NotImplementedError):
            ensemble(
                struct=DATA_PATH / "NaCl.cif",
                arch="mace_mp",
                model=MODEL_PATH,
                steps=5,
                timestep=0.5,
                temp=300.0,
                plumed_input=plumed_input_file,
                plumed_log=plumed_log,
                file_prefix=file_prefix,
            )
