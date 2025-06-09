"""Test integration of PLUMED with molecular dynamics."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import pytest

from janus_core.calculations.md import NVT

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"

PLUMED_INPUT_CONTENT = """
UNITS LENGTH=A TIME=fs ENERGY=eV
d: DISTANCE ATOMS=1,2
PRINT ARG=d FILE=COLVAR STRIDE=1
"""


@contextlib.contextmanager
def cwd(path):
    """Change working directory and return to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


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
    plumed_file = tmp_path / "plumed.dat"
    plumed_file.write_text(PLUMED_INPUT_CONTENT)
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


def test_nvt_plumed(tmp_path, plumed_input_file):
    """Test NVT with PLUMED."""
    plumed_log = tmp_path / "plumed.log"

    colvar_file = tmp_path / "COLVAR"

    with cwd(tmp_path):
        nvt = NVT(
            struct=DATA_PATH / "benzene.xyz",
            arch="mace_mp",
            model=MODEL_PATH,
            steps=5,
            timestep=0.5,
            temp=300.0,
            plumed_input=plumed_input_file,
            plumed_log=plumed_log,
            file_prefix=tmp_path / "benzene",
        )

        nvt.run()

        assert plumed_log.exists()
        assert colvar_file.exists()

        with open(colvar_file) as f:
            lines = f.readlines()
        assert len(lines) > 1
        assert "d" in lines[0]


def test_no_plumed_input():
    """Test MD without PLUMED input."""
    nvt = NVT(
        struct=DATA_PATH / "benzene.xyz",
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
                struct=DATA_PATH / "benzene.xyz",
                arch="mace_mp",
                model=MODEL_PATH,
                steps=5,
                timestep=0.5,
                temp=300.0,
                plumed_input=plumed_input_file,
                plumed_log="",
                file_prefix=tmp_path / "plumed",
            )
