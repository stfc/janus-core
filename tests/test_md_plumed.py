"""Test integration of PLUMED with molecular dynamics."""

from __future__ import annotations

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

    output_files = nvt.output_files
    assert "plumed_log" in output_files
    assert output_files["plumed_log"] is None
