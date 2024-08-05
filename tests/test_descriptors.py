"""Test MLIP desriptors calculations."""

from pathlib import Path

from ase import Atoms
from ase.io import read
import pytest

from janus_core.calculations.descriptors import Descriptors
from janus_core.calculations.single_point import SinglePoint
from janus_core.helpers.mlip_calculators import choose_calculator
from tests.utils import assert_log_contains

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"


def test_calc_descriptors(tmp_path):
    """Test calculating equation of state from ASE atoms object."""
    struct = read(DATA_PATH / "NaCl.cif")
    log_file = tmp_path / "descriptors.log"
    struct.calc = choose_calculator(architecture="mace_mp", model=MODEL_PATH)

    descriptors = Descriptors(
        struct,
        log_kwargs={"filename": log_file},
    )
    descriptors.run()
    atoms = descriptors.struct
    assert isinstance(atoms, Atoms)
    assert "mace_mp_descriptor" in atoms.info
    assert "mace_mp_Na_descriptor" not in atoms.info
    assert "mace_mp_Cl_descriptor" not in atoms.info

    # Check logging
    assert_log_contains(
        log_file,
        includes=["Starting descriptors calculation"],
    )


def test_calc_per_element(tmp_path):
    """Test calculating descriptors for each element from SinglePoint object."""
    log_file = tmp_path / "descriptors.log"
    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    descriptors = Descriptors(
        single_point.struct,
        calc_per_element=True,
        log_kwargs={"filename": log_file},
    )
    descriptors.run()
    atoms = descriptors.struct

    assert isinstance(atoms, Atoms)
    assert "mace_descriptor" in atoms.info
    assert "mace_Na_descriptor" in atoms.info
    assert "mace_Cl_descriptor" in atoms.info

    assert atoms.info["mace_descriptor"] == pytest.approx(-0.005626419559511429)
    assert atoms.info["mace_Cl_descriptor"] == pytest.approx(-0.009215340539869301)
    assert atoms.info["mace_Na_descriptor"] == pytest.approx(-0.0020374985791535563)
