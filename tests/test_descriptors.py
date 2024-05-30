"""Test MLIP desriptors calculations."""

from pathlib import Path

from ase import Atoms
from ase.io import read

from janus_core.calculations.single_point import SinglePoint
from janus_core.helpers.descriptors import calc_descriptors
from janus_core.helpers.mlip_calculators import choose_calculator
from tests.utils import assert_log_contains

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"


def test_calc_descriptors(tmp_path):
    """Test calculating equation of state from ASE atoms object."""
    struct = read(DATA_PATH / "NaCl.cif")
    log_file = tmp_path / "descriptors.log"
    struct.calc = choose_calculator(architecture="mace_mp", model=MODEL_PATH)

    atoms = calc_descriptors(
        struct,
        log_kwargs={"filename": log_file},
    )
    assert isinstance(atoms, Atoms)
    assert "descriptor" in atoms.info
    assert "Na_descriptor" not in atoms.info
    assert "Cl_descriptor" not in atoms.info

    # Check logging
    assert_log_contains(
        log_file,
        includes=["Starting descriptors calculation"],
    )


def test_calc_elements(tmp_path):
    """Test calculating descriptors for each element from SinglePoint object."""
    log_file = tmp_path / "descriptors.log"
    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    atoms = calc_descriptors(
        single_point.struct,
        calc_elements=True,
        log_kwargs={"filename": log_file},
    )
    assert isinstance(atoms, Atoms)
    assert "descriptor" in atoms.info
    assert "Na_descriptor" in atoms.info
    assert "Cl_descriptor" in atoms.info
