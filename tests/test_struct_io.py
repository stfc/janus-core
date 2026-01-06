"""Test utility functions."""

from __future__ import annotations

from pathlib import Path

from ase.calculators.lj import LennardJones
from ase.io import read

from janus_core.helpers.struct_io import PhonopyAtomsAdaptor

DATA_PATH = Path(__file__).parent / "data"


def test_phonopyadaptor():
    """Test Phonopy atoms to ASE atoms adpator."""
    struct = read(DATA_PATH / "NaCl.cif")
    phonopy_struct = PhonopyAtomsAdaptor.get_phonopy_atoms(struct)
    assert struct == PhonopyAtomsAdaptor.get_atoms(phonopy_struct)

    struct.calc = LennardJones()
    phonopy_struct = PhonopyAtomsAdaptor.get_phonopy_atoms(struct)

    assert PhonopyAtomsAdaptor.get_atoms(phonopy_struct).calc is None
    assert (
        PhonopyAtomsAdaptor.get_atoms(phonopy_struct, struct.calc).calc == struct.calc
    )
