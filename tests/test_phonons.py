"""Test phonons calculations."""

from pathlib import Path

from ase.io import read

from janus_core.calculations.phonons import Phonons
from janus_core.calculations.single_point import SinglePoint
from janus_core.helpers.mlip_calculators import choose_calculator
from tests.utils import assert_log_contains

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"


def test_init():
    """Test initialising Phonons."""
    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    phonons = Phonons(
        struct=single_point.struct,
        struct_name=single_point.struct_name,
    )
    assert phonons.struct_name == "NaCl"


def test_calc_phonons():
    """Test calculating phonons from ASE atoms object."""
    struct = read(DATA_PATH / "NaCl.cif")
    struct.calc = choose_calculator(architecture="mace_mp", model=MODEL_PATH)

    phonons = Phonons(
        struct=struct,
    )

    phonons.calc_phonons(write_results=False)
    assert "phonon" in phonons.results


def test_optimize(tmp_path):
    """Test optimizing structure before calculation."""
    log_file = tmp_path / "phonons.log"
    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    phonons = Phonons(
        struct=single_point.struct,
        log_kwargs={"filename": log_file},
        minimize=True,
    )
    phonons.calc_phonons(write_results=False)

    assert_log_contains(
        log_file,
        includes=["Using filter", "Using optimizer", "Starting geometry optimization"],
    )
