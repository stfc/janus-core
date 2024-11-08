"""Test phonons calculations."""

from __future__ import annotations

from pathlib import Path

from ase.io import read
import pytest

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
        arch="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    phonons = Phonons(struct=single_point.struct)
    assert str(phonons.file_prefix) == "Cl4Na4"


def test_calc_phonons():
    """Test calculating phonons from ASE atoms object."""
    struct = read(DATA_PATH / "NaCl.cif")
    struct.calc = choose_calculator(arch="mace_mp", model=MODEL_PATH)

    phonons = Phonons(
        struct=struct,
    )

    phonons.calc_force_constants(write_force_consts=False)
    assert "phonon" in phonons.results


def test_optimize(tmp_path):
    """Test optimizing structure before calculation."""
    log_file = tmp_path / "phonons.log"
    opt_file = tmp_path / "NaCl-opt.extxyz"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    phonons = Phonons(
        struct=single_point.struct,
        log_kwargs={"filename": log_file},
        minimize=True,
        minimize_kwargs={"write_kwargs": {"filename": opt_file}},
    )
    phonons.calc_force_constants(write_force_consts=False)

    assert opt_file.exists()
    assert_log_contains(
        log_file,
        includes=["Using filter", "Using optimizer", "Starting geometry optimization"],
    )


def test_invalid_struct():
    """Test setting invalid structure."""
    single_point = SinglePoint(
        struct_path=DATA_PATH / "benzene-traj.xyz",
        arch="mace_mp",
        calc_kwargs={"model": MODEL_PATH},
    )

    with pytest.raises(NotImplementedError):
        Phonons(
            single_point.struct,
        )
    with pytest.raises(ValueError):
        Phonons(
            "structure",
        )


def test_logging(tmp_path):
    """Test attaching logger to Phonons and emissions are saved to info."""
    log_file = tmp_path / "phonons.log"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace_mp",
        calc_kwargs={"model": MODEL_PATH},
    )

    phonons = Phonons(
        struct=single_point.struct,
        log_kwargs={"filename": log_file},
        write_results=False,
    )

    assert "emissions" not in single_point.struct.info

    phonons.run()

    assert log_file.exists()
    assert single_point.struct.info["emissions"] > 0


def test_symmetrize(tmp_path):
    """Test symmetrize."""
    file_prefix = tmp_path / "NaCl"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl-deformed.cif",
        arch="mace_mp",
        calc_kwargs={"model": MODEL_PATH},
    )

    phonons_1 = Phonons(
        struct=single_point.struct.copy(),
        write_results=False,
        minimize=True,
        minimize_kwargs={"fmax": 0.001},
        symmetrize=False,
        file_prefix=file_prefix,
    )
    phonons_1.calc_force_constants()

    phonons_2 = Phonons(
        struct=phonons_1.struct.copy(),
        write_results=False,
        minimize=True,
        minimize_kwargs={"fmax": 0.001},
        symmetrize=True,
        file_prefix=file_prefix,
    )
    phonons_2.calc_force_constants()

    assert phonons_1.struct.positions != pytest.approx(phonons_2.struct.positions)
    assert phonons_1.results["phonon"].forces != pytest.approx(
        phonons_2.results["phonon"].forces
    )
