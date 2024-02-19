"""Test geometry optimisation."""

from pathlib import Path

try:
    from ase.filters import UnitCellFilter
except ImportError:
    from ase.constraints import UnitCellFilter
import pytest

from janus_core.geo_opt import optimize
from janus_core.single_point import SinglePoint


def test_optimize():
    """Test optimizing geometry using MACE."""
    data_path = Path(Path(__file__).parent, "data", "NaCl.cif")
    model_path = Path(Path(__file__).parent, "models", "mace_mp_small.model")
    single_point = SinglePoint(
        system=data_path, architecture="mace", model_paths=model_path
    )

    init_energy = single_point.run_single_point("energy")["energy"]

    atoms = optimize(single_point.sys)

    assert atoms.get_potential_energy() < init_energy
    assert atoms.get_potential_energy() == pytest.approx(-27.046359959669214)


@pytest.mark.slow
def test_fmax():
    """Test changing fmax."""
    data_path = Path(Path(__file__).parent, "data", "NaCl.cif")
    model_path = Path(Path(__file__).parent, "models", "mace_mp_small.model")
    single_point = SinglePoint(
        system=data_path, architecture="mace", model_paths=model_path
    )

    atoms = optimize(single_point.sys)
    init_energy = atoms.get_potential_energy()
    atoms = optimize(single_point.sys, fmax=0.001)

    assert atoms.get_potential_energy() < init_energy
    assert atoms.get_potential_energy() == pytest.approx(-27.04636199814088)


@pytest.mark.slow
def test_filter_func():
    """Test passing a different filter to optimize."""
    data_path = Path(Path(__file__).parent, "data", "NaCl.cif")
    model_path = Path(Path(__file__).parent, "models", "mace_mp_small.model")
    single_point = SinglePoint(
        system=data_path, architecture="mace", model_paths=model_path
    )

    init_energy = single_point.run_single_point("energy")["energy"]
    atoms = optimize(single_point.sys, filter_func=UnitCellFilter)

    assert atoms.get_potential_energy() < init_energy
    assert atoms.get_potential_energy() == pytest.approx(-27.0463392211678)


@pytest.mark.slow
def test_no_filter():
    """Test passing None filter to optimize."""
    data_path = Path(Path(__file__).parent, "data", "H2O.cif")
    model_path = Path(Path(__file__).parent, "models", "mace_mp_small.model")
    single_point = SinglePoint(
        system=data_path, architecture="mace", model_paths=model_path
    )

    init_energy = single_point.run_single_point("energy")["energy"]
    atoms = optimize(single_point.sys, filter_func=None)

    assert atoms.get_potential_energy() < init_energy
    assert atoms.get_potential_energy() == pytest.approx(-14.051389496520015)


@pytest.mark.slow
def test_filter_kwargs():
    """Test passing kwargs to filter."""
    data_path = Path(Path(__file__).parent, "data", "NaCl.cif")
    model_path = Path(Path(__file__).parent, "models", "mace_mp_small.model")
    single_point = SinglePoint(
        system=data_path, architecture="mace", model_paths=model_path
    )

    init_energy = single_point.run_single_point("energy")["energy"]
    atoms = optimize(single_point.sys, filter_kwargs={"hydrostatic_strain": True})

    assert atoms.get_potential_energy() < init_energy
    assert atoms.get_potential_energy() == pytest.approx(-27.046359959669214)


@pytest.mark.slow
def test_opt_kwargs():
    """Test passing kwargs to optimizer."""
    data_path = Path(Path(__file__).parent, "data", "NaCl.cif")
    model_path = Path(Path(__file__).parent, "models", "mace_mp_small.model")
    single_point = SinglePoint(
        system=data_path, architecture="mace", model_paths=model_path
    )

    init_energy = single_point.run_single_point("energy")["energy"]
    atoms = optimize(single_point.sys, opt_kwargs={"alpha": 100})

    assert atoms.get_potential_energy() < init_energy
    assert atoms.get_potential_energy() == pytest.approx(-27.046353221978332)


@pytest.mark.slow
def test_dyn_kwargs():
    """Test passing kwargs to dynamics.run."""
    data_path = Path(Path(__file__).parent, "data", "NaCl.cif")
    model_path = Path(Path(__file__).parent, "models", "mace_mp_small.model")
    single_point = SinglePoint(
        system=data_path, architecture="mace", model_paths=model_path
    )

    init_energy = single_point.run_single_point("energy")["energy"]

    atoms = optimize(single_point.sys, dyn_kwargs={"steps": 1})

    assert atoms.get_potential_energy() < init_energy
    assert atoms.get_potential_energy() == pytest.approx(-27.044723953811527)
