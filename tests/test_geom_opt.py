"""Test geometry optimization."""

from pathlib import Path

try:
    from ase.filters import UnitCellFilter
except ImportError:
    from ase.constraints import UnitCellFilter
from ase.io import read
import pytest

from janus_core.geom_opt import optimize
from janus_core.mlip_calculators import choose_calculator
from janus_core.single_point import SinglePoint

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"

test_data = [
    ("mace", "NaCl.cif", -27.046359959669214, {}),
    ("mace", "NaCl.cif", -27.04636199814088, {"fmax": 0.001}),
    ("mace", "NaCl.cif", -27.0463392211678, {"filter_func": UnitCellFilter}),
    ("mace", "H2O.cif", -14.051389496520015, {"filter_func": None}),
    (
        "mace",
        "NaCl.cif",
        -27.04631447979008,
        {"filter_func": UnitCellFilter, "filter_kwargs": {"scalar_pressure": 0.0001}},
    ),
    ("mace", "NaCl.cif", -27.046353221978332, {"opt_kwargs": {"alpha": 100}}),
    (
        "mace",
        "NaCl.cif",
        -27.03561540212425,
        {"filter_func": UnitCellFilter, "steps": 1},
    ),
]


@pytest.mark.parametrize("architecture, struct_path, expected, kwargs", test_data)
def test_optimize(architecture, struct_path, expected, kwargs):
    """Test optimizing geometry using MACE."""
    single_point = SinglePoint(
        struct_path=DATA_PATH / struct_path,
        architecture=architecture,
        calc_kwargs={"model": MODEL_PATH},
    )

    init_energy = single_point.run("energy")["energy"]

    atoms = optimize(single_point.struct, **kwargs)

    assert atoms.get_potential_energy() < init_energy
    assert atoms.get_potential_energy() == pytest.approx(expected)


def test_saving_struct(tmp_path):
    """Test saving optimized structure."""
    struct_path = tmp_path / "NaCl.xyz"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    init_energy = single_point.run("energy")["energy"]

    optimize(
        single_point.struct,
        write_results=True,
        write_kwargs={"filename": struct_path, "format": "extxyz"},
    )
    opt_struct = read(struct_path)

    assert opt_struct.get_potential_energy() < init_energy


def test_saving_traj(tmp_path):
    """Test saving optimization trajectory output."""
    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    optimize(
        single_point.struct, opt_kwargs={"trajectory": str(tmp_path / "NaCl.traj")}
    )
    traj = read(tmp_path / "NaCl.traj", index=":")
    assert len(traj) == 3


def test_traj_reformat(tmp_path):
    """Test saving optimization trajectory in different format."""
    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    traj_path_binary = tmp_path / "NaCl.traj"
    traj_path_xyz = tmp_path / "NaCl-traj.xyz"

    optimize(
        single_point.struct,
        opt_kwargs={"trajectory": str(traj_path_binary)},
        traj_kwargs={"filename": traj_path_xyz},
    )
    traj = read(tmp_path / "NaCl-traj.xyz", index=":")

    assert len(traj) == 3


def test_missing_traj_kwarg(tmp_path):
    """Test saving optimization trajectory in different format."""
    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    traj_path = tmp_path / "NaCl-traj.xyz"
    with pytest.raises(ValueError):
        optimize(single_point.struct, traj_kwargs={"filename": traj_path})


def test_hydrostatic_strain():
    """Test setting hydrostatic strain for filter."""
    single_point_1 = SinglePoint(
        struct_path="./tests/data/NaCl-deformed.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    single_point_2 = SinglePoint(
        struct_path="./tests/data/NaCl-deformed.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    atoms_1 = optimize(
        single_point_1.struct, filter_kwargs={"hydrostatic_strain": True}
    )
    atoms_2 = optimize(
        single_point_2.struct, filter_kwargs={"hydrostatic_strain": False}
    )

    expected_1 = [5.69139709, 5.69139709, 5.69139709, 89.0, 90.0, 90.0]
    expected_2 = [5.68834069, 5.68893345, 5.68932555, 89.75938298, 90.0, 90.0]
    assert atoms_1.cell.cellpar() == pytest.approx(expected_1)
    assert atoms_2.cell.cellpar() == pytest.approx(expected_2)


def test_set_calc():
    """Test setting the calculator without SinglePoint."""
    atoms = read(DATA_PATH / "NaCl.cif")
    atoms.calc = choose_calculator(architecture="mace_mp", model=MODEL_PATH)

    init_energy = atoms.get_potential_energy()
    opt_atoms = optimize(atoms)
    assert opt_atoms.get_potential_energy() < init_energy
