"""Test geometry optimization."""

from pathlib import Path

try:
    from ase.filters import UnitCellFilter
except ImportError:
    from ase.constraints import UnitCellFilter

from ase.io import read
import pytest

from janus_core.calculations.geom_opt import optimize
from janus_core.calculations.single_point import SinglePoint
from janus_core.helpers.mlip_calculators import choose_calculator

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

    struct = optimize(single_point.struct, **kwargs)

    assert struct.get_potential_energy() < init_energy
    assert struct.get_potential_energy() == pytest.approx(expected)


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
        struct_path=DATA_PATH / "NaCl-deformed.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    single_point_2 = SinglePoint(
        struct_path=DATA_PATH / "NaCl-deformed.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    struct_1 = optimize(
        single_point_1.struct, filter_kwargs={"hydrostatic_strain": True}
    )
    struct_2 = optimize(
        single_point_2.struct, filter_kwargs={"hydrostatic_strain": False}
    )

    expected_1 = [5.69139709, 5.69139709, 5.69139709, 89.0, 90.0, 90.0]
    expected_2 = [5.68834069, 5.68893345, 5.68932555, 89.75938298, 90.0, 90.0]
    assert struct_1.cell.cellpar() == pytest.approx(expected_1)
    assert struct_2.cell.cellpar() == pytest.approx(expected_2)


def test_set_calc():
    """Test setting the calculator without SinglePoint."""
    struct = read(DATA_PATH / "NaCl.cif")
    struct.calc = choose_calculator(architecture="mace_mp", model=MODEL_PATH)

    init_energy = struct.get_potential_energy()
    opt_struct = optimize(struct)
    assert opt_struct.get_potential_energy() < init_energy


def test_converge_warning():
    """Test warning raised if not converged."""
    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl-deformed.cif",
        calc_kwargs={"model": MODEL_PATH},
    )
    with pytest.warns(UserWarning):
        optimize(single_point.struct, steps=1)


def test_restart(tmp_path):
    """Test restarting geometry optimization."""
    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl-deformed.cif",
        calc_kwargs={"model": MODEL_PATH},
    )

    init_energy = single_point.run("energy")["energy"]

    with pytest.warns(UserWarning):
        optimize(
            single_point.struct, steps=2, opt_kwargs={"restart": tmp_path / "NaCl.pkl"}
        )

    intermediate_energy = single_point.run("energy")["energy"]
    assert intermediate_energy < init_energy

    optimize(
        single_point.struct, steps=2, opt_kwargs={"restart": tmp_path / "NaCl.pkl"}
    )
    final_energy = single_point.run("energy")["energy"]
    assert final_energy < intermediate_energy


def test_space_group():
    """Test spacegroup of the structure."""

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl-sg.cif",
        architecture="mace_mp",
        calc_kwargs={"model": MODEL_PATH},
    )

    optimize(
        single_point.struct,
        fmax=0.001,
    )

    assert single_point.struct.info["initial_spacegroup"] == "I4/mmm (139)"
    assert single_point.struct.info["final_spacegroup"] == "Fm-3m (225)"
