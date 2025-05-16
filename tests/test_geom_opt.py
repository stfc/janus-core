"""Test geometry optimization."""

from __future__ import annotations

from pathlib import Path

from ase.filters import FrechetCellFilter, UnitCellFilter
from ase.io import read
import pytest

from janus_core.calculations.geom_opt import GeomOpt
from janus_core.calculations.single_point import SinglePoint
from janus_core.helpers.mlip_calculators import choose_calculator
from tests.utils import assert_log_contains

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"

test_data = [
    ("mace", "NaCl.cif", -27.046349600581266, {}),
    ("mace", "NaCl.cif", -27.046361983699768, {"fmax": 0.001}),
    ("mace", "NaCl.cif", -27.04633922116779, {"filter_class": UnitCellFilter}),
    ("mace", "H2O.cif", -14.051389496520015, {"filter_class": None}),
    (
        "mace",
        "NaCl.cif",
        -26.727162796978426,
        {
            "fmax": 0.001,
            "filter_class": FrechetCellFilter,
            "filter_kwargs": {"scalar_pressure": 5.0},
        },
    ),
    ("mace", "NaCl.cif", -27.04634943785021, {"opt_kwargs": {"alpha": 100}}),
]


@pytest.mark.parametrize("arch, struct, expected, kwargs", test_data)
def test_optimize(arch, struct, expected, kwargs):
    """Test optimizing geometry using MACE."""
    single_point = SinglePoint(
        struct=DATA_PATH / struct,
        arch=arch,
        model=MODEL_PATH,
    )

    optimizer = GeomOpt(single_point.struct, **kwargs)
    optimizer.run()

    assert single_point.struct.get_potential_energy() == pytest.approx(
        expected, rel=1e-8
    )


def test_saving_struct(tmp_path):
    """Test saving optimized structure."""
    results_path = tmp_path / "NaCl.extxyz"

    single_point = SinglePoint(
        struct=DATA_PATH / "NaCl.cif",
        arch="mace",
        model=MODEL_PATH,
        properties="energy",
    )

    init_energy = single_point.run()["energy"]

    optimizer = GeomOpt(
        single_point.struct,
        write_results=True,
        write_kwargs={"filename": results_path, "format": "extxyz"},
    )
    optimizer.run()

    opt_struct = read(results_path)
    assert opt_struct.info["mace_energy"] < init_energy


def test_saving_traj(tmp_path):
    """Test saving optimization trajectory output."""
    optimizer = GeomOpt(
        struct=DATA_PATH / "NaCl.cif",
        arch="mace",
        model=MODEL_PATH,
        write_traj=True,
        traj_kwargs={"filename": tmp_path / "NaCl.traj"},
    )
    optimizer.run()
    traj = read(tmp_path / "NaCl.traj", index=":")
    assert len(traj) == 3


def test_traj_opt_kwargs(tmp_path):
    """Test trajectory file cannot be defined through opt_kwargs."""
    traj_path_binary = tmp_path / "NaCl.traj"

    with pytest.raises(ValueError):
        GeomOpt(
            struct=DATA_PATH / "NaCl.cif",
            arch="mace",
            model=MODEL_PATH,
            opt_kwargs={"trajectory": str(traj_path_binary)},
            write_traj=True,
        )

    with pytest.raises(ValueError):
        GeomOpt(
            struct=DATA_PATH / "NaCl.cif",
            arch="mace",
            model=MODEL_PATH,
            opt_kwargs={"trajectory": str(traj_path_binary)},
            write_traj=False,
        )


def test_traj_without_write(tmp_path):
    """Test trajectory file cannot be defined without write_traj."""
    with pytest.raises(ValueError):
        GeomOpt(
            struct=DATA_PATH / "NaCl.cif",
            arch="mace",
            model=MODEL_PATH,
            traj_kwargs={"filename": tmp_path / "NaCl.traj"},
            write_traj=False,
        )


def test_hydrostatic_strain():
    """Test setting hydrostatic strain for filter."""
    single_point_1 = SinglePoint(
        struct=DATA_PATH / "NaCl-deformed.cif",
        arch="mace",
        model=MODEL_PATH,
    )

    single_point_2 = SinglePoint(
        struct=DATA_PATH / "NaCl-deformed.cif",
        arch="mace",
        model=MODEL_PATH,
    )

    optimizer_1 = GeomOpt(
        single_point_1.struct, filter_kwargs={"hydrostatic_strain": True}
    )
    optimizer_1.run()

    optimizer_2 = GeomOpt(
        single_point_2.struct, filter_kwargs={"hydrostatic_strain": False}
    )
    optimizer_2.run()

    expected_1 = [
        5.687545288920282,
        5.687545288920282,
        5.687545288920282,
        89.0,
        90.0,
        90.0,
    ]
    expected_2 = [
        5.688268799219085,
        5.688750772505896,
        5.688822747326383,
        89.26002493790229,
        90.0,
        90.0,
    ]
    assert single_point_1.struct.cell.cellpar() == pytest.approx(expected_1)
    assert single_point_2.struct.cell.cellpar() == pytest.approx(expected_2)


def test_set_calc():
    """Test setting the calculator without SinglePoint."""
    struct = read(DATA_PATH / "NaCl.cif")
    struct.calc = choose_calculator(arch="mace_mp", model=MODEL_PATH)

    init_energy = struct.get_potential_energy()
    optimizer = GeomOpt(struct)
    optimizer.run()
    assert struct.get_potential_energy() < init_energy


def test_converge_warning():
    """Test warning raised if not converged."""
    single_point = SinglePoint(
        struct=DATA_PATH / "NaCl-deformed.cif",
        arch="mace_mp",
        model=MODEL_PATH,
    )
    optimizer = GeomOpt(single_point.struct, steps=1)

    with pytest.warns(UserWarning):
        optimizer.run()


def test_restart(tmp_path):
    """Test restarting geometry optimization."""
    single_point = SinglePoint(
        struct=DATA_PATH / "NaCl-deformed.cif",
        arch="mace_mp",
        model=MODEL_PATH,
        properties="energy",
    )

    init_energy = single_point.run()["energy"]
    optimizer = GeomOpt(
        single_point.struct,
        steps=2,
        opt_kwargs={"restart": tmp_path / "NaCl.pkl"},
        fmax=0.0001,
    )
    # Check unconverged warning
    with pytest.warns(UserWarning):
        optimizer.run()

    intermediate_energy = single_point.run()["energy"]
    assert intermediate_energy < init_energy

    optimizer = GeomOpt(
        single_point.struct,
        steps=2,
        opt_kwargs={"restart": tmp_path / "NaCl.pkl"},
        fmax=0.0001,
    )
    optimizer.run()
    final_energy = single_point.run()["energy"]
    assert final_energy < intermediate_energy


def test_space_group():
    """Test spacegroup of the structure."""
    single_point = SinglePoint(
        struct=DATA_PATH / "NaCl-sg.cif",
        arch="mace_mp",
        model=MODEL_PATH,
    )

    optimizer = GeomOpt(single_point.struct, fmax=0.001)
    optimizer.run()

    assert single_point.struct.info["initial_spacegroup"] == "I4/mmm (139)"
    assert single_point.struct.info["final_spacegroup"] == "Fm-3m (225)"


def test_str_optimizer(tmp_path):
    """Test setting optimizer function with string."""
    log_file = tmp_path / "opt.log"

    single_point = SinglePoint(
        struct=DATA_PATH / "NaCl-sg.cif",
        arch="mace_mp",
        model=MODEL_PATH,
    )

    optimizer = GeomOpt(
        single_point.struct,
        fmax=0.001,
        optimizer="FIRE",
        log_kwargs={"filename": log_file},
    )
    optimizer.run()

    assert_log_contains(
        log_file, includes=["Starting geometry optimization", "Using optimizer: FIRE"]
    )


def test_invalid_str_optimizer():
    """Test setting invalid optimizer function with string."""
    single_point = SinglePoint(
        struct=DATA_PATH / "NaCl-sg.cif",
        arch="mace_mp",
        model=MODEL_PATH,
    )

    with pytest.raises(AttributeError):
        GeomOpt(
            single_point.struct,
            fmax=0.001,
            optimizer="test",
        )


def test_str_filter(tmp_path):
    """Test setting filter with string."""
    log_file = tmp_path / "opt.log"

    single_point = SinglePoint(
        struct=DATA_PATH / "NaCl-sg.cif",
        arch="mace_mp",
        model=MODEL_PATH,
    )

    optimizer = GeomOpt(
        single_point.struct,
        fmax=0.001,
        filter_class="UnitCellFilter",
        log_kwargs={"filename": log_file},
    )
    optimizer.run()

    assert_log_contains(
        log_file,
        includes=["Starting geometry optimization", "Using filter: UnitCellFilter"],
    )


def test_invalid_str_filter():
    """Test setting invalid filter with string."""
    single_point = SinglePoint(
        struct=DATA_PATH / "NaCl-sg.cif",
        arch="mace_mp",
        model=MODEL_PATH,
    )

    with pytest.raises(AttributeError):
        GeomOpt(single_point.struct, fmax=0.001, filter_class="test")


def test_invalid_struct():
    """Test setting invalid structure."""
    single_point = SinglePoint(
        struct=DATA_PATH / "benzene-traj.xyz",
        arch="mace_mp",
        model=MODEL_PATH,
    )

    with pytest.raises(NotImplementedError):
        GeomOpt(
            single_point.struct,
            fmax=0.001,
            optimizer="test",
        )
    with pytest.raises(ValueError):
        GeomOpt(
            "structure",
            fmax=0.001,
            optimizer="test",
        )


def test_logging(tmp_path):
    """Test attaching logger to GeomOpt and emissions are saved to info."""
    log_file = tmp_path / "geomopt.log"
    single_point = SinglePoint(
        struct=DATA_PATH / "NaCl.cif",
        arch="mace_mp",
        model=MODEL_PATH,
    )

    assert "emissions" not in single_point.struct.info

    optimizer = GeomOpt(
        single_point.struct,
        log_kwargs={"filename": log_file},
    )
    optimizer.run()

    assert log_file.exists()
    assert "emissions" in single_point.struct.info
    assert single_point.struct.info["emissions"] > 0


def test_write_xyz(tmp_path):
    """Test writing a non-extended xyz file."""
    optimizer = GeomOpt(
        struct=DATA_PATH / "NaCl-deformed.cif",
        arch="mace_mp",
        model=MODEL_PATH,
        fmax=0.1,
    )
    optimizer.run()


@pytest.mark.parametrize(
    "struct", (DATA_PATH / "NaCl.cif", read(DATA_PATH / "NaCl.cif"))
)
def test_missing_arch(struct):
    """Test missing arch."""
    with pytest.raises(ValueError, match="A calculator must be attached"):
        GeomOpt(struct=struct)


def test_traj_new_dir(tmp_path):
    """Test writing trajectory extxyz in new directory via file_prefix."""
    new_dir = tmp_path / "test" / "test"
    traj_path = new_dir / "NaCl-traj.extxyz"

    optimizer = GeomOpt(
        struct=DATA_PATH / "NaCl.cif",
        arch="mace_mp",
        model=MODEL_PATH,
        write_traj=True,
        file_prefix=new_dir / "NaCl",
    )
    optimizer.run()
    assert traj_path.exists()
    traj = read(traj_path, index=":")
    assert len(traj) == 3


def test_traj_kwargs_new_dir(tmp_path):
    """Test writing trajectory in new directory via traj_kwargs."""
    new_dir = tmp_path / "test" / "test"
    traj_path = new_dir / "NaCl-traj.traj"

    optimizer = GeomOpt(
        struct=DATA_PATH / "NaCl.cif",
        arch="mace_mp",
        model=MODEL_PATH,
        write_traj=True,
        file_prefix=tmp_path / "NaCl",
        traj_kwargs={"filename": traj_path},
    )
    optimizer.run()
    assert traj_path.exists()
    traj = read(traj_path, index=":")
    assert len(traj) == 3


def test_deprecation_filter_func(tmp_path):
    """Test FutureWarning raised for model in calc_kwargs."""
    with pytest.warns(FutureWarning, match="`filter_func` has been deprecated"):
        geom_opt = GeomOpt(
            struct=DATA_PATH / "NaCl.cif",
            arch="mace_mp",
            model=MODEL_PATH,
            fmax=0.001,
            filter_func="UnitCellFilter",
        )
    assert isinstance(geom_opt.filtered_struct, UnitCellFilter)
