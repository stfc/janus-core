"""Test configuration of MLIP calculators."""

from pathlib import Path

from ase.io import read
from numpy import isfinite
import pytest

from janus_core.single_point import SinglePoint

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"

test_data = [
    (DATA_PATH / "benzene.xyz", -76.0605725422795, "energy", "energy", {}, None),
    (
        DATA_PATH / "benzene.xyz",
        -76.06057739257812,
        ["energy"],
        "energy",
        {"default_dtype": "float32"},
        None,
    ),
    (DATA_PATH / "benzene.xyz", -0.0360169762840179, ["forces"], "forces", {}, [0, 1]),
    (DATA_PATH / "NaCl.cif", -0.004783275999053424, ["stress"], "stress", {}, [0]),
]


@pytest.mark.parametrize(
    "struct_path, expected, properties, prop_key, calc_kwargs, idx", test_data
)
def test_potential_energy(
    struct_path, expected, properties, prop_key, calc_kwargs, idx
):
    """Test single point energy using MACE calculators."""
    calc_kwargs["model_paths"] = MODEL_PATH
    single_point = SinglePoint(
        struct_path=struct_path, architecture="mace", calc_kwargs=calc_kwargs
    )
    results = single_point.run_single_point(properties)[prop_key]

    # Check correct values returned
    if idx is not None:
        if len(idx) == 1:
            assert results[idx[0]] == pytest.approx(expected)
        elif len(idx) == 2:
            assert results[idx[0], idx[1]] == pytest.approx(expected)
        else:
            raise ValueError(f"Invalid index: {idx}")
    else:
        assert results == pytest.approx(expected)


def test_single_point_none():
    """Test single point stress using MACE calculator."""
    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model_paths": MODEL_PATH},
    )

    results = single_point.run_single_point()
    for prop in ["energy", "forces", "stress"]:
        assert prop in results


def test_single_point_traj():
    """Test single point stress using MACE calculator."""
    single_point = SinglePoint(
        struct_path=DATA_PATH / "benzene-traj.xyz",
        architecture="mace",
        read_kwargs={"index": ":"},
        calc_kwargs={"model_paths": MODEL_PATH},
    )

    assert len(single_point.struct) == 2
    results = single_point.run_single_point("energy")
    assert results["energy"][0] == pytest.approx(-76.0605725422795)
    assert results["energy"][1] == pytest.approx(-74.80419118083256)


def test_single_point_write():
    """Test writing singlepoint results."""
    data_path = DATA_PATH / "NaCl.cif"
    results_path = Path(".").absolute() / "NaCl-results.xyz"
    assert not Path(results_path).exists()

    single_point = SinglePoint(
        struct_path=data_path,
        architecture="mace",
        calc_kwargs={"model_paths": MODEL_PATH},
    )
    assert "forces" not in single_point.struct.arrays

    single_point.run_single_point(write_results=True)
    atoms = read(results_path)
    assert atoms.get_potential_energy() is not None
    assert "forces" in atoms.arrays

    Path(results_path).unlink()


def test_single_point_write_kwargs(tmp_path):
    """Test passing write_kwargs to singlepoint results."""
    data_path = DATA_PATH / "NaCl.cif"
    results_path = tmp_path / "NaCl.xyz"

    single_point = SinglePoint(
        struct_path=data_path,
        architecture="mace",
        calc_kwargs={"model_paths": MODEL_PATH},
    )
    assert "forces" not in single_point.struct.arrays

    single_point.run_single_point(
        write_results=True, write_kwargs={"filename": results_path}
    )
    atoms = read(results_path)
    assert atoms.get_potential_energy() is not None
    assert "forces" in atoms.arrays


def test_single_point_write_nan(tmp_path):
    """Test non-finite singlepoint results removed."""
    data_path = DATA_PATH / "H2O.cif"
    results_path = tmp_path / "H2O.xyz"
    single_point = SinglePoint(
        struct_path=data_path,
        architecture="mace",
        calc_kwargs={"model_paths": MODEL_PATH},
    )

    assert isfinite(single_point.run_single_point("energy")["energy"]).all()
    with pytest.raises(ValueError):
        single_point.run_single_point("stress")

    single_point.run_single_point(
        write_results=True, write_kwargs={"filename": results_path}
    )
    atoms = read(results_path)
    assert atoms.get_potential_energy() is not None
    assert "forces" in atoms.calc.results
    assert "stress" not in atoms.calc.results


def test_invalid_prop():
    """Test invalid property request."""
    single_point = SinglePoint(
        struct_path=DATA_PATH / "H2O.cif",
        architecture="mace",
        calc_kwargs={"model_paths": MODEL_PATH},
    )
    with pytest.raises(NotImplementedError):
        single_point.run_single_point("invalid")


def test_atoms():
    """Test passing ASE Atoms structure."""
    struct = read(DATA_PATH / "NaCl.cif")
    single_point = SinglePoint(
        struct=struct,
        struct_name="NaCl",
        architecture="mace",
        calc_kwargs={"model_paths": MODEL_PATH},
    )
    assert single_point.struct_name == "NaCl"
    assert single_point.run_single_point("energy")["energy"] < 0


def test_default_atoms_name():
    """Test default structure name when passing ASE Atoms structure."""
    struct = read(DATA_PATH / "NaCl.cif")
    single_point = SinglePoint(
        struct=struct,
        architecture="mace",
        calc_kwargs={"model_paths": MODEL_PATH},
    )
    assert single_point.struct_name == "struct"


def test_default_path_name():
    """Test default structure name when passing structure path."""
    struct_path = DATA_PATH / "NaCl.cif"
    single_point = SinglePoint(
        struct_path=struct_path,
        architecture="mace",
        calc_kwargs={"model_paths": MODEL_PATH},
    )
    assert single_point.struct_name == "NaCl"


def test_path_specify_name():
    """Test specifying structure name with structure path."""
    struct_path = DATA_PATH / "NaCl.cif"
    single_point = SinglePoint(
        struct_path=struct_path,
        struct_name="example_name",
        architecture="mace",
        calc_kwargs={"model_paths": MODEL_PATH},
    )
    assert single_point.struct_name == "example_name"


def test_atoms_and_path():
    """Test passing ASE Atoms structure and structure path togther."""
    struct = read(DATA_PATH / "NaCl.cif")
    struct_path = DATA_PATH / "NaCl.cif"
    with pytest.raises(ValueError):
        SinglePoint(
            struct=struct,
            struct_path=struct_path,
            architecture="mace",
            calc_kwargs={"model_paths": MODEL_PATH},
        )


def test_no_atoms_or_path():
    """Test passing neither ASE Atoms structure nor structure path."""
    with pytest.raises(ValueError):
        SinglePoint(
            architecture="mace",
            calc_kwargs={"model_paths": MODEL_PATH},
        )
