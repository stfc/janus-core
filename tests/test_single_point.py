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
    "structure, expected, properties, prop_key, calc_kwargs, idx", test_data
)
def test_potential_energy(structure, expected, properties, prop_key, calc_kwargs, idx):
    """Test single point energy using MACE calculators."""
    calc_kwargs["model_paths"] = MODEL_PATH
    single_point = SinglePoint(
        structure=structure, architecture="mace", calc_kwargs=calc_kwargs
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
        structure=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model_paths": MODEL_PATH},
    )

    results = single_point.run_single_point()
    for prop in ["energy", "forces", "stress"]:
        assert prop in results


def test_single_point_traj():
    """Test single point stress using MACE calculator."""
    single_point = SinglePoint(
        structure=DATA_PATH / "benzene-traj.xyz",
        architecture="mace",
        read_kwargs={"index": ":"},
        calc_kwargs={"model_paths": MODEL_PATH},
    )

    assert len(single_point.struct) == 2
    results = single_point.run_single_point("energy")
    assert results["energy"][0] == pytest.approx(-76.0605725422795)
    assert results["energy"][1] == pytest.approx(-74.80419118083256)


def test_single_point_write(tmp_path):
    """Test writing singlepoint results."""
    data_path = DATA_PATH / "NaCl.cif"
    results_path = tmp_path / "NaCl.xyz"
    single_point = SinglePoint(
        structure=data_path,
        architecture="mace",
        calc_kwargs={"model_paths": MODEL_PATH},
    )

    assert "forces" not in single_point.struct.arrays
    single_point.run_single_point(write_kwargs={"filename": results_path})

    atoms = read(results_path)
    assert atoms.get_potential_energy() is not None
    assert "forces" in atoms.arrays


def test_single_point_write_missing():
    """Test writing singlepoint results."""
    data_path = DATA_PATH / "NaCl.cif"
    single_point = SinglePoint(
        structure=data_path,
        architecture="mace",
        calc_kwargs={"model_paths": MODEL_PATH},
    )
    with pytest.raises(ValueError):
        single_point.run_single_point(write_kwargs={"file": "file.xyz"})


def test_single_point_write_nan(tmp_path):
    """Test non-finite singlepoint results removed."""
    data_path = DATA_PATH / "H2O.cif"
    results_path = tmp_path / "H2O.xyz"
    single_point = SinglePoint(
        structure=data_path,
        architecture="mace",
        calc_kwargs={"model_paths": MODEL_PATH},
    )

    assert isfinite(single_point.run_single_point("energy")["energy"]).all()
    assert not isfinite(single_point.run_single_point("stress")["stress"]).all()

    single_point.run_single_point(write_kwargs={"filename": results_path})
    atoms = read(results_path)
    assert atoms.get_potential_energy() is not None
    assert "forces" in atoms.calc.results
    assert "stress" not in atoms.calc.results
