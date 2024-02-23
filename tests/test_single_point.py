"""Test configuration of MLIP calculators."""

from pathlib import Path

import pytest

from janus_core.single_point import SinglePoint

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"

test_data = [
    (DATA_PATH / "benzene.xyz", -76.0605725422795, "energy", "energy", {}),
    (
        DATA_PATH / "benzene.xyz",
        -76.06057739257812,
        ["energy"],
        "energy",
        {"default_dtype": "float32"},
    ),
    (
        DATA_PATH / "benzene.xyz",
        -0.0360169762840179,
        ["forces"],
        "forces",
        {"idx": [0, 1]},
    ),
    (DATA_PATH / "NaCl.cif", -0.004783275999053424, ["stress"], "stress", {"idx": [0]}),
]


@pytest.mark.parametrize("system, expected, properties, prop_key, kwargs", test_data)
def test_potential_energy(system, expected, properties, prop_key, kwargs):
    """Test single point energy using MACE calculators."""
    single_point = SinglePoint(
        system=system, architecture="mace", model_paths=MODEL_PATH, **kwargs
    )
    results = single_point.run_single_point(properties)[prop_key]

    # Check correct values returned
    idx = kwargs.pop("idx", None)
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
    data_path = DATA_PATH / "NaCl.cif"
    model_path = MODEL_PATH
    single_point = SinglePoint(
        system=data_path, architecture="mace", model_paths=model_path
    )

    results = single_point.run_single_point()
    for prop in ["energy", "forces", "stress"]:
        assert prop in results


def test_single_point_traj():
    """Test single point stress using MACE calculator."""
    data_path = DATA_PATH / "benzene-traj.xyz"
    model_path = MODEL_PATH
    single_point = SinglePoint(
        system=data_path,
        architecture="mace",
        model_paths=model_path,
        read_kwargs={"index": ":"},
    )

    assert len(single_point.sys) == 2
    results = single_point.run_single_point("energy")
    assert results["energy"][0] == pytest.approx(-76.0605725422795)
    assert results["energy"][1] == pytest.approx(-74.80419118083256)
