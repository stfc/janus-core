"""Test configuration of MLIP calculators."""

from pathlib import Path

import pytest

from janus_core.single_point import SinglePoint


def test_potential_energy():
    """Test single point energy using MACE calculator."""
    data_path = Path(Path(__file__).parent, "data", "benzene.xyz")
    single_point = SinglePoint(system=data_path)

    assert single_point.run_single_point("energy")["energy"] == -76.0605725422795


def test_single_point_kwargs():
    """Test kwargs passed when using MACE calculator for single point energy."""

    data_path = Path(Path(__file__).parent, "data", "benzene.xyz")
    model_path = Path(Path(__file__).parent, "models", "MACE_small.model")
    single_point = SinglePoint(
        system=data_path, architecture="mace", model_paths=model_path
    )

    assert single_point.run_single_point(["energy"])["energy"] == -76.0605725422795


def test_single_point_forces():
    """Test single point forces using MACE calculator."""
    data_path = Path(Path(__file__).parent, "data", "benzene.xyz")
    model_path = Path(Path(__file__).parent, "models", "MACE_small.model")
    single_point = SinglePoint(
        system=data_path, architecture="mace", model_paths=model_path
    )

    assert single_point.run_single_point(["forces"])["forces"][0, 1] == pytest.approx(
        -0.0360169762840179
    )


def test_single_point_stress():
    """Test single point stress using MACE calculator."""
    data_path = Path(Path(__file__).parent, "data", "NaCl.cif")
    model_path = Path(Path(__file__).parent, "models", "MACE_small.model")
    single_point = SinglePoint(
        system=data_path, architecture="mace", model_paths=model_path
    )

    assert single_point.run_single_point(["stress"])["stress"][0] == pytest.approx(
        -0.004783275999053424
    )


def test_single_point_none():
    """Test single point stress using MACE calculator."""
    data_path = Path(Path(__file__).parent, "data", "NaCl.cif")
    model_path = Path(Path(__file__).parent, "models", "MACE_small.model")
    single_point = SinglePoint(
        system=data_path, architecture="mace", model_paths=model_path
    )

    results = single_point.run_single_point()
    for prop in ["energy", "forces", "stress"]:
        assert prop in results


def test_single_point_traj():
    """Test single point stress using MACE calculator."""
    data_path = Path(Path(__file__).parent, "data", "benzene-traj.xyz")
    model_path = Path(Path(__file__).parent, "models", "MACE_small.model")
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
