"""Test configuration of MLIP calculators."""

import os

import pytest

from janus_core.single_point import SinglePoint


def test_potential_energy():
    """Test single point energy using MACE calculator."""
    data_path = os.path.join(os.path.dirname(__file__), "data", "benzene.xyz")
    single_point = SinglePoint(system=data_path)

    assert single_point.run_single_point("energy")["energy"] == -76.0605725422795


def test_single_point_kwargs():
    """Test kwargs passed when using MACE calculator for single point energy."""
    data_path = os.path.join(os.path.dirname(__file__), "data", "benzene.xyz")
    model_path = os.path.join(os.path.dirname(__file__), "models", "MACE_small.model")
    single_point = SinglePoint(
        system=data_path, architecture="mace", model_paths=model_path
    )

    assert single_point.run_single_point(["energy"])["energy"] == -76.0605725422795


def test_single_point_forces():
    """Test single point forces using MACE calculator."""
    data_path = os.path.join(os.path.dirname(__file__), "data", "benzene.xyz")
    model_path = os.path.join(os.path.dirname(__file__), "models", "MACE_small.model")
    single_point = SinglePoint(
        system=data_path, architecture="mace", model_paths=model_path
    )

    assert single_point.run_single_point(["forces"])["forces"][0, 1] == pytest.approx(
        -0.0360169762840179
    )


def test_single_point_stress():
    """Test single point stress using MACE calculator."""
    data_path = os.path.join(os.path.dirname(__file__), "data", "UiO66.cif")
    model_path = os.path.join(os.path.dirname(__file__), "models", "MACE_small.model")
    single_point = SinglePoint(
        system=data_path, architecture="mace", model_paths=model_path
    )

    assert single_point.run_single_point(["stress"])["stress"][0] == pytest.approx(
        -0.00415290516
    )


def test_single_point_none():
    """Test single point stress using MACE calculator."""
    data_path = os.path.join(os.path.dirname(__file__), "data", "UiO66.cif")
    model_path = os.path.join(os.path.dirname(__file__), "models", "MACE_small.model")
    single_point = SinglePoint(
        system=data_path, architecture="mace", model_paths=model_path
    )

    results = single_point.run_single_point()
    for prop in ["energy", "forces", "stress"]:
        assert prop in results

