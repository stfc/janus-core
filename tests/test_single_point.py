"""Test configuration of MLIP calculators."""

import os

from janus_core.single_point import SinglePoint


def test_potential_energy():
    """Test single point energy using MACE calculator."""
    data_path = os.path.join(os.path.dirname(__file__), "data", "benzene.xyz")
    single_point = SinglePoint(system=data_path)

    assert single_point.get_potential_energy() == -76.0605725422795


def test_single_point_kwargs():
    """Test kwargs passed when using MACE calculator for single point energy."""
    data_path = os.path.join(os.path.dirname(__file__), "data", "benzene.xyz")
    model_path = os.path.join(os.path.dirname(__file__), "models", "MACE_small.model")
    single_point = SinglePoint(
        system=data_path, architecture="mace", model_paths=model_path
    )

    assert single_point.get_potential_energy() == -76.0605725422795
