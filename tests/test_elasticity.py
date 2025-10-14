"""Test elasticity calculations."""

from __future__ import annotations

from pathlib import Path

from ase.build import bulk
from pytest import approx

from janus_core.calculations.elasticity import Elasticity
from janus_core.helpers.mlip_calculators import choose_calculator

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"


def test_calc_elasticity(tmp_path):
    """Test calculating elasticity for Aluminium."""
    struct = bulk("Al", crystalstructure="fcc")
    struct.calc = choose_calculator(arch="mace_mp", model=MODEL_PATH)

    elasticity = Elasticity(struct, file_prefix=tmp_path / "elasticity")
    elastic_tensor = elasticity.run()

    assert elastic_tensor.k_reuss == approx(80.34637735135381, rel=1e-3)
