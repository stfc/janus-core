"""Test elasticity calculations."""

from __future__ import annotations

from pathlib import Path

from ase.build import bulk
import numpy as np
from pytest import approx

from janus_core.calculations.elasticity import Elasticity
from janus_core.helpers.mlip_calculators import choose_calculator
from tests.utils import assert_log_contains

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"


def test_calc_elasticity(tmp_path):
    """Test calculating elasticity for Aluminium."""
    elasticity_path = tmp_path / "elasticity-elastic_tensor.dat"
    log_file = tmp_path / "elasticity.log"

    struct = bulk("Al", crystalstructure="fcc")
    struct.calc = choose_calculator(arch="mace_mp", model=MODEL_PATH)

    elasticity = Elasticity(
        struct, file_prefix=tmp_path / "elasticity", log_kwargs={"filename": log_file}
    )
    elastic_tensor = elasticity.run()

    assert elastic_tensor.k_reuss == approx(80.34637735135381, rel=1e-3)

    # Check geometry optimization run by default
    assert_log_contains(
        log_file,
        includes=["Using filter", "Using optimizer", "Starting geometry optimization"],
    )

    assert elasticity_path.exists()
    written_elasticity = np.loadtxt(elasticity_path)
    # Defaulting to 6x6 Voigt notation
    assert len(written_elasticity) == 45

    for i, prop in enumerate(
        (
            "k_reuss",
            "k_voigt",
            "k_vrh",
            "g_reuss",
            "g_voigt",
            "g_vrh",
            "y_mod",
            "universal_anisotropy",
            "homogeneous_poisson",
        )
    ):
        assert elastic_tensor.property_dict[prop] == approx(
            written_elasticity[i], rel=1e-3
        )

    assert written_elasticity[9:] == approx(elastic_tensor.voigt.flatten(), rel=1e-3)


def test_no_optimize_no_write_voigt(tmp_path):
    """Test calculating elasticity for Aluminium without optimization."""
    elasticity_path = tmp_path / "elasticity-elastic_tensor.dat"
    log_file = tmp_path / "elasticity.log"

    struct = bulk("Al", crystalstructure="fcc")
    struct.calc = choose_calculator(arch="mace_mp", model=MODEL_PATH)

    elasticity = Elasticity(
        struct,
        file_prefix=tmp_path / "elasticity",
        log_kwargs={"filename": log_file},
        minimize=False,
        write_voigt=False,
    )
    elastic_tensor = elasticity.run()

    assert elastic_tensor.k_reuss == approx(80.34637735135381, rel=1e-3)

    # Check geometry optimization run by default
    assert_log_contains(
        log_file,
        excludes=["Using filter", "Using optimizer", "Starting geometry optimization"],
    )

    assert elasticity_path.exists()
    written_elasticity = np.loadtxt(elasticity_path)
    # Selected to full tensor 3x3x3x3
    assert len(written_elasticity) == 90

    assert written_elasticity[0] == approx(elastic_tensor.k_reuss, rel=1e-3)

    for i, prop in enumerate(
        (
            "k_reuss",
            "k_voigt",
            "k_vrh",
            "g_reuss",
            "g_voigt",
            "g_vrh",
            "y_mod",
            "universal_anisotropy",
            "homogeneous_poisson",
        )
    ):
        assert elastic_tensor.property_dict[prop] == approx(
            written_elasticity[i], rel=1e-3
        )

    assert written_elasticity[9:] == approx(elastic_tensor.flatten(), rel=1e-3)
