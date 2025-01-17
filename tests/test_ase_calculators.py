"""Test external ASE calculators."""

from __future__ import annotations

from pathlib import Path

from ase.calculators.lj import LennardJones
from ase.io import read
import numpy as np
import pytest

from janus_core.calculations.geom_opt import GeomOpt
from janus_core.calculations.single_point import SinglePoint

DATA_PATH = Path(__file__).parent / "data"


def test_single_point():
    """Test single point calculation using LJ calculator."""
    struct = read(DATA_PATH / "NaCl.cif")
    struct.calc = LennardJones()
    single_point = SinglePoint(struct=struct, properties="energy")
    results = single_point.run()

    assert isinstance(struct.calc, LennardJones)
    assert results["energy"] == pytest.approx(-0.05900093815771919)


def test_geom_opt():
    """Test geometry optimisation using LJ calculator."""
    struct = read(DATA_PATH / "NaCl-deformed.cif")
    struct.calc = LennardJones()
    geom_opt = GeomOpt(struct=struct, fmax=0.01)
    geom_opt.run()

    assert isinstance(geom_opt.struct.calc, LennardJones)
    assert struct.get_forces() == pytest.approx(np.zeros((8, 3)))
