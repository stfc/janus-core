"""Test the Correlator."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ase import Atoms
from ase.io import read
from ase.units import GPa
import numpy as np
from pytest import approx
from typer.testing import CliRunner
from yaml import Loader, load

from janus_core.calculations.md import NVE
from janus_core.calculations.single_point import SinglePoint
from janus_core.processing.correlator import Correlator
from janus_core.processing.observables import Stress

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"
runner = CliRunner()


def correlate(
    x: Iterable[float], y: Iterable[float], *, fft: bool = True
) -> Iterable[float]:
    """Direct correlation of x and y. If fft uses np.correlate in full mode."""
    n = min(len(x), len(y))
    if fft:
        cor = np.correlate(x, y, "full")
        cor = cor[len(cor) // 2 :]
        return cor / (n - np.arange(n))
    cor = np.zeros(n)
    for j in range(n):
        for i in range(n - j):
            cor[j] += x[i] * y[i + j]
        cor[j] /= n - j
    return cor


def test_setup():
    """Test initial values."""
    cor = Correlator(blocks=1, points=100, averaging=2)
    correlation, lags = cor.get()
    assert len(correlation) == len(lags)
    assert len(correlation) == 0


def test_correlation():
    """Test Correlator against np.correlate."""
    points = 100
    cor = Correlator(blocks=1, points=points, averaging=1)
    signal = np.exp(-np.linspace(0.0, 1.0, points))
    for val in signal:
        cor.update(val, val)
    correlation, lags = cor.get()

    direct = correlate(signal, signal, fft=False)
    fft = correlate(signal, signal, fft=True)

    assert len(correlation) == len(lags)
    assert all(lags == range(points))
    assert direct == approx(correlation, rel=1e-10)
    assert fft == approx(correlation, rel=1e-10)


def test_md_correlations(tmp_path):
    """Test correlations as part of MD cycle."""
    file_prefix = tmp_path / "Cl4Na4-nve-T300.0"
    traj_path = tmp_path / "Cl4Na4-nve-T300.0-traj.extxyz"
    cor_path = tmp_path / "Cl4Na4-nve-T300.0-cor.dat"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    def user_observable_a(atoms: Atoms, kappa, **kwargs) -> float:
        """User specified getter for correlation."""
        return (
            kwargs["gamma"]
            * kappa
            * atoms.get_stress(include_ideal_gas=True, voigt=True)[-1]
            / GPa
        )

    nve = NVE(
        struct=single_point.struct,
        temp=300.0,
        steps=10,
        seed=1,
        traj_every=1,
        stats_every=1,
        file_prefix=file_prefix,
        correlation_kwargs=[
            {
                "a": (user_observable_a, (2,), {"gamma": 2}),
                "b": Stress("xy"),
                "name": "user_correlation",
                "blocks": 1,
                "points": 11,
                "averaging": 1,
                "update_frequency": 1,
            },
            {
                "a": Stress("xy"),
                "b": Stress("xy"),
                "name": "stress_xy_auto_cor",
                "blocks": 1,
                "points": 11,
                "averaging": 1,
                "update_frequency": 1,
            },
        ],
        write_kwargs={"invalidate_calc": False},
    )
    nve.run()

    pxy = [
        atom.get_stress(include_ideal_gas=True, voigt=False).flatten()[1] / GPa
        for atom in read(traj_path, index=":")
    ]

    assert cor_path.exists()
    with open(cor_path, encoding="utf8") as in_file:
        cor = load(in_file, Loader=Loader)
    assert len(cor) == 2
    assert "user_correlation" in cor
    assert "stress_xy_auto_cor" in cor

    stress_cor = cor["stress_xy_auto_cor"]
    value, lags = stress_cor["value"], stress_cor["lags"]
    assert len(value) == len(lags) == 11

    direct = correlate(pxy, pxy, fft=False)
    # input data differs due to i/o, error is expected 1e-5
    assert direct == approx(value, rel=1e-5)

    user_cor = cor["user_correlation"]
    value, lags = user_cor["value"], stress_cor["lags"]
    assert len(value) == len(lags) == 11

    direct = correlate([v * 4.0 for v in pxy], pxy, fft=False)
    # input data differs due to i/o, error is expected 1e-5
    assert direct == approx(value, rel=1e-5)
