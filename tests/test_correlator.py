"""Test the Correlator"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from pytest import approx
from typer.testing import CliRunner
from yaml import Loader, load

from janus_core.calculations.md import NVE
from janus_core.calculations.single_point import SinglePoint
from janus_core.helpers.correlator import Correlator
from janus_core.helpers.stats import Stats

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"
runner = CliRunner()


# pylint: disable=invalid-name
def correlate(
    x: Iterable[float], y: Iterable[float], *, fft: bool = True
) -> Iterable[float]:
    """
    Direct correlation of x and y. If fft uses np.correlate in full mode.
    """
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
    """Test initial values"""
    cor = Correlator(blocks=1, points=100, window=2)
    correlation, lags = cor.get()
    assert len(correlation) == len(lags)
    assert len(correlation) == 0


def test_correlation():
    """Test Correlator against np.correlate"""
    points = 100
    cor = Correlator(blocks=1, points=points, window=1)
    signal = np.exp(-np.linspace(0.0, 1.0, points))
    for val in signal:
        cor.update(val, val)
    correlation, lags = cor.get()

    direct = correlate(signal, signal, fft=False)
    fft = correlate(signal, signal, fft=True)

    assert len(correlation) == len(lags)
    assert all(lags == range(points))
    assert np.mean(direct - correlation) == approx(0)
    assert np.mean(fft - correlation) == approx(0)


def test_md_correlations(tmp_path):
    """Test correlations as part of MD cycle."""
    file_prefix = tmp_path / "Cl4Na4-nve-T300.0"
    traj_path = tmp_path / "Cl4Na4-nve-T300.0-traj.xyz"
    stats_path = tmp_path / "Cl4Na4-nve-T300.0-stats.dat"
    cor_path = tmp_path / "Cl4Na4-nve-T300.0-cor.dat"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    nve = NVE(
        struct=single_point.struct,
        temp=300.0,
        steps=10,
        traj_every=2,
        stats_every=1,
        file_prefix=file_prefix,
        correlation_kwargs={
            "correlations": [("s_xy", "s_xy")],
            "correlation_parameters": [(1, 10, 1, 1)],
        },
    )

    try:
        nve.run()
        assert cor_path.exists()
        with open(cor_path, encoding="utf8") as in_file:
            cor = load(in_file, Loader=Loader)
        assert "correlations" in cor
        assert len(cor["correlations"]) == 1
        assert "stress_xy-stress_xy" in cor["correlations"][0]
        stress_cor = cor["correlations"][0]["stress_xy-stress_xy"]
        value, lags = stress_cor["value"], stress_cor["lags"]
        assert len(value) == len(lags) == 10

        stats = Stats(stats_path)
        pxy = stats["Pxy"][0:-1]
        direct = correlate(pxy, pxy, fft=False)
        assert np.mean(direct - value) == approx(0)
    finally:
        traj_path.unlink(missing_ok=True)
        stats_path.unlink(missing_ok=True)
        cor_path.unlink(missing_ok=True)
