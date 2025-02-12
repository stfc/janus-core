"""Test post processing."""

from __future__ import annotations

from pathlib import Path

from ase.io import read
import numpy as np
import pytest
from pytest import approx
from typer.testing import CliRunner

from janus_core.calculations.md import NVE
from janus_core.calculations.single_point import SinglePoint
from janus_core.cli.janus import app
from janus_core.processing import post_process

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"
runner = CliRunner()


def test_md_pp(tmp_path):
    """Test post-processing as part of MD cycle."""
    file_prefix = tmp_path / "Cl4Na4-nve-T300.0"
    traj_path = tmp_path / "Cl4Na4-nve-T300.0-traj.extxyz"
    stats_path = tmp_path / "Cl4Na4-nve-T300.0-stats.dat"
    rdf_path = tmp_path / "Cl4Na4-nve-T300.0-rdf.dat"
    vaf_path = tmp_path / "Cl4Na4-nve-T300.0-vaf.dat"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    nve = NVE(
        struct=single_point.struct,
        temp=300.0,
        steps=10,
        traj_every=2,
        stats_every=15,
        file_prefix=file_prefix,
        post_process_kwargs={
            "rdf_compute": True,
            "rdf_rmax": 2.5,
            "vaf_compute": True,
        },
    )

    try:
        nve.run()

        assert rdf_path.exists()
        rdf = np.loadtxt(rdf_path)
        assert len(rdf) == 50

        # Cell too small to really compute RDF
        assert np.all(rdf[:, 1] == 0)

        assert vaf_path.exists()

    finally:
        traj_path.unlink(missing_ok=True)
        stats_path.unlink(missing_ok=True)
        rdf_path.unlink(missing_ok=True)
        vaf_path.unlink(missing_ok=True)


def test_md_pp_cli(tmp_path):
    """Test all MD simulations are able to run."""
    file_prefix = tmp_path / "nve-T300"
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    rdf_path = tmp_path / "nve-T300-rdf.dat"
    vaf_na_path = Path("vaf_na.dat")
    vaf_cl_path = Path("vaf_cl.dat")
    result = runner.invoke(
        app,
        [
            "md",
            "--ensemble",
            "nve",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--file-prefix",
            file_prefix,
            "--steps",
            10,
            "--traj-every",
            2,
            "--log",
            log_path,
            "--summary",
            summary_path,
            "--post-process-kwargs",
            """{'vaf_compute': True,
              'vaf_atoms': (('Na',),('Cl',)),
              'vaf_output_files': ('vaf_na.dat', 'vaf_cl.dat'),
              'rdf_compute': True,
              'rdf_rmax': 2.5}""",
        ],
    )

    assert result.exit_code == 0

    assert rdf_path.exists()
    rdf = np.loadtxt(rdf_path)
    assert len(rdf) == 50

    # Cell too small to really compute RDF
    assert np.all(rdf[:, 1] == 0)

    assert vaf_na_path.exists()
    assert vaf_cl_path.exists()


def test_rdf():
    """Test computation of RDF."""
    data = read(DATA_PATH / "benzene.xyz")
    rdf = post_process.compute_rdf(data, index=0, rmax=5.0, nbins=100)

    assert isinstance(rdf, tuple)
    assert isinstance(rdf[0], np.ndarray)

    expected_peaks = np.asarray(
        (
            1.075,
            1.375,
            2.175,
            2.425,
            2.475,
            2.775,
            3.425,
            3.875,
            4.275,
            4.975,
        )
    )
    peaks = np.where(rdf[1] > 0.0)
    assert (np.isclose(expected_peaks, rdf[0][peaks])).all()


def test_rdf_by_elements():
    """Test the by_elements method of compute rdf."""
    data = read(DATA_PATH / "benzene.xyz")

    rdfs = post_process.compute_rdf(
        data,
        index=0,
        rmax=5.0,
        nbins=100,
        by_elements=True,
    )

    assert isinstance(rdfs, dict)
    assert isinstance(rdfs[("C", "C")], tuple)
    assert isinstance(rdfs[("C", "C")][0], np.ndarray)

    expected_peaks = {
        ("C", "C"): (1.375, 2.425, 2.775),
        ("C", "H"): (
            1.075,
            1.375,
            2.175,
            2.425,
            2.475,
            2.775,
            3.425,
            3.875,
            4.275,
            4.975,
        ),
        ("H", "H"): (2.475, 4.275, 4.975),
    }

    for element, rdf in rdfs.items():
        peaks = np.where(rdf[1] > 0.0)
        assert (np.isclose(expected_peaks[element], rdf[0][peaks])).all()


def test_vaf(tmp_path):
    """Test vaf will run."""
    vaf_names = ("vaf-lj-3-4.dat", "vaf-lj-1-2-3.dat")
    vaf_filter = ((3, 4), (1, 2, 3))

    data = read(DATA_PATH / "lj-traj.xyz", index=":")
    lags, vaf = post_process.compute_vaf(data)
    expected = np.loadtxt(DATA_PATH / "vaf-lj.dat")

    assert isinstance(vaf, list)
    assert len(vaf) == 1
    assert isinstance(vaf[0], np.ndarray)
    assert vaf[0] == approx(expected, rel=1e-9)

    lags, vaf = post_process.compute_vaf(data, fft=True)

    assert isinstance(vaf, list)
    assert len(vaf) == 1
    assert isinstance(vaf[0], np.ndarray)

    lags, vaf = post_process.compute_vaf(
        data, atoms_filter=vaf_filter, filenames=[tmp_path / name for name in vaf_names]
    )

    assert isinstance(vaf, list)
    assert len(vaf) == 2
    assert isinstance(vaf[0], np.ndarray)

    for i, name in enumerate(vaf_names):
        assert (tmp_path / name).exists()
        expected = np.loadtxt(DATA_PATH / name)
        written = np.loadtxt(tmp_path / name)
        w_lag, w_vaf = written[:, 0], written[:, 1]

        assert vaf[i] == approx(expected, rel=1e-9)
        assert lags == approx(w_lag, rel=1e-9)
        assert vaf[i] == approx(w_vaf, rel=1e-9)


def test_vaf_by_symbols(tmp_path):
    """Test vaf using element symbols."""
    vaf_names = ("vaf-Na-by-indices.dat", "vaf-Na-by-element.dat")
    vaf_filter = ((0, 2, 4, 6), ("Na",))

    data = read(DATA_PATH / "NaCl-traj.xyz", index=":")
    lags, vaf = post_process.compute_vaf(
        data, atoms_filter=vaf_filter, filenames=[tmp_path / name for name in vaf_names]
    )
    expected = np.loadtxt(tmp_path / "vaf-Na-by-indices.dat")
    actual = np.loadtxt(tmp_path / "vaf-Na-by-element.dat")

    assert expected == approx(actual, rel=1e-9)


def test_vaf_invalid_symbols(tmp_path):
    """Test vaf using invalid element symbols."""
    data = read(DATA_PATH / "NaCl-traj.xyz", index=":")
    with pytest.raises(ValueError):
        # "C" is not an atom in data.
        post_process.compute_vaf(data, atoms_filter=(("C",)))
    with pytest.raises(ValueError):
        # str is also iterable.
        post_process.compute_vaf(data, atoms_filter="NOT AN ATOM")
