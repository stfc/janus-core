"""Test geometry optimization."""
# ruff: noqa: N802, N803

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import pytest

from janus_core.calculations.neb import NEB
from janus_core.calculations.single_point import SinglePoint
from janus_core.helpers.mlip_calculators import choose_calculator

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"


@pytest.fixture
def LFPO():
    """Read optimized LiFePO4 supercell structure."""
    return read(DATA_PATH / "LiFePO4_supercell-opt.cif")


@pytest.fixture
def LFPO_start_b(LFPO):
    """Set up initial structure for LiFePO4 NEB."""
    struct = LFPO.copy()
    del struct[5]
    return struct


@pytest.fixture
def LFPO_end_b(LFPO):
    """Set up final structure for LiFePO4 NEB."""
    struct = LFPO.copy()
    del struct[11]
    return struct


def test_neb(tmp_path, LFPO_start_b, LFPO_end_b):
    """Test NEB."""
    # Write initial and final structures
    init_struct_path = tmp_path / "init_struct.cif"
    final_struct_path = tmp_path / "final_struct.cif"
    write(init_struct_path, images=LFPO_start_b)
    write(final_struct_path, images=LFPO_end_b)

    neb = NEB(
        init_struct_path=init_struct_path,
        final_struct_path=final_struct_path,
        model_path=MODEL_PATH,
        n_images=5,
    )
    neb.run()

    assert len(neb.images) == 7
    assert all(key in neb.results for key in ("barrier", "delta_E", "max_force"))
    assert neb.results["barrier"] == pytest.approx(7817.150960456944)
    assert neb.results["delta_E"] == pytest.approx(78.34034136421758)
    assert neb.results["max_force"] == pytest.approx(148695.846153840771)


def test_neb_pymatgen(tmp_path, LFPO_start_b, LFPO_end_b):
    """Test pymatgen interpolation."""
    file_prefix = tmp_path / "LFPO"
    single_point_start = SinglePoint(
        struct=LFPO_start_b,
        arch="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    single_point_end = SinglePoint(
        struct=LFPO_end_b,
        arch="mace_mp",
        model_path=MODEL_PATH,
    )

    neb = NEB(
        init_struct=single_point_start.struct,
        final_struct=single_point_end.struct,
        model_path=MODEL_PATH,
        n_images=5,
        interpolator="pymatgen",
        fmax=4,
        file_prefix=file_prefix,
    )
    neb.run()

    assert len(neb.images) == 7
    assert all(key in neb.results for key in ("barrier", "delta_E", "max_force"))
    assert neb.results["barrier"] == pytest.approx(0.7536690819713392)
    assert neb.results["delta_E"] == pytest.approx(5.363118589229998e-07)
    assert neb.results["max_force"] == pytest.approx(2.533305796378263)


def test_set_calc(tmp_path, LFPO_start_b, LFPO_end_b):
    """Test setting the calculators explicitly."""
    file_prefix = tmp_path / "LFPO"
    start_struct = LFPO_start_b
    end_struct = LFPO_end_b
    start_struct.calc = choose_calculator(arch="mace_mp", model_path=MODEL_PATH)
    end_struct.calc = choose_calculator(arch="mace_mp", model_path=MODEL_PATH)

    neb = NEB(
        init_struct=start_struct,
        final_struct=end_struct,
        n_images=5,
        interpolator="ase",
        fmax=4,
        file_prefix=file_prefix,
    )
    neb.run()

    # Same as results from test_neb
    assert len(neb.images) == 7
    assert all(key in neb.results for key in ("barrier", "delta_E", "max_force"))
    assert neb.results["barrier"] == pytest.approx(7817.150960456944)
    assert neb.results["delta_E"] == pytest.approx(78.34034136421758)
    assert neb.results["max_force"] == pytest.approx(148695.846153840771)


def test_neb_functions(tmp_path, LFPO_start_b, LFPO_end_b):
    """Test individual NEB functions."""
    file_prefix = tmp_path / "LFPO"

    neb = NEB(
        init_struct=LFPO_start_b,
        final_struct=LFPO_end_b,
        arch="mace",
        model_path=MODEL_PATH,
        n_images=5,
        interpolator="ase",
        file_prefix=file_prefix,
    )
    neb.interpolate()
    neb.optimize()
    neb.run_nebtools()

    assert len(neb.images) == 7
    assert all(key in neb.results for key in ("barrier", "delta_E", "max_force"))
    assert neb.results["barrier"] == pytest.approx(7817.150960456944)
    assert neb.results["delta_E"] == pytest.approx(78.34034136421758)
    assert neb.results["max_force"] == pytest.approx(148695.846153840771)


def test_neb_plot(tmp_path):
    """Test plotting NEB before running NEBTools."""
    file_prefix = tmp_path / "LFPO"

    neb = NEB(
        band_path=DATA_PATH / "LiFePO4-neb-band.xyz",
        arch="mace",
        model_path=MODEL_PATH,
        steps=2,
        file_prefix=file_prefix,
    )
    neb.optimize()
    neb.plot()

    assert len(neb.images) == 7
    assert all(key in neb.results for key in ("barrier", "delta_E", "max_force"))
    assert neb.results["barrier"] == pytest.approx(0.67567742247752)
    assert neb.results["delta_E"] == pytest.approx(5.002693796996027e-07)
    assert neb.results["max_force"] == pytest.approx(1.5425684122118983)
