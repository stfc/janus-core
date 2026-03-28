"""Test geometry optimization."""
# ruff: noqa: N802, N803

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
from ase.mep.neb import DyNEB
from ase.optimize import BFGS
import numpy as np
import pytest
import torch

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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_neb(tmp_path, LFPO_start_b, LFPO_end_b, device):
    """Test NEB."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Write initial and final structures
    init_struct = tmp_path / "init_struct.cif"
    final_struct = tmp_path / "final_struct.cif"
    write(init_struct, images=LFPO_start_b)
    write(final_struct, images=LFPO_end_b)

    neb = NEB(
        init_struct=init_struct,
        final_struct=final_struct,
        arch="mace_mp",
        model=MODEL_PATH,
        n_images=5,
        neb_kwargs={"method": "aseneb"},
        file_prefix=tmp_path / "LFPO",
        device=device,
    )
    neb.run()

    assert len(neb.images) == 7
    assert all(key in neb.results for key in ("barrier", "delta_E", "max_force"))
    assert neb.results["barrier"] == pytest.approx(8117.328587956959)
    assert neb.results["delta_E"] == pytest.approx(-3.0149328722473e-07)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_neb_pymatgen(tmp_path, LFPO_start_b, LFPO_end_b, device):
    """Test pymatgen interpolation."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    file_prefix = tmp_path / "LFPO"
    single_point_start = SinglePoint(
        struct=LFPO_start_b,
        arch="mace",
        model=MODEL_PATH,
        device=device,
    )
    single_point_end = SinglePoint(
        struct=LFPO_end_b,
        arch="mace_mp",
        model=MODEL_PATH,
        device=device,
    )

    neb = NEB(
        init_struct=single_point_start.struct,
        final_struct=single_point_end.struct,
        arch="mace_mp",
        model=MODEL_PATH,
        n_images=5,
        interpolator="pymatgen",
        fmax=4,
        neb_kwargs={"method": "aseneb"},
        file_prefix=file_prefix,
        device=device,
    )
    neb.run()

    assert len(neb.images) == 7
    assert all(key in neb.results for key in ("barrier", "delta_E", "max_force"))
    assert neb.results["barrier"] == pytest.approx(0.5852889569624722)
    assert neb.results["delta_E"] == pytest.approx(-3.0149328722473e-07)
    assert neb.results["max_force"] == pytest.approx(2.533305796378263)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_set_calc(tmp_path, LFPO_start_b, LFPO_end_b, device):
    """Test setting the calculators explicitly."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    file_prefix = tmp_path / "LFPO"
    start_struct = LFPO_start_b
    end_struct = LFPO_end_b
    start_struct.calc = choose_calculator(
        arch="mace_mp", model=MODEL_PATH, device=device
    )
    end_struct.calc = choose_calculator(arch="mace_mp", model=MODEL_PATH, device=device)

    neb = NEB(
        init_struct=start_struct,
        final_struct=end_struct,
        arch="mace_mp",
        n_images=5,
        interpolator="ase",
        fmax=4,
        neb_kwargs={"method": "aseneb"},
        file_prefix=file_prefix,
        device=device,
    )
    neb.run()

    # Same as results from test_neb
    assert len(neb.images) == 7
    assert all(key in neb.results for key in ("barrier", "delta_E", "max_force"))
    assert neb.results["barrier"] == pytest.approx(8117.328587986063)
    assert neb.results["delta_E"] == pytest.approx(-3.0149328722473e-07)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_neb_functions(tmp_path, LFPO_start_b, LFPO_end_b, device):
    """Test individual NEB functions."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    file_prefix = tmp_path / "LFPO"

    neb = NEB(
        init_struct=LFPO_start_b,
        final_struct=LFPO_end_b,
        arch="mace",
        model=MODEL_PATH,
        n_images=5,
        interpolator="ase",
        neb_kwargs={"method": "aseneb"},
        file_prefix=file_prefix,
        device=device,
    )
    neb.interpolate()
    neb.optimize()
    neb.run_nebtools()

    assert len(neb.images) == 7
    assert all(key in neb.results for key in ("barrier", "delta_E", "max_force"))
    assert neb.results["barrier"] == pytest.approx(8117.328587986063)
    assert neb.results["delta_E"] == pytest.approx(-3.0149328722473e-07)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_neb_plot(tmp_path, device):
    """Test plotting NEB before running NEBTools."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    file_prefix = tmp_path / "LFPO"

    neb = NEB(
        neb_structs=DATA_PATH / "LiFePO4-neb-band.xyz",
        arch="mace",
        model=MODEL_PATH,
        steps=2,
        neb_kwargs={"method": "aseneb"},
        file_prefix=file_prefix,
        device=device,
    )
    neb.optimize()
    neb.plot()

    assert len(neb.images) == 7
    assert all(key in neb.results for key in ("barrier", "delta_E", "max_force"))
    assert neb.results["barrier"] == pytest.approx(0.4790452267999399)
    assert neb.results["delta_E"] == pytest.approx(-2.814124400174478e-07)
    assert neb.results["max_force"] == pytest.approx(1.5425684122118983)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_converge_warning(tmp_path, LFPO_start_b, LFPO_end_b, device):
    """Test warning raised if NEB does not converge."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    neb = NEB(
        init_struct=LFPO_start_b,
        final_struct=LFPO_end_b,
        arch="mace_mp",
        model=MODEL_PATH,
        n_images=5,
        steps=1,
        fmax=0.1,
        neb_kwargs={"method": "aseneb"},
        file_prefix=tmp_path / "LFPO",
        device=device,
    )
    with pytest.warns(UserWarning, match="NEB optimization has not converged"):
        neb.run()
    assert not neb.converged


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_restart(tmp_path, device):
    """Test restarting NEB optimisation."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    neb = NEB(
        neb_structs=DATA_PATH / "LiFePO4-neb-band.xyz",
        arch="mace",
        model=MODEL_PATH,
        interpolator=None,
        file_prefix=tmp_path / "LFPO",
        neb_kwargs={"method": "aseneb"},
        fmax=1.3,
        device=device,
    )
    neb.optimize()
    neb.run_nebtools()
    init_force = neb.results["max_force"]

    neb.optimize(fmax=1.2)

    neb.run_nebtools()
    final_force = neb.results["max_force"]

    assert final_force < init_force


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_restart_update_climb(tmp_path, device):
    """Test updating NEB climb setting when continuing optimization."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    results = {}

    for label in ("climb", "no-climb"):
        neb = NEB(
            neb_structs=DATA_PATH / "LiFePO4-neb-band.xyz",
            arch="mace",
            model=MODEL_PATH,
            interpolator=None,
            fmax=1.3,
            neb_kwargs={"method": "aseneb"},
            file_prefix=tmp_path / "LFPO",
            device=device,
        )
        neb.run()
        neb.neb.climb = label == "climb"

        neb.run(fmax=1.2)
        results[label] = neb.results

    assert results["climb"]["barrier"] != results["no-climb"]["barrier"]


@pytest.mark.parametrize("n_images", (-5, 0, 1.5))
def test_invalid_n_images(LFPO_start_b, LFPO_end_b, n_images):
    """Test setting invalid invalid n_images."""
    with pytest.raises(ValueError):
        NEB(
            arch="mace_mp",
            init_struct=LFPO_start_b,
            final_struct=LFPO_end_b,
            n_images=n_images,
        )


def test_invalid_interpolator(LFPO_start_b, LFPO_end_b):
    """Test setting invalid interpolator."""
    with pytest.raises(ValueError):
        NEB(
            arch="mace_mp",
            init_struct=LFPO_start_b,
            final_struct=LFPO_end_b,
            interpolator="invalid",
        )


def test_missing_interpolator(LFPO_start_b, LFPO_end_b):
    """Test setting missing interpolator."""
    with pytest.raises(ValueError):
        NEB(
            arch="mace_mp",
            init_struct=LFPO_start_b,
            final_struct=LFPO_end_b,
            interpolator=None,
        )


def test_invalid_structs(LFPO_start_b, LFPO_end_b):
    """Test error raised if init_struct/final_struct are passed with neb_structs."""
    with pytest.raises(ValueError):
        NEB(
            arch="mace_mp",
            init_struct=LFPO_start_b,
            neb_structs=DATA_PATH / "LiFePO4-neb-band.xyz",
        )
    with pytest.raises(ValueError):
        NEB(
            arch="mace_mp",
            final_struct=LFPO_end_b,
            neb_structs=DATA_PATH / "LiFePO4-neb-band.xyz",
        )
    with pytest.raises(ValueError):
        NEB(
            arch="mace_mp",
            init_struct=LFPO_start_b,
            final_struct=LFPO_end_b,
            neb_structs=DATA_PATH / "LiFePO4-neb-band.xyz",
        )


@pytest.mark.parametrize(
    "struct", (DATA_PATH / "NaCl.cif", read(DATA_PATH / "NaCl.cif"))
)
def test_missing_arch(struct):
    """Test missing arch."""
    with pytest.raises(ValueError, match="A calculator must be attached"):
        NEB(neb_structs=struct)


def test_max_force(tmp_path, LFPO_start_b, LFPO_end_b):
    """Test saved max force corresponds to correct NEB."""
    neb = NEB(
        init_struct=LFPO_start_b,
        final_struct=LFPO_end_b,
        arch="mace_mp",
        model=MODEL_PATH,
        n_images=5,
        neb_class=DyNEB,
        neb_kwargs={"climb": True, "method": "eb", "scale_fmax": 1.0},
        optimizer=BFGS,
        interpolator="ase",
        steps=1,
        file_prefix=tmp_path / "LFPO",
    )
    neb.run()

    assert all(key in neb.results for key in ("barrier", "delta_E", "max_force"))

    stored_max_force = neb.results["max_force"]
    forces = neb.neb.get_forces()
    max_force = np.sqrt((forces**2).sum(axis=1).max())
    assert stored_max_force == pytest.approx(max_force)
