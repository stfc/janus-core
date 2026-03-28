"""Test equation of state calculations."""

from __future__ import annotations

from pathlib import Path
from zipfile import BadZipFile

from ase.eos import EquationOfState
from ase.io import read
import pytest
import torch

from janus_core.calculations.eos import EoS
from janus_core.calculations.single_point import SinglePoint
from janus_core.helpers.mlip_calculators import choose_calculator
from tests.utils import assert_log_contains, skip_extras

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_calc_eos(tmp_path, device):
    """Test calculating equation of state from ASE atoms object."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    struct = read(DATA_PATH / "NaCl.cif")
    log_file = tmp_path / "eos.log"
    struct.calc = choose_calculator(arch="mace_mp", model=MODEL_PATH, device=device)

    eos = EoS(
        struct,
        device=device,
        file_prefix=tmp_path / "NaCl",
        log_kwargs={"filename": log_file},
    )
    results = eos.run()
    assert all(key in results for key in ("eos", "bulk_modulus", "e_0", "v_0"))

    # Check geometry optimization run by default
    assert_log_contains(
        log_file,
        includes=["Using filter", "Using optimizer", "Starting geometry optimization"],
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_no_optimize(tmp_path, device):
    """Test not optimizing structure before calculation."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    log_file = tmp_path / "eos.log"
    single_point = SinglePoint(
        struct=DATA_PATH / "NaCl.cif",
        arch="mace",
        model=MODEL_PATH,
        device=device,
    )
    eos = EoS(
        single_point.struct,
        device=device,
        minimize=False,
        file_prefix=tmp_path / "NaCl",
        log_kwargs={"filename": log_file},
    )
    eos.run()

    # Check geometry optimization turned off
    assert_log_contains(
        log_file,
        excludes=["Using filter", "Using optimizer", "Starting geometry optimization"],
    )


@pytest.mark.parametrize(
    "arch, device",
    [
        ("chgnet", "cpu"),
        ("chgnet", "cuda"),
        ("sevennet", "cpu"),
        ("sevennet", "cuda"),
    ],
)
def test_extras(arch, device, tmp_path):
    """Test extra potentials."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    skip_extras(arch)

    eos_fit_path = tmp_path / "NaCl-eos-fit.dat"
    log_file = tmp_path / "eos.log"

    try:
        eos = EoS(
            struct=DATA_PATH / "NaCl.cif",
            arch=arch,
            device=device,
            minimize=False,
            file_prefix=tmp_path / "NaCl",
            log_kwargs={"filename": log_file},
        )
        results = eos.run()

        assert isinstance(results["eos"], EquationOfState)

        # Check contents of EoS fit data file
        with open(eos_fit_path, encoding="utf8") as eos_fit_file:
            lines = eos_fit_file.readlines()

        assert len(lines) == 2
        assert len(lines[1].split()) == 3

    except BadZipFile:
        pytest.skip()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_invalid_struct(device):
    """Test setting invalid structure."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    single_point = SinglePoint(
        struct=DATA_PATH / "benzene-traj.xyz",
        arch="mace_mp",
        model=MODEL_PATH,
        device=device,
    )

    with pytest.raises(NotImplementedError):
        EoS(
            single_point.struct,
        )
    with pytest.raises(ValueError):
        EoS(
            "structure",
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_logging(tmp_path, device):
    """Test attaching logger to EoS and emissions are saved to info."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    log_file = tmp_path / "eos.log"

    single_point = SinglePoint(
        struct=DATA_PATH / "NaCl.cif",
        arch="mace_mp",
        model=MODEL_PATH,
        device=device,
    )

    eos = EoS(
        single_point.struct,
        device=device,
        file_prefix=tmp_path / "NaCl",
        log_kwargs={"filename": log_file},
    )

    assert "emissions" not in single_point.struct.info

    eos.run()

    assert log_file.exists()
    assert single_point.struct.info["emissions"] > 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_plot(tmp_path, device):
    """Test plotting equation of state."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    plot_file = tmp_path / "plot.svg"

    eos = EoS(
        struct=DATA_PATH / "NaCl.cif",
        arch="mace_mp",
        model=MODEL_PATH,
        device=device,
        plot_to_file=True,
        plot_kwargs={"filename": plot_file},
        file_prefix=tmp_path / "NaCl",
    )

    results = eos.run()
    assert all(key in results for key in ("eos", "bulk_modulus", "e_0", "v_0"))
    assert plot_file.exists()


@pytest.mark.parametrize(
    "struct", (DATA_PATH / "NaCl.cif", read(DATA_PATH / "NaCl.cif"))
)
def test_missing_arch(struct):
    """Test missing arch."""
    with pytest.raises(ValueError, match="A calculator must be attached"):
        EoS(struct=struct)
