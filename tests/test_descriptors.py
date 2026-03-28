"""Test MLIP desriptors calculations."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import read
import pytest
import torch

from janus_core.calculations.descriptors import Descriptors
from janus_core.calculations.single_point import SinglePoint
from janus_core.helpers.mlip_calculators import choose_calculator
from tests.utils import assert_log_contains

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_calc_descriptors(tmp_path, device):
    """Test calculating equation of state from ASE atoms object."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    struct = read(DATA_PATH / "NaCl.cif")
    log_file = tmp_path / "descriptors.log"
    struct.calc = choose_calculator(arch="mace_mp", model=MODEL_PATH, device=device)

    descriptors = Descriptors(
        struct,
        log_kwargs={"filename": log_file},
    )
    descriptors.run()
    atoms = descriptors.struct
    assert isinstance(atoms, Atoms)
    assert "mace_mp_descriptor" in atoms.info
    assert "mace_mp_Na_descriptor" not in atoms.info
    assert "mace_mp_Cl_descriptor" not in atoms.info

    # Check logging
    assert_log_contains(
        log_file,
        includes=["Starting descriptors calculation"],
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_calc_per_element(tmp_path, device):
    """Test calculating descriptors for each element from SinglePoint object."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    log_file = tmp_path / "descriptors.log"
    single_point = SinglePoint(
        struct=DATA_PATH / "NaCl.cif",
        arch="mace",
        model=MODEL_PATH,
        device=device,
    )

    descriptors = Descriptors(
        single_point.struct,
        calc_per_element=True,
        log_kwargs={"filename": log_file},
    )
    descriptors.run()
    atoms = descriptors.struct

    assert isinstance(atoms, Atoms)
    assert "mace_descriptor" in atoms.info
    assert "mace_Na_descriptor" in atoms.info
    assert "mace_Cl_descriptor" in atoms.info

    assert atoms.info["mace_descriptor"] == pytest.approx(-0.005626419559511429)
    assert atoms.info["mace_Cl_descriptor"] == pytest.approx(-0.009215340539869301)
    assert atoms.info["mace_Na_descriptor"] == pytest.approx(-0.0020374985791535563)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_logging(tmp_path, device):
    """Test attaching logger to Descriptors and emissions are saved to info."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    log_file = tmp_path / "descriptors.log"

    single_point = SinglePoint(
        struct=DATA_PATH / "NaCl.cif",
        arch="mace_mp",
        model=MODEL_PATH,
        device=device,
    )

    descriptors = Descriptors(
        single_point.struct,
        calc_per_element=True,
        log_kwargs={"filename": log_file},
    )

    assert "emissions" not in single_point.struct.info

    descriptors.run()

    assert log_file.exists()
    assert single_point.struct.info["emissions"] > 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dispersion(device):
    """Test using mace_mp with dispersion."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    single_point = SinglePoint(
        struct=DATA_PATH / "NaCl.cif",
        arch="mace_mp",
        calc_kwargs={"dispersion": False},
        device=device,
    )

    descriptors = Descriptors(
        single_point.struct,
        calc_per_element=True,
    )
    descriptors.run()

    single_point_disp = SinglePoint(
        struct=DATA_PATH / "NaCl.cif",
        arch="mace_mp",
        calc_kwargs={"dispersion": True},
        device=device,
    )

    descriptors_disp = Descriptors(
        single_point_disp.struct,
        calc_per_element=True,
    )
    descriptors_disp.run()

    assert descriptors_disp.struct.info["mace_mp_d3_descriptor"] == pytest.approx(
        descriptors.struct.info["mace_mp_descriptor"]
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_not_implemented_error(device):
    """Test correct implemented error raised if descriptors not installed."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from tests.utils import skip_extras

    skip_extras("chgnet")
    single_point = SinglePoint(
        struct=DATA_PATH / "NaCl.cif",
        arch="chgnet",
        device=device,
    )
    with pytest.raises(NotImplementedError):
        Descriptors(
            single_point.struct,
            calc_per_element=True,
        )


@pytest.mark.parametrize(
    "struct", (DATA_PATH / "NaCl.cif", read(DATA_PATH / "NaCl.cif"))
)
def test_missing_arch(struct):
    """Test missing arch."""
    with pytest.raises(ValueError, match="A calculator must be attached"):
        Descriptors(struct=struct)
