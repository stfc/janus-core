"""Test configuration of MLIP calculators."""

from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError, URLError
from zipfile import BadZipFile

import pytest

from janus_core.helpers.mlip_calculators import add_dispersion, choose_calculator
from tests.utils import skip_extras

MODEL_PATH = Path(__file__).parent / "models"

MACE_MP_PATH = MODEL_PATH / "mace_mp_small.model"

MACE_OFF_PATH = MODEL_PATH / "MACE-OFF23_small.model"

CHGNET_PATH = MODEL_PATH / "chgnet_0.3.0_e29f68s314m37.pth.tar"

try:
    from chgnet.model.model import CHGNet

    CHGNET_MODEL = CHGNet.from_file(path=CHGNET_PATH)
except ImportError:
    CHGNET_MODEL = None

DPA3_PATH = MODEL_PATH / "2025-01-10-dpa3-mptrj.pth"

NEQUIP_PATH = MODEL_PATH / "toluene.nequip.pth"

ORB_WEIGHTS_PATH = MODEL_PATH / "orb-d3-xs-v2-20241011.ckpt"

try:
    from orb_models.forcefield.pretrained import orb_d3_xs_v2

    ORB_MODEL = orb_d3_xs_v2(weights_path=ORB_WEIGHTS_PATH)
except ImportError:
    ORB_MODEL = None

SEVENNET_PATH = MODEL_PATH / "sevennet_0.pth"

UMA_LABEL = "uma-s-1"

try:
    from fairchem.core import pretrained_mlip
    from huggingface_hub.errors import GatedRepoError

    try:
        UMA_PREDICT_UNIT = pretrained_mlip.get_predict_unit(
            model_name=UMA_LABEL, device="cpu"
        )
    except GatedRepoError:
        UMA_PREDICT_UNIT = None

except ImportError:
    UMA_PREDICT_UNIT = None


PET_MAD_CHECKPOINT = (
    "https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.1.0/models/pet-mad-v1.1.0.ckpt"
)


@pytest.mark.parametrize(
    "arch, device, kwargs",
    [
        ("chgnet", "cpu", {}),
        ("chgnet", "cpu", {"model": "0.2.0"}),
        ("chgnet", "cpu", {"model": CHGNET_PATH}),
        ("chgnet", "cpu", {"model": CHGNET_MODEL}),
        ("dpa3", "cpu", {"model": DPA3_PATH}),
        ("grace", "cpu", {}),
        ("grace", "cpu", {"model": "GRACE-1L-MP-r6"}),
        ("mace", "cpu", {"model": MACE_MP_PATH}),
        ("mace", "cpu", {"model_paths": MACE_MP_PATH}),
        ("mace_mp", "cpu", {}),
        ("mace_mp", "cpu", {"model": "small"}),
        ("mace_mp", "cpu", {"model": MACE_MP_PATH}),
        ("mace_off", "cpu", {}),
        ("mace_off", "cpu", {"model": "small"}),
        ("mace_off", "cpu", {"model": MACE_OFF_PATH}),
        ("mace_omol", "cpu", {}),
        ("mace_omol", "cpu", {"model": "extra_large"}),
        ("mattersim", "cpu", {}),
        ("mattersim", "cpu", {"model": "mattersim-v1.0.0-1m"}),
        ("nequip", "cpu", {"model": NEQUIP_PATH}),
        ("orb", "cpu", {}),
        ("orb", "cpu", {"model": ORB_MODEL}),
        ("pet_mad", "cpu", {}),
        ("pet_mad", "cpu", {"model": PET_MAD_CHECKPOINT}),
        ("pet_mad", "cpu", {"checkpoint_path": PET_MAD_CHECKPOINT}),
        ("sevennet", "cpu", {"model": SEVENNET_PATH}),
        ("sevennet", "cpu", {"path": SEVENNET_PATH}),
        ("sevennet", "cpu", {}),
        ("sevennet", "cpu", {"model": "sevennet-0"}),
        ("fairchem", "cpu", {"model": UMA_LABEL}),
        ("fairchem", "cpu", {"model_name": UMA_LABEL}),
        ("fairchem", "cpu", {"model": UMA_PREDICT_UNIT}),
        ("fairchem", "cpu", {"predict_unit": UMA_PREDICT_UNIT}),
    ],
)
def test_mlips(arch, device, kwargs):
    """Test calculators can be configured."""
    skip_extras(arch)

    try:
        calculator = choose_calculator(arch=arch, device=device, **kwargs)
        assert calculator.parameters["version"] is not None
        assert calculator.parameters["model"] is not None
    except BadZipFile:
        pytest.skip("Model download failed")
    except HTTPError as err:  # Inherits from URLError, so check first
        if "Service Unavailable" in err.msg or "Too Many Requests" in err.msg:
            pytest.skip("Model download failed")
        raise err
    except URLError as err:
        if "Connection timed out" in err.reason:
            pytest.skip("Model download failed")
        raise err


def test_invalid_arch():
    """Test error raised for invalid architecture."""
    with pytest.raises(ValueError):
        choose_calculator(arch="invalid")


@pytest.mark.parametrize(
    "arch, model",
    [
        ("chgnet", "/invalid/path"),
        ("dpa3", "/invalid/path"),
        ("grace", "/invalid/path"),
        ("mace", "/invalid/path"),
        ("mace_mp", "/invalid/path"),
        ("mace_off", "/invalid/path"),
        ("mattersim", "/invalid/path"),
        ("nequip", "/invalid/path"),
        ("orb", "/invalid/path"),
        ("pet_mad", "/invalid/path"),
        ("sevenn", "/invalid/path"),
        ("uma", "/invalid/path"),
    ],
)
def test_invalid_model(arch, model):
    """Test error raised for invalid model."""
    skip_extras(arch)
    with pytest.raises((ValueError, RuntimeError, KeyError, AssertionError)):
        choose_calculator(arch=arch, model=model)


@pytest.mark.parametrize("arch", ["mace_mp", "mace_off"])
def test_invalid_device(arch):
    """Test error raised if invalid device is specified."""
    with pytest.raises(ValueError):
        choose_calculator(arch=arch, device="invalid")


def test_d3():
    """Test adding D3 dispersion calculator automatically."""
    skip_extras("mace_mp")

    calculator = choose_calculator(arch="mace_mp", dispersion=True)
    assert calculator.parameters["version"] is not None
    assert calculator.parameters["model"] is not None
    assert calculator.parameters["arch"] == "mace_mp_d3"


def test_d3_manual():
    """Test adding D3 dispersion calculator manually."""
    skip_extras("mace_mp")

    calculator = choose_calculator(arch="mace_mp")
    calculator = add_dispersion(calculator)
    assert calculator.parameters["version"] is not None
    assert calculator.parameters["model"] is not None
    assert calculator.parameters["arch"] == "mace_mp_d3"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"arch": "chgnet", "model": CHGNET_PATH, "path": CHGNET_PATH},
        {"arch": "dpa3", "model": DPA3_PATH, "path": DPA3_PATH},
        {"arch": "mace", "model": MACE_MP_PATH, "model_paths": MACE_MP_PATH},
        {"arch": "mace", "model": MACE_MP_PATH, "model_paths": MACE_MP_PATH},
        {"arch": "mace", "model": MACE_MP_PATH, "potential": MACE_MP_PATH},
        {
            "arch": "mattersim",
            "model": "mattersim-v1.0.0-1m",
            "path": "mattersim-v1.0.0-1m",
        },
        {"arch": "nequip", "model": NEQUIP_PATH, "path": NEQUIP_PATH},
        {"arch": "orb", "model": ORB_MODEL, "path": ORB_MODEL},
        {
            "arch": "pet_mad",
            "model": PET_MAD_CHECKPOINT,
            "checkpoint_path": PET_MAD_CHECKPOINT,
        },
        {"arch": "sevennet", "model": SEVENNET_PATH, "path": SEVENNET_PATH},
    ],
)
def test_duplicate_model_input(kwargs):
    """Test error raised if model is specified in multiple ways."""
    skip_extras(kwargs["arch"])
    with pytest.raises(ValueError):
        choose_calculator(**kwargs)
