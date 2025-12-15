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

EQUIFORMER_LABEL = "EquiformerV2-83M-S2EF-OC20-2M"
ESEN_LABEL = "eSEN-30M-MP"


try:
    from fairchem.core.models.model_registry import model_name_to_local_file

    EQUIFORMER_PATH = model_name_to_local_file(
        EQUIFORMER_LABEL,
        local_cache=Path("~/.cache/fairchem").expanduser(),
    )
    ESEN_PATH = model_name_to_local_file(
        EQUIFORMER_LABEL,
        local_cache="~/.cache/fairchem",
    )
except ImportError:
    EQUIFORMER_PATH = None
    ESEN_PATH = None

NEQUIP_PATH = MODEL_PATH / "toluene.nequip.pth"

# AlphaNet MATPES model - download if not present
ALPHANET_CKPT = MODEL_PATH / "alphanet" / "MATPES" / "r2scan_1021.ckpt"
ALPHANET_CONFIG = MODEL_PATH / "alphanet" / "MATPES" / "matpes.json"

if not ALPHANET_CKPT.exists() or not ALPHANET_CONFIG.exists():
    try:
        from tests.utils import download_alphanet_model
        ALPHANET_CKPT, ALPHANET_CONFIG = download_alphanet_model("MATPES")
    except Exception as e:
        print(f"Warning: Could not download AlphaNet MATPES model: {e}")

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


ALIGNN_PATH = MODEL_PATH / "v5.27.2024"
M3GNET_DIR_PATH = MODEL_PATH / "M3GNet-MP-2021.2.8-DIRECT-PES"
M3GNET_MODEL_PATH = M3GNET_DIR_PATH / "model.pt"

try:
    from matgl import load_model

    M3GNET_POTENTIAL = load_model(path=M3GNET_DIR_PATH)
except ImportError:
    M3GNET_POTENTIAL = None

PET_MAD_CHECKPOINT = (
    "https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.1.0/models/pet-mad-v1.1.0.ckpt"
)


@pytest.mark.parametrize(
    "arch, device, kwargs",
    [
        ("alignn", "cpu", {}),
        ("alignn", "cpu", {"model_path": ALIGNN_PATH}),
        ("alignn", "cpu", {"model_path": ALIGNN_PATH / "best_model.pt"}),
        ("alignn", "cpu", {"model": "alignnff_wt10"}),
        ("alignn", "cpu", {"path": ALIGNN_PATH}),
        ("alphanet", "cpu", {"model": ALPHANET_CKPT, "config": ALPHANET_CONFIG}),
        ("alphanet", "cpu", {"model": ALPHANET_CKPT, "config": ALPHANET_CONFIG, "precision": "32"}),
        ("alphanet", "cpu", {"model": ALPHANET_CKPT, "config": ALPHANET_CONFIG, "precision": "64"}),
        ("chgnet", "cpu", {}),
        ("chgnet", "cpu", {"model": "0.2.0"}),
        ("chgnet", "cpu", {"model_path": CHGNET_PATH}),
        ("chgnet", "cpu", {"model": CHGNET_MODEL}),
        ("dpa3", "cpu", {"model_path": DPA3_PATH}),
        ("dpa3", "cpu", {"model": DPA3_PATH}),
        ("equiformer", "cpu", {}),
        ("equiformer", "cpu", {"model": EQUIFORMER_LABEL}),
        ("equiformer", "cpu", {"model_path": EQUIFORMER_LABEL}),
        ("equiformer", "cpu", {"model_name": EQUIFORMER_LABEL}),
        ("equiformer", "cpu", {"model_name": EQUIFORMER_PATH}),
        ("equiformer", "cpu", {"checkpoint_path": EQUIFORMER_PATH}),
        ("esen", "cpu", {}),
        ("esen", "cpu", {"model": ESEN_LABEL}),
        ("esen", "cpu", {"model_path": ESEN_LABEL}),
        ("esen", "cpu", {"model_name": ESEN_LABEL}),
        ("esen", "cpu", {"model_name": ESEN_PATH}),
        ("esen", "cpu", {"checkpoint_path": ESEN_PATH}),
        ("grace", "cpu", {}),
        ("grace", "cpu", {"model_path": "GRACE-1L-MP-r6"}),
        ("mace", "cpu", {"model": MACE_MP_PATH}),
        ("mace", "cpu", {"model_paths": MACE_MP_PATH}),
        ("mace", "cpu", {"model_path": MACE_MP_PATH}),
        ("mace_mp", "cpu", {}),
        ("mace_mp", "cpu", {"model": "small"}),
        ("mace_mp", "cpu", {"model_path": MACE_MP_PATH}),
        ("mace_mp", "cpu", {"model": MACE_MP_PATH}),
        ("mace_off", "cpu", {}),
        ("mace_off", "cpu", {"model": "small"}),
        ("mace_off", "cpu", {"model_path": MACE_OFF_PATH}),
        ("mace_off", "cpu", {"model": MACE_OFF_PATH}),
        ("mace_omol", "cpu", {}),
        ("mace_omol", "cpu", {"model": "extra_large"}),
        ("mattersim", "cpu", {}),
        ("mattersim", "cpu", {"model_path": "mattersim-v1.0.0-1m"}),
        ("m3gnet", "cpu", {}),
        ("m3gnet", "cpu", {"model_path": M3GNET_DIR_PATH}),
        ("m3gnet", "cpu", {"model_path": M3GNET_MODEL_PATH}),
        ("m3gnet", "cpu", {"potential": M3GNET_DIR_PATH}),
        ("m3gnet", "cpu", {"potential": M3GNET_POTENTIAL}),
        ("nequip", "cpu", {"model_path": NEQUIP_PATH}),
        ("nequip", "cpu", {"model": NEQUIP_PATH}),
        ("orb", "cpu", {}),
        ("orb", "cpu", {"model": ORB_MODEL}),
        ("pet_mad", "cpu", {}),
        ("pet_mad", "cpu", {"model": PET_MAD_CHECKPOINT}),
        ("pet_mad", "cpu", {"checkpoint_path": PET_MAD_CHECKPOINT}),
        ("sevennet", "cpu", {"model": SEVENNET_PATH}),
        ("sevennet", "cpu", {"path": SEVENNET_PATH}),
        ("sevennet", "cpu", {"model_path": SEVENNET_PATH}),
        ("sevennet", "cpu", {}),
        ("sevennet", "cpu", {"model": "sevennet-0"}),
        ("uma", "cpu", {"model": UMA_LABEL}),
        ("uma", "cpu", {"model_path": UMA_LABEL}),
        ("uma", "cpu", {"model_name": UMA_LABEL}),
        ("uma", "cpu", {"model": UMA_PREDICT_UNIT}),
        ("uma", "cpu", {"predict_unit": UMA_PREDICT_UNIT}),
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
        ("alignn", "invalid/path"),
        ("alphanet", "/invalid/path"),
        ("chgnet", "/invalid/path"),
        ("dpa3", "/invalid/path"),
        ("equiformer", "/invalid/path"),
        ("esen", "/invalid/path"),
        ("grace", "/invalid/path"),
        ("mace", "/invalid/path"),
        ("mace_mp", "/invalid/path"),
        ("mace_off", "/invalid/path"),
        ("mattersim", "/invalid/path"),
        ("m3gnet", "/invalid/path"),
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


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "arch": "alignn",
            "model_path": ALIGNN_PATH / "best_model.pt",
            "model": ALIGNN_PATH / "best_model.pt",
        },
        {
            "arch": "alignn",
            "model": ALIGNN_PATH / "best_model.pt",
            "path": ALIGNN_PATH / "best_model.pt",
        },
        {
            "arch": "alphanet",
            "model_path": ALPHANET_CKPT,
            "model": ALPHANET_CKPT,
        },
        {"arch": "chgnet", "model": CHGNET_PATH, "path": CHGNET_PATH},
        {"arch": "dpa3", "model_path": DPA3_PATH, "model": DPA3_PATH},
        {"arch": "dpa3", "model": DPA3_PATH, "path": DPA3_PATH},
        {
            "arch": "equiformer",
            "model_path": EQUIFORMER_LABEL,
            "model": EQUIFORMER_LABEL,
        },
        {
            "arch": "grace",
            "model_path": "GRACE-1L-MP-r6",
            "model": "GRACE-1L-MP-r6",
        },
        {"arch": "esen", "model_path": ESEN_LABEL, "model": ESEN_LABEL},
        {"arch": "mace", "model": MACE_MP_PATH, "model_paths": MACE_MP_PATH},
        {"arch": "mace", "model": MACE_MP_PATH, "model_paths": MACE_MP_PATH},
        {"arch": "mace", "model_path": MACE_MP_PATH, "model": MACE_MP_PATH},
        {"arch": "mace", "model": MACE_MP_PATH, "potential": MACE_MP_PATH},
        {
            "arch": "mattersim",
            "model": "mattersim-v1.0.0-1m",
            "path": "mattersim-v1.0.0-1m",
        },
        {
            "arch": "mattersim",
            "model_path": "mattersim-v1.0.0-1m",
            "path": "mattersim-v1.0.0-1m",
        },
        {"arch": "nequip", "model_path": NEQUIP_PATH, "model": NEQUIP_PATH},
        {"arch": "nequip", "model": NEQUIP_PATH, "path": NEQUIP_PATH},
        {"arch": "orb", "model_path": ORB_MODEL, "model": ORB_MODEL},
        {"arch": "orb", "model": ORB_MODEL, "path": ORB_MODEL},
        {
            "arch": "pet_mad",
            "model": PET_MAD_CHECKPOINT,
            "checkpoint_path": PET_MAD_CHECKPOINT,
        },
        {"arch": "sevennet", "model_path": SEVENNET_PATH, "model": SEVENNET_PATH},
        {"arch": "sevennet", "model": SEVENNET_PATH, "path": SEVENNET_PATH},
        {"arch": "uma", "model_path": UMA_LABEL, "model": UMA_LABEL},
    ],
)
def test_model_model_paths(kwargs):
    """Test error raised if model path is specified in multiple ways."""
    skip_extras(kwargs["arch"])
    with pytest.raises(ValueError):
        choose_calculator(**kwargs)


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
