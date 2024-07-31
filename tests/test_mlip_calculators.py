"""Test configuration of MLIP calculators."""

from pathlib import Path

from chgnet.model.model import CHGNet
from matgl import load_model
import pytest

from janus_core.helpers.mlip_calculators import choose_calculator

MODEL_PATH = Path(__file__).parent / "models"

MACE_MP_PATH = MODEL_PATH / "mace_mp_small.model"

MACE_OFF_PATH = MODEL_PATH / "MACE-OFF23_small.model"

M3GNET_DIR_PATH = MODEL_PATH / "M3GNet-MP-2021.2.8-DIRECT-PES"
M3GNET_MODEL_PATH = M3GNET_DIR_PATH / "model.pt"
M3GNET_POTENTIAL = load_model(path=M3GNET_DIR_PATH)

CHGNET_PATH = MODEL_PATH / "chgnet_0.3.0_e29f68s314m37.pth.tar"
CHGNET_MODEL = CHGNet.from_file(path=CHGNET_PATH)

SEVENNET_PATH = MODEL_PATH / "sevennet_0.pth"

ALIGNN_PATH = MODEL_PATH / "v5.27.2024"


@pytest.mark.parametrize(
    "architecture, device, kwargs",
    [
        ("mace", "cpu", {"model": MACE_MP_PATH}),
        ("mace", "cpu", {"model_paths": MACE_MP_PATH}),
        ("mace", "cpu", {"model_path": MACE_MP_PATH}),
        ("mace_off", "cpu", {}),
        ("mace_off", "cpu", {"model": "small"}),
        ("mace_off", "cpu", {"model_path": MACE_OFF_PATH}),
        ("mace_off", "cpu", {"model": MACE_OFF_PATH}),
        ("mace_mp", "cpu", {}),
        ("mace_mp", "cpu", {"model": "small"}),
        ("mace_mp", "cpu", {"model_path": MACE_MP_PATH}),
        ("mace_mp", "cpu", {"model": MACE_MP_PATH}),
        ("m3gnet", "cpu", {}),
        ("m3gnet", "cpu", {"model_path": M3GNET_DIR_PATH}),
        ("m3gnet", "cpu", {"model_path": M3GNET_MODEL_PATH}),
        ("m3gnet", "cpu", {"potential": M3GNET_DIR_PATH}),
        ("m3gnet", "cpu", {"potential": M3GNET_POTENTIAL}),
        ("chgnet", "cpu", {}),
        ("chgnet", "cpu", {"model": "0.2.0"}),
        ("chgnet", "cpu", {"model_path": CHGNET_PATH}),
        ("chgnet", "cpu", {"model": CHGNET_MODEL}),
    ],
)
def test_mlips(architecture, device, kwargs):
    """Test mace calculators can be configured."""
    calculator = choose_calculator(architecture=architecture, device=device, **kwargs)
    assert calculator.parameters["version"] is not None


def test_invalid_arch():
    """Test error raised for invalid architecture."""
    with pytest.raises(ValueError):
        choose_calculator(architecture="invalid")


@pytest.mark.parametrize(
    "architecture, model_path",
    [
        ("mace", "/invalid/path"),
        ("mace_off", "/invalid/path"),
        ("mace_mp", "/invalid/path"),
        ("m3gnet", "/invalid/path"),
        ("chgnet", "/invalid/path"),
    ],
)
def test_invalid_model_path(architecture, model_path):
    """Test error raised for invalid model_path."""
    with pytest.raises((ValueError, RuntimeError)):
        choose_calculator(architecture=architecture, model_path=model_path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": MACE_MP_PATH, "model_paths": MACE_MP_PATH},
        {"model_path": MACE_MP_PATH, "model_paths": MACE_MP_PATH},
        {"model_path": MACE_MP_PATH, "model": MACE_MP_PATH},
        {"model_path": MACE_MP_PATH, "potential": MACE_MP_PATH},
    ],
)
def test_model_model_paths(kwargs):
    """Test error raised if model path is specified in multiple ways."""
    with pytest.raises(ValueError):
        choose_calculator(architecture="mace", **kwargs)


@pytest.mark.parametrize("architecture", ["mace_mp", "mace_off"])
def test_invalid_device(architecture):
    """Test error raised for invalid device is specified."""
    with pytest.raises(ValueError):
        choose_calculator(architecture=architecture, device="invalid")


@pytest.mark.extra_mlips
@pytest.mark.parametrize(
    "architecture, device, kwargs",
    [
        ("alignn", "cpu", {}),
        ("alignn", "cpu", {"model_path": ALIGNN_PATH}),
        ("alignn", "cpu", {"model_path": ALIGNN_PATH / "best_model.pt"}),
        ("alignn", "cpu", {"model": "alignnff_wt10"}),
        ("alignn", "cpu", {"path": ALIGNN_PATH} ),
        ("sevennet", "cpu", {"model": SEVENNET_PATH}),
        ("sevennet", "cpu", {"path": SEVENNET_PATH}),
        ("sevennet", "cpu", {"model_path": SEVENNET_PATH}),
    ],
)
def test_extra_mlips(architecture, device, kwargs):
    """Test extra MLIPs (alignn) can be configured."""
    calculator = choose_calculator(
        architecture=architecture,
        device=device,
        **kwargs,
    )
    assert calculator.parameters["version"] is not None


@pytest.mark.extra_mlips
@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "architecture": "alignn",
            "model_path": ALIGNN_PATH / "best_model.pt",
            "model": ALIGNN_PATH / "best_model.pt",
        },
        {
            "architecture": "alignn",
            "model_path": ALIGNN_PATH / "best_model.pt",
            "path": ALIGNN_PATH / "best_model.pt",
        },
        {
            "architecture": "sevennet",
            "model_path": SEVENNET_PATH,
            "path": SEVENNET_PATH,
        },
        {
            "architecture": "sevennet",
            "model_path": SEVENNET_PATH,
            "model": SEVENNET_PATH,
        },
        {
            "architecture": "sevennet",
            "model_path": "",
        },
    ],
)
def test_alignff_invalid(kwargs):
    """Test error raised if multiple model paths defined for extra MLIPs."""
    with pytest.raises(ValueError):
        choose_calculator(**kwargs)
