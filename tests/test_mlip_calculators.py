"""Test configuration of MLIP calculators."""

from pathlib import Path

import pytest

from janus_core.helpers.mlip_calculators import choose_calculator

MODEL_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"

test_data_mace = [
    ("mace", "cpu", {"model": MODEL_PATH}),
    ("mace", "cpu", {"model_paths": MODEL_PATH}),
    ("mace_off", "cpu", {}),
    ("mace_mp", "cpu", {}),
    ("mace_mp", "cpu", {"model": MODEL_PATH}),
    ("mace_off", "cpu", {"model": "small"}),
]

test_data_extras = [("m3gnet", "cpu"), ("chgnet", "")]


@pytest.mark.parametrize("architecture, device, kwargs", test_data_mace)
def test_mace(architecture, device, kwargs):
    """Test mace calculators can be configured."""
    calculator = choose_calculator(architecture=architecture, device=device, **kwargs)
    assert calculator.parameters["version"] is not None


@pytest.mark.parametrize("architecture, device", test_data_extras)
def test_extra_mlips(architecture, device):
    """Test m3gnet and chgnet calculators can be configured."""
    calculator = choose_calculator(
        architecture=architecture,
        device=device,
    )
    assert calculator.parameters["version"] is not None


def test_invalid_arch():
    """Test error raised for invalid architecture."""
    with pytest.raises(ValueError):
        choose_calculator(architecture="invalid")


def test_model_model_paths():
    """Test error raised if both model and model_paths are specified."""
    with pytest.raises(ValueError):
        choose_calculator(
            architecture="mace",
            model=MODEL_PATH,
            model_paths=MODEL_PATH,
        )
