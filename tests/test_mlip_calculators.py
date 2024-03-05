"""Test configuration of MLIP calculators."""

from pathlib import Path

import pytest

from janus_core.mlip_calculators import choose_calculator

test_data_mace = [
    (
        "mace",
        "cpu",
        {"model_paths": Path(__file__).parent / "models" / "mace_mp_small.model"},
    ),
    ("mace_off", "cpu", {}),
    ("mace_mp", "cpu", {}),
    (
        "mace_mp",
        "cpu",
        {"model_paths": Path(__file__).parent / "models" / "mace_mp_small.model"},
    ),
    (
        "mace_off",
        "cpu",
        {"model_paths": "small"},
    ),
]

test_data_extras = [("m3gnet", "cpu"), ("chgnet", "")]


@pytest.mark.parametrize("architecture, device, kwargs", test_data_mace)
def test_mace(architecture, device, kwargs):
    """Test mace calculators can be configured."""
    calculator = choose_calculator(architecture=architecture, device=device, **kwargs)
    assert calculator.parameters["version"] is not None


@pytest.mark.extra_mlips
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
