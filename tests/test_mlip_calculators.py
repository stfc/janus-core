"""Test configuration of MLIP calculators."""

from pathlib import Path

import pytest

from janus_core.mlip_calculators import choose_calculator

test_data = [
    (
        "mace",
        "cpu",
        {"model_paths": Path(__file__).parent / "models" / "mace_mp_small.model"},
    ),
    ("mace_off", "cpu", {}),
    ("mace_mp", "cpu", {}),
]


@pytest.mark.parametrize("architecture, device, kwargs", test_data)
def test_mace(architecture, device, kwargs):
    """Test mace calculators can be configured."""
    calculator = choose_calculator(architecture=architecture, device=device, **kwargs)
    assert calculator.parameters["version"] is not None


@pytest.mark.extra_mlips
def test_m3gnet():
    """Test m3gnet calculator can be configured."""
    calculator = choose_calculator(
        architecture="m3gnet",
    )
    assert calculator.parameters["version"] is not None


@pytest.mark.extra_mlips
def test_chgnet():
    """Test chgnet calculator can be configured."""
    calculator = choose_calculator(
        architecture="chgnet",
    )
    assert calculator.parameters["version"] is not None


def test_invalid_arch():
    """Test error raised for invalid architecture."""
    with pytest.raises(ValueError):
        choose_calculator(architecture="invalid")
