"""Test configuration of MLIP calculators."""

import os

import pytest

from janus_core.mlip_calculators import choose_calculator


def test_mace():
    """
    Test mace calculator can be configured.
    """
    model_path = os.path.join(os.path.dirname(__file__), "models", "MACE_small.model")
    calculator = choose_calculator(
        architecture="mace", device="cpu", model_paths=model_path
    )
    assert calculator.parameters["version"] is not None


def test_mace_off():
    """Test mace_off calculator can be configured."""
    calculator = choose_calculator(
        architecture="mace_off",
        device="cpu",
    )
    assert calculator.parameters["version"] is not None


def test_mace_mp():
    """Test mace_mp calculator can be configured."""
    calculator = choose_calculator(
        architecture="mace_mp",
        device="cpu",
    )
    assert calculator.parameters["version"] is not None


@pytest.mark.extra_mlips
def test_m3gnet():
    """
    Test m3gnet calculator can be configured.
    """
    calculator = choose_calculator(
        architecture="m3gnet",
    )
    assert calculator.parameters["version"] is not None


@pytest.mark.extra_mlips
def test_chgnet():
    """
    Test chgnet calculator can be configured.
    """
    calculator = choose_calculator(
        architecture="chgnet",
    )
    assert calculator.parameters["version"] is not None


def test_invalid_arch():
    """
    Test error raised for invalid architecture.
    """
    with pytest.raises(ValueError):
        choose_calculator(architecture="invalid")
