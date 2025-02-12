"""Test configuration of MLIP calculators."""

from __future__ import annotations

from pathlib import Path

from ase.eos import EquationOfState
import pytest

from janus_core.calculations.eos import EoS
from janus_core.calculations.single_point import SinglePoint
from janus_core.helpers.mlip_calculators import choose_calculator

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models"

MACE_PATH = MODEL_PATH / "mace_mp_small.model"
M3GNET_DIR_PATH = MODEL_PATH / "M3GNet-MP-2021.2.8-DIRECT-PES"
M3GNET_MODEL_PATH = M3GNET_DIR_PATH / "model.pt"
ALIGNN_PATH = MODEL_PATH / "v5.27.2024"

try:
    from matgl import load_model

    M3GNET_POTENTIAL = load_model(path=M3GNET_DIR_PATH)
except ImportError:
    M3GNET_POTENTIAL = None


@pytest.mark.parametrize(
    "arch, device, kwargs",
    [
        ("alignn", "cpu", {}),
        ("alignn", "cpu", {"model_path": ALIGNN_PATH}),
        ("alignn", "cpu", {"model_path": ALIGNN_PATH / "best_model.pt"}),
        ("alignn", "cpu", {"model": "alignnff_wt10"}),
        ("alignn", "cpu", {"path": ALIGNN_PATH}),
        ("m3gnet", "cpu", {}),
        ("m3gnet", "cpu", {"model_path": M3GNET_DIR_PATH}),
        ("m3gnet", "cpu", {"model_path": M3GNET_MODEL_PATH}),
        ("m3gnet", "cpu", {"potential": M3GNET_DIR_PATH}),
        ("m3gnet", "cpu", {"potential": M3GNET_POTENTIAL}),
    ],
)
def test_configure_mlips(arch, device, kwargs):
    """Test mace calculators can be configured."""
    if arch == "alignn":
        pytest.importorskip("alignn")
    if arch == "m3gnet":
        pytest.importorskip("matgl")

    calculator = choose_calculator(arch=arch, device=device, **kwargs)
    assert calculator.parameters["version"] is not None
    assert calculator.parameters["model_path"] is not None


@pytest.mark.parametrize(
    "arch, model_path",
    [
        ("alignn", Path("invalid/path")),
        ("m3gnet", "/invalid/path"),
    ],
)
def test_invalid_model_path(arch, model_path):
    """Test error raised for invalid model_path."""
    if arch == "alignn":
        pytest.importorskip("alignn")
    if arch == "m3gnet":
        pytest.importorskip("matgl")
    with pytest.raises((ValueError, RuntimeError, OSError)):
        choose_calculator(arch=arch, model_path=model_path)


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
            "model_path": ALIGNN_PATH / "best_model.pt",
            "path": ALIGNN_PATH / "best_model.pt",
        },
    ],
)
def test_extra_mlips_invalid(kwargs):
    """Test error raised if multiple model paths defined for extra MLIPs."""
    if kwargs["arch"] == "alignn":
        pytest.importorskip("alignn")
    if kwargs["arch"] == "m3gnet":
        pytest.importorskip("matgl")
    with pytest.raises(ValueError):
        choose_calculator(**kwargs)


@pytest.mark.parametrize(
    "arch, device, expected_energy, kwargs",
    [
        (
            "alignn",
            "cpu",
            -11.148092269897461,
            {"model_path": ALIGNN_PATH / "best_model.pt"},
        ),
        ("mace_mp", "cpu", -27.035127799332745, {"model_path": MACE_PATH}),
        ("m3gnet", "cpu", -26.729949951171875, {}),
    ],
)
def test_single_point_extras(arch, device, expected_energy, kwargs):
    """Test single point energy using extra MLIP calculators."""
    if arch == "alignn":
        pytest.importorskip("alignn")
    if arch == "m3gnet":
        pytest.importorskip("matgl")
    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch=arch,
        device=device,
        properties="energy",
        **kwargs,
    )
    energy = single_point.run()["energy"]
    assert energy == pytest.approx(expected_energy)


@pytest.mark.parametrize("arch, device", [("m3gnet", "cpu"), ("alignn", "cpu")])
def test_eos_extras(arch, device, tmp_path):
    """Test extra potentials."""
    if arch == "alignn":
        pytest.importorskip("alignn")
    if arch == "m3gnet":
        pytest.importorskip("matgl")

    log_file = tmp_path / "eos.log"
    eos_fit_path = tmp_path / "NaCl-eos-fit.dat"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch=arch,
        device=device,
    )

    eos = EoS(
        single_point.struct,
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
