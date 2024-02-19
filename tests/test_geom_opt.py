"""Test geometry optimisation."""

from pathlib import Path

try:
    from ase.filters import UnitCellFilter
except ImportError:
    from ase.constraints import UnitCellFilter
import pytest

from janus_core.geom_opt import optimize
from janus_core.single_point import SinglePoint

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"

test_data = [
    ("mace", "NaCl.cif", MODEL_PATH, -27.046359959669214, {}),
    ("mace", "NaCl.cif", MODEL_PATH, -27.04636199814088, {"fmax": 0.001}),
    (
        "mace",
        "NaCl.cif",
        MODEL_PATH,
        -27.0463392211678,
        {"filter_func": UnitCellFilter},
    ),
    ("mace", "H2O.cif", MODEL_PATH, -14.051389496520015, {"filter_func": None}),
    (
        "mace",
        "NaCl.cif",
        MODEL_PATH,
        -27.04631447979008,
        {"filter_func": UnitCellFilter, "filter_kwargs": {"scalar_pressure": 0.0001}},
    ),
    (
        "mace",
        "NaCl.cif",
        MODEL_PATH,
        -27.046353221978332,
        {"opt_kwargs": {"alpha": 100}},
    ),
    (
        "mace",
        "NaCl.cif",
        MODEL_PATH,
        -27.03561540212425,
        {"filter_func": UnitCellFilter, "dyn_kwargs": {"steps": 1}},
    ),
]


@pytest.mark.parametrize(
    "architecture, structure, model_path, expected, kwargs", test_data
)
def test_optimize(architecture, structure, model_path, expected, kwargs):
    """Test optimizing geometry using MACE."""
    data_path = DATA_PATH / structure
    single_point = SinglePoint(
        system=data_path, architecture=architecture, model_paths=model_path
    )

    init_energy = single_point.run_single_point("energy")["energy"]

    atoms = optimize(single_point.sys, **kwargs)

    assert atoms.get_potential_energy() < init_energy
    assert atoms.get_potential_energy() == pytest.approx(expected)
