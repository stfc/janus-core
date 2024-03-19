"""Test molecular dynamics."""

from pathlib import Path

from ase import Atoms
from ase.io import read
import pytest

from janus_core.md import NPH, NPT, NVE, NVT, NVT_NH
from janus_core.single_point import SinglePoint

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"

test_data = [
    (NVT, "nvt"),
    (NVE, "nve"),
    (NPT, "npt"),
    (NVT_NH, "nvt-nh"),
    (NPH, "nph"),
]


@pytest.mark.parametrize("ensemble, expected", test_data)
def test_init(ensemble, expected):
    """Test initialising ensembles."""
    single_point = SinglePoint(
        struct_path=DATA_PATH / "H2O.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    dyn = ensemble(
        struct=single_point.struct,
    )
    assert dyn.ensemble == expected


def test_npt(tmp_path):
    """Test NPT MD."""
    restart_stem = tmp_path / "res-Cl4Na4-npt"
    restart_path_1 = tmp_path / "res-Cl4Na4-npt-300.00K-2.xyz"
    restart_path_2 = tmp_path / "res-Cl4Na4-npt-300.00K-4.xyz"
    traj_path = tmp_path / "Cl4Na4-npt-300.0-traj.xyz"
    md_path = tmp_path / "Cl4Na4-npt-300.0-md.log"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    npt = NPT(
        struct=single_point.struct,
        pressure=1.0,
        temp=300.0,
        steps=4,
        traj_file=traj_path,
        traj_every=1,
        restart_stem=restart_stem,
        restart_every=2,
        md_file=md_path,
        output_every=1,
    )
    npt.run()

    restart_atoms_1 = read(restart_path_1)
    assert isinstance(restart_atoms_1, Atoms)
    restart_atoms_2 = read(restart_path_2)
    assert isinstance(restart_atoms_2, Atoms)

    traj = read(traj_path, index=":")
    assert all(isinstance(image, Atoms) for image in traj)
    assert len(traj) == 4
