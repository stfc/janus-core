"""Test molecular dynamics."""

from pathlib import Path

from ase import Atoms
from ase.io import read
import numpy as np
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
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    dyn = ensemble(
        struct=single_point.struct,
    )
    assert dyn.ensemble == expected


def test_npt(tmp_path):
    """Test NPT molecular dynamics."""
    restart_stem = tmp_path / "res-Cl4Na4-npt"
    restart_path_1 = tmp_path / "res-Cl4Na4-npt-300.0-2.xyz"
    restart_path_2 = tmp_path / "res-Cl4Na4-npt-300.0-4.xyz"
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

    with open(md_path, encoding="utf8") as md_file:
        lines = md_file.readlines()
        assert "Pressure[bar] | T [K]" in lines[0]
        assert len(lines) == 5


def test_nvt_nh(tmp_path):
    """Test NVT-Nosé–Hoover  molecular dynamics."""
    restart_stem = tmp_path / "res-Cl4Na4-nvt-nh"
    restart_path = tmp_path / "res-Cl4Na4-nvt-nh-300.0-3.xyz"
    traj_path = tmp_path / "Cl4Na4-nvt-nh-300.0-traj.xyz"
    md_path = tmp_path / "Cl4Na4-nvt-nh-300.0-md.log"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    nvt_nh = NVT_NH(
        struct=single_point.struct,
        temp=300.0,
        steps=3,
        traj_file=traj_path,
        traj_every=1,
        restart_stem=restart_stem,
        restart_every=3,
        md_file=md_path,
        output_every=1,
    )
    nvt_nh.run()

    restart_atoms = read(restart_path)
    assert isinstance(restart_atoms, Atoms)

    traj = read(traj_path, index=":")
    assert all(isinstance(image, Atoms) for image in traj)
    assert len(traj) == 3

    with open(md_path, encoding="utf8") as md_file:
        lines = md_file.readlines()
        assert " | T [K]" in lines[0]
        assert len(lines) == 4


def test_nve(tmp_path):
    """Test NVE molecular dynamics."""
    restart_stem = tmp_path / "res-Cl4Na4-nve"
    restart_path = tmp_path / "res-Cl4Na4-nve-300.0-4.xyz"
    traj_path = tmp_path / "Cl4Na4-nve-300.0-traj.xyz"
    md_path = tmp_path / "Cl4Na4-nve-300.0-md.log"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    nve = NVE(
        struct=single_point.struct,
        temp=300.0,
        steps=4,
        traj_file=traj_path,
        traj_every=3,
        restart_stem=restart_stem,
        restart_every=4,
        md_file=md_path,
        output_every=1,
    )
    nve.run()

    restart_atoms = read(restart_path)
    assert isinstance(restart_atoms, Atoms)

    traj = read(traj_path, index=":")
    assert all(isinstance(image, Atoms) for image in traj)
    assert len(traj) == 2

    with open(md_path, encoding="utf8") as md_file:
        lines = md_file.readlines()
        assert " | Epot/N [eV]" in lines[0]
        # Includes step 0
        assert len(lines) == 6


def test_nph(tmp_path):
    """Test NPH molecular dynamics."""
    restart_stem = tmp_path / "res-Cl4Na4-nph"
    restart_path = tmp_path / "res-Cl4Na4-nph-300.0-4.xyz"
    traj_path = tmp_path / "Cl4Na4-nph-300.0-traj.xyz"
    md_path = tmp_path / "Cl4Na4-nph-300.0-md.log"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    nph = NPH(
        struct=single_point.struct,
        temp=300.0,
        steps=3,
        traj_file=traj_path,
        traj_every=2,
        restart_stem=restart_stem,
        restart_every=4,
        md_file=md_path,
        output_every=2,
    )
    nph.run()

    with pytest.raises(FileNotFoundError):
        read(restart_path)

    traj = read(traj_path, index=":")
    assert all(isinstance(image, Atoms) for image in traj)
    assert len(traj) == 1

    with open(md_path, encoding="utf8") as md_file:
        lines = md_file.readlines()
        assert " | Epot/N [eV]" in lines[0]
        assert len(lines) == 2


def test_restart_nvt(tmp_path):
    """Test restarting NVT molecular dynamics."""
    restart_stem = tmp_path / "res-Cl4Na4-nvt"
    traj_path = tmp_path / "Cl4Na4-nvt-300.0-traj.xyz"
    md_path = tmp_path / "Cl4Na4-nvt-300.0-md.log"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    nvt = NVT(
        struct=single_point.struct,
        temp=300.0,
        steps=4,
        traj_file=traj_path,
        traj_every=1,
        restart_stem=restart_stem,
        restart_every=4,
        md_file=md_path,
        output_every=1,
    )
    nvt.run()

    assert nvt.dyn.nsteps == 4

    nvt_restart = NVT(
        struct=single_point.struct,
        temp=300.0,
        steps=4,
        traj_file=traj_path,
        traj_every=1,
        restart_stem=restart_stem,
        restart_every=4,
        md_file=md_path,
        output_every=1,
        restart=True,
    )
    nvt_restart.run()
    assert nvt_restart.offset == 4

    with open(md_path, encoding="utf8") as md_file:
        lines = md_file.readlines()
        assert " | T [K]" in lines[0]
        # Includes step 0, and step 4 from restart
        assert len(lines) == 11
        assert 8 == int(lines[-1].split()[0])

    traj = read(traj_path, index=":")
    assert all(isinstance(image, Atoms) for image in traj)
    assert len(traj) == 10


def test_minimize(tmp_path):
    """Test geometry optimzation before dynamics."""
    traj_path = tmp_path / "Cl4Na4-npt-300.0-traj.xyz"
    md_path = tmp_path / "Cl4Na4-npt-300.0-md.log"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    init_energy = single_point.struct.get_potential_energy()

    nvt = NVT(
        struct=single_point.struct,
        temp=300.0,
        steps=0,
        traj_file=traj_path,
        traj_every=1,
        md_file=md_path,
        output_every=1,
        minimize=True,
    )
    nvt.run()

    final_energy = single_point.struct.get_potential_energy()
    assert final_energy < init_energy


def test_reset_velocities(tmp_path):
    """Test rescaling velocities before dynamics."""
    traj_path = tmp_path / "Cl4Na4-npt-300.0-traj.xyz"
    md_path = tmp_path / "Cl4Na4-npt-300.0-md.log"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "H2O.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    # Give structure arbitrary velocities
    single_point.struct.set_velocities(np.ones((3, 3)))
    init_momentum = np.sum(single_point.struct.get_momenta())

    nvt = NVT(
        struct=single_point.struct,
        temp=300.0,
        steps=0,
        traj_file=traj_path,
        traj_every=1,
        md_file=md_path,
        output_every=1,
        rescale_velocities=True,
        equil_steps=1,
    )
    nvt.run()

    final_momentum = np.sum(single_point.struct.get_momenta())
    assert abs(final_momentum) < abs(init_momentum)
    assert init_momentum != pytest.approx(0)
    assert final_momentum == pytest.approx(0)


def test_remove_rot(tmp_path):
    """Test removing rotation before dynamics."""
    traj_path = tmp_path / "Cl4Na4-npt-300.0-traj.xyz"
    md_path = tmp_path / "Cl4Na4-npt-300.0-md.log"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "H2O.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    # Give structure arbitrary angular velocities
    single_point.struct.set_velocities(np.arange(9).reshape(3, 3))

    nvt = NVT(
        struct=single_point.struct,
        temp=300.0,
        steps=0,
        traj_file=traj_path,
        traj_every=1,
        md_file=md_path,
        output_every=1,
        rescale_velocities=True,
        equil_steps=1,
    )
    nvt.run()

    # Calculate total angular momentum based on ASE's ZeroRotation function
    _, basis = single_point.struct.get_moments_of_inertia(vectors=True)
    tot_ang_mom = np.dot(basis, single_point.struct.get_angular_momentum())
    init_ang_mom = np.sum(tot_ang_mom)

    nvt = NVT(
        struct=single_point.struct,
        temp=300.0,
        steps=0,
        traj_file=traj_path,
        traj_every=1,
        md_file=md_path,
        output_every=1,
        rescale_velocities=True,
        remove_rot=True,
        equil_steps=1,
    )
    nvt.run()

    # Calculate total angular momentum based on ASE's ZeroRotation function
    _, basis = single_point.struct.get_moments_of_inertia(vectors=True)
    tot_ang_mom = np.dot(basis, single_point.struct.get_angular_momentum())
    final_ang_mom = np.sum(tot_ang_mom)

    assert abs(final_ang_mom) < abs(init_ang_mom)
    assert init_ang_mom != pytest.approx(0)
    assert final_ang_mom == pytest.approx(0)


def test_traj_start(tmp_path):
    """Test starting trajectory after n steps."""
    traj_path = tmp_path / "Cl4Na4-nvt-nh-300.0-traj.xyz"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    nvt = NVT(
        struct=single_point.struct,
        temp=300.0,
        steps=3,
        traj_file=traj_path,
        traj_every=1,
        restart_every=100,
        output_every=100,
    )
    nvt.run()

    traj = read(traj_path, index=":")
    assert all(isinstance(image, Atoms) for image in traj)
    assert len(traj) == 4

    nvt = NVT(
        struct=single_point.struct,
        temp=300.0,
        steps=3,
        traj_file=traj_path,
        traj_every=1,
        traj_start=2,
        restart_every=100,
        output_every=100,
    )
    nvt.run()

    traj = read(traj_path, index=":")
    assert all(isinstance(image, Atoms) for image in traj)
    assert len(traj) == 2


def test_minimize_every(tmp_path):
    """Test setting minimize_every."""
    md_path = tmp_path / "Cl4Na4-npt-300.0-md.log"
    log_file = tmp_path / "nvt.log"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    nvt = NVT(
        struct=single_point.struct,
        temp=300.0,
        steps=4,
        traj_start=100,
        md_file=md_path,
        output_every=100,
        minimize=True,
        minimize_every=2,
        equil_steps=4,
        log_kwargs={"filename": log_file, "force": True},
    )
    nvt.run()

    with open(log_file, encoding="utf8") as file:
        log_txt = file.readlines()
        assert any("Minimizing at step 0" in line for line in log_txt)
        assert not any("Minimizing at step 1" in line for line in log_txt)
        assert any("Minimizing at step 2" in line for line in log_txt)


def test_rescale_every(tmp_path):
    """Test setting minimize_every."""
    md_path = tmp_path / "Cl4Na4-npt-300.0-md.log"
    log_file = tmp_path / "nvt.log"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    nvt = NVT(
        struct=single_point.struct,
        temp=300.0,
        steps=4,
        traj_start=100,
        md_file=md_path,
        output_every=1,
        rescale_velocities=True,
        remove_rot=True,
        rescale_every=3,
        equil_steps=4,
        log_kwargs={"filename": log_file, "force": True},
    )
    nvt.run()

    # Note: four timesteps required as rescaling is performed before start of step
    with open(log_file, encoding="utf8") as file:
        log_txt = file.readlines()
        assert any("Velocities reset at step 0" in line for line in log_txt)
        assert not any("Velocities reset at step 1" in line for line in log_txt)
        assert any("Velocities reset at step 3" in line for line in log_txt)

        assert any("Rotation reset at step 0" in line for line in log_txt)
        assert not any("Rotation reset at step 1" in line for line in log_txt)
        assert any("Rotation reset at step 3" in line for line in log_txt)
