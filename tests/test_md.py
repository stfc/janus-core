"""Test molecular dynamics."""

from pathlib import Path

from ase import Atoms
from ase.io import read
import numpy as np
import pytest

from janus_core.md import NPH, NPT, NVE, NVT, NVT_NH
from janus_core.mlip_calculators import choose_calculator
from janus_core.single_point import SinglePoint
from tests.utils import assert_log_contains

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


def test_npt():
    """Test NPT molecular dynamics."""
    restart_path_1 = Path("Cl4Na4-npt-T300.0-p1.0-res-2.xyz")
    restart_path_2 = Path("Cl4Na4-npt-T300.0-p1.0-res-4.xyz")
    traj_path = Path("Cl4Na4-npt-T300.0-p1.0-traj.xyz")
    stats_path = Path("Cl4Na4-npt-T300.0-p1.0-stats.dat")

    assert not restart_path_1.exists()
    assert not restart_path_2.exists()
    assert not traj_path.exists()
    assert not stats_path.exists()

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
        traj_every=1,
        restart_every=2,
        stats_every=1,
    )

    try:
        npt.run()

        restart_atoms_1 = read(restart_path_1)
        assert isinstance(restart_atoms_1, Atoms)
        restart_atoms_2 = read(restart_path_2)
        assert isinstance(restart_atoms_2, Atoms)

        traj = read(traj_path, index=":")
        assert all(isinstance(image, Atoms) for image in traj)
        assert len(traj) == 4

        with open(stats_path, encoding="utf8") as stats_file:
            lines = stats_file.readlines()
            assert "Target P [bar] | Target T [K]" in lines[0]
            assert len(lines) == 5
    finally:
        restart_path_1.unlink(missing_ok=True)
        restart_path_2.unlink(missing_ok=True)
        traj_path.unlink(missing_ok=True)
        stats_path.unlink(missing_ok=True)


def test_nvt_nh():
    """Test NVT-Nosé–Hoover  molecular dynamics."""
    restart_path = Path("Cl4Na4-nvt-nh-T300.0-res-3.xyz")
    traj_path = Path("Cl4Na4-nvt-nh-T300.0-traj.xyz")
    stats_path = Path("Cl4Na4-nvt-nh-T300.0-stats.dat")

    assert not restart_path.exists()
    assert not traj_path.exists()
    assert not stats_path.exists()

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    nvt_nh = NVT_NH(
        struct=single_point.struct,
        temp=300.0,
        steps=3,
        traj_every=1,
        restart_every=3,
        stats_every=1,
    )

    try:
        nvt_nh.run()

        restart_atoms = read(restart_path)
        assert isinstance(restart_atoms, Atoms)

        traj = read(traj_path, index=":")
        assert all(isinstance(image, Atoms) for image in traj)
        assert len(traj) == 3

        with open(stats_path, encoding="utf8") as stats_file:
            lines = stats_file.readlines()
            assert " | T [K]" in lines[0]
            assert len(lines) == 4
    finally:
        restart_path.unlink(missing_ok=True)
        traj_path.unlink(missing_ok=True)
        stats_path.unlink(missing_ok=True)


def test_nve(tmp_path):
    """Test NVE molecular dynamics."""
    file_prefix = tmp_path / "Cl4Na4-nve-T300.0"
    traj_path = tmp_path / "Cl4Na4-nve-T300.0-traj.xyz"
    stats_path = tmp_path / "Cl4Na4-nve-T300.0-stats.dat"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    nve = NVE(
        struct=single_point.struct,
        temp=300.0,
        steps=4,
        traj_every=3,
        stats_every=1,
        file_prefix=file_prefix,
    )
    nve.run()

    traj = read(traj_path, index=":")
    assert all(isinstance(image, Atoms) for image in traj)
    assert len(traj) == 2

    with open(stats_path, encoding="utf8") as stats_file:
        lines = stats_file.readlines()
        assert " | Epot/N [eV]" in lines[0]
        # Includes step 0
        assert len(lines) == 6


def test_nph():
    """Test NPH molecular dynamics."""
    restart_path = Path("Cl4Na4-nph-T300.0-p0.0-res-2.xyz")
    traj_path = Path("Cl4Na4-nph-T300.0-p0.0-traj.xyz")
    stats_path = Path("Cl4Na4-nph-T300.0-p0.0-stats.dat")

    assert not restart_path.exists()
    assert not traj_path.exists()
    assert not stats_path.exists()

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    nph = NPH(
        struct=single_point.struct,
        temp=300.0,
        steps=3,
        traj_every=2,
        restart_every=4,
        stats_every=2,
    )

    try:
        nph.run()

        with pytest.raises(FileNotFoundError):
            read(restart_path)

        traj = read(traj_path, index=":")
        assert all(isinstance(image, Atoms) for image in traj)
        assert len(traj) == 1

        with open(stats_path, encoding="utf8") as stats_file:
            lines = stats_file.readlines()
            assert " | Epot/N [eV]" in lines[0]
            assert len(lines) == 2
    finally:
        restart_path.unlink(missing_ok=True)
        traj_path.unlink(missing_ok=True)
        stats_path.unlink(missing_ok=True)


def test_restart(tmp_path):
    """Test restarting molecular dynamics simulation."""
    file_prefix = tmp_path / "Cl4Na4-nvt-T300.0"
    traj_path = tmp_path / "Cl4Na4-nvt-T300.0-traj.xyz"
    stats_path = tmp_path / "Cl4Na4-nvt-T300.0-stats.dat"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    nvt = NVT(
        struct=single_point.struct,
        temp=300.0,
        steps=4,
        traj_every=1,
        restart_every=4,
        stats_every=1,
        file_prefix=file_prefix,
    )
    nvt.run()

    assert nvt.dyn.nsteps == 4

    nvt_restart = NVT(
        struct=single_point.struct,
        temp=300.0,
        steps=4,
        traj_every=1,
        restart_every=4,
        stats_every=1,
        restart=True,
        file_prefix=file_prefix,
    )
    nvt_restart.run()
    assert nvt_restart.offset == 4

    with open(stats_path, encoding="utf8") as stats_file:
        lines = stats_file.readlines()
        assert " | Target T [K]" in lines[0]
        # Includes step 0, and step 4 from restart
        assert len(lines) == 11
        assert 8 == int(lines[-1].split()[0])

    traj = read(traj_path, index=":")
    assert all(isinstance(image, Atoms) for image in traj)
    assert len(traj) == 10


def test_minimize(tmp_path):
    """Test geometry optimzation before dynamics."""
    file_prefix = tmp_path / "Cl4Na4-nvt-T300.0"

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
        traj_every=1,
        stats_every=1,
        minimize=True,
        file_prefix=file_prefix,
    )
    nvt.run()

    final_energy = single_point.struct.get_potential_energy()
    assert final_energy < init_energy


def test_reset_velocities(tmp_path):
    """Test rescaling velocities before dynamics."""
    file_prefix = tmp_path / "Cl4Na4-nvt-T300.0"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "H2O.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    # Give structure arbitrary velocities
    single_point.struct.set_velocities(np.ones((3, 3)))
    init_momentum = np.sum(single_point.struct.get_momenta())

    with pytest.warns(UserWarning, match="Velocities and angular momentum will not"):
        nvt = NVT(
            struct=single_point.struct,
            temp=300.0,
            steps=0,
            traj_every=1,
            stats_every=1,
            rescale_velocities=True,
            equil_steps=1,
            file_prefix=file_prefix,
        )
    nvt.run()

    final_momentum = np.sum(single_point.struct.get_momenta())
    assert abs(final_momentum) < abs(init_momentum)
    assert init_momentum != pytest.approx(0)
    assert final_momentum == pytest.approx(0)


def test_remove_rot(tmp_path):
    """Test removing rotation before dynamics."""
    file_prefix = tmp_path / "Cl4Na4-nvt-T300.0"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "H2O.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    # Give structure arbitrary angular velocities
    single_point.struct.set_velocities(np.arange(9).reshape(3, 3))

    # Calculate total angular momentum based on ASE's ZeroRotation function
    _, basis = single_point.struct.get_moments_of_inertia(vectors=True)
    tot_ang_mom = np.dot(basis, single_point.struct.get_angular_momentum())
    init_ang_mom = np.sum(tot_ang_mom)

    with pytest.warns(UserWarning, match="Velocities and angular momentum will not"):
        nvt = NVT(
            struct=single_point.struct,
            temp=300.0,
            steps=0,
            traj_every=1,
            stats_every=1,
            rescale_velocities=True,
            remove_rot=True,
            equil_steps=1,
            file_prefix=file_prefix,
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
    file_prefix = tmp_path / "Cl4Na4-nvt-T300.0"
    traj_path = tmp_path / "Cl4Na4-nvt-T300.0-traj.xyz"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    nvt = NVT(
        struct=single_point.struct,
        temp=300.0,
        steps=3,
        traj_every=1,
        restart_every=100,
        stats_every=100,
        file_prefix=file_prefix,
    )
    nvt.run()

    traj = read(traj_path, index=":")
    assert all(isinstance(image, Atoms) for image in traj)
    assert len(traj) == 4

    nvt = NVT(
        struct=single_point.struct,
        temp=300.0,
        steps=3,
        traj_every=1,
        traj_start=2,
        restart_every=100,
        stats_every=100,
        file_prefix=file_prefix,
    )
    nvt.run()

    traj = read(traj_path, index=":")
    assert all(isinstance(image, Atoms) for image in traj)
    assert len(traj) == 2


def test_minimize_every(tmp_path):
    """Test setting minimize_every."""
    file_prefix = tmp_path / "Cl4Na4-nvt-T300.0"
    log_file = tmp_path / "nvt.log"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    nvt = NVT(
        struct=single_point.struct,
        temp=300.0,
        steps=6,
        traj_start=100,
        stats_every=100,
        minimize=True,
        minimize_every=2,
        equil_steps=4,
        file_prefix=file_prefix,
        log_kwargs={"filename": log_file},
    )
    nvt.run()
    assert_log_contains(
        log_file,
        includes=["Minimizing at step 0", "Minimizing at step 2"],
        excludes=["Minimizing at step 1", "Minimizing at step 4"],
    )


def test_rescale_every(tmp_path):
    """Test setting minimize_every."""
    file_prefix = tmp_path / "Cl4Na4-nvt-T300.0"
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
        stats_every=1,
        rescale_velocities=True,
        remove_rot=True,
        rescale_every=3,
        equil_steps=4,
        file_prefix=file_prefix,
        log_kwargs={"filename": log_file},
    )
    nvt.run()
    # Note: four timesteps required as rescaling is performed before start of step
    assert_log_contains(
        log_file,
        includes=[
            "Velocities reset at step 0",
            "Velocities reset at step 3",
            "Rotation reset at step 0",
            "Rotation reset at step 3",
        ],
        excludes=[
            "Velocities reset at step 1",
            "Minimizing at step 4",
            "Rotation reset at step 1",
        ],
    )


def test_rotate_restart(tmp_path):
    """Test setting rotate_restart."""
    file_prefix = tmp_path / "Cl4Na4-nvt-T300.0"
    restart_path_1 = tmp_path / "Cl4Na4-nvt-T300.0-res-1.xyz"
    restart_path_2 = tmp_path / "Cl4Na4-nvt-T300.0-res-2.xyz"
    restart_path_3 = tmp_path / "Cl4Na4-nvt-T300.0-res-3.xyz"
    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    nvt = NVT(
        struct=single_point.struct,
        temp=300.0,
        steps=3,
        traj_start=100,
        stats_every=100,
        rotate_restart=True,
        restart_every=1,
        restarts_to_keep=2,
        file_prefix=file_prefix,
    )
    nvt.run()

    assert not restart_path_1.exists()
    assert restart_path_2.exists()
    assert restart_path_3.exists()


def test_atoms_struct(tmp_path):
    """Test restarting NVT molecular dynamics."""
    file_prefix = tmp_path / "Cl4Na4-nvt-T300.0"
    traj_path = tmp_path / "Cl4Na4-nvt-T300.0-traj.xyz"
    stats_path = tmp_path / "Cl4Na4-nvt-T300.0-stats.dat"

    struct = read(DATA_PATH / "NaCl.cif")
    struct.calc = choose_calculator(architecture="mace_mp", model=MODEL_PATH)

    nvt = NVT(
        struct=struct,
        temp=300.0,
        steps=4,
        stats_every=1,
        traj_every=1,
        file_prefix=file_prefix,
    )
    nvt.run()

    traj = read(traj_path, index=":")
    assert len(traj) == 5

    with open(stats_path, encoding="utf8") as stats_file:
        lines = stats_file.readlines()
        assert " | Epot/N [eV]" in lines[0]
        # Includes step 0
        assert len(lines) == 6


def test_heating(tmp_path):
    """Test heating with no MD."""
    file_prefix = tmp_path / "NaCl-heating"
    log_file = tmp_path / "nvt.log"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    nvt = NVT(
        struct=single_point.struct,
        temp=300.0,
        steps=0,
        traj_every=10,
        stats_every=1,
        file_prefix=file_prefix,
        temp_start=10,
        temp_end=40,
        temp_step=20,
        temp_time=0.5,
        log_kwargs={"filename": log_file},
    )
    nvt.run()
    assert_log_contains(
        log_file,
        includes=[
            "Beginning temperature ramp at 10K",
            "Temperature ramp complete at 30K",
        ],
        excludes=["Starting molecular dynamics simulation"],
    )


def test_noramp_heating(tmp_path):
    """Test ValueError is thrown for invalid temperature ramp."""
    file_prefix = tmp_path / "NaCl-heating"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    with pytest.raises(ValueError):
        NVT(
            struct=single_point.struct,
            file_prefix=file_prefix,
            temp_start=10,
            temp_end=10,
            temp_step=20,
        )


def test_heating_md(tmp_path):
    """Test heating followed by MD."""
    file_prefix = tmp_path / "NaCl-heating"
    log_file = tmp_path / "nvt.log"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    nvt = NVT(
        struct=single_point.struct,
        temp=25.0,
        steps=5,
        traj_every=100,
        stats_every=1,
        file_prefix=file_prefix,
        temp_start=10,
        temp_end=20,
        temp_step=10,
        temp_time=0.5,
        log_kwargs={"filename": log_file},
    )
    nvt.run()
    assert_log_contains(
        log_file,
        includes=[
            "Beginning temperature ramp at 10K",
            "Temperature ramp complete at 20K",
            "Starting molecular dynamics simulation",
            "Molecular dynamics simulation complete",
        ],
    )
