"""Test molecular dynamics."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import read
import ase.md.nose_hoover_chain
from ase.md.npt import NPT as ASE_NPT
import numpy as np
import pytest

from janus_core.calculations.md import NPH, NPT, NVE, NVT, NVT_CSVR, NVT_NH
from janus_core.calculations.single_point import SinglePoint
from janus_core.helpers.mlip_calculators import choose_calculator
from janus_core.helpers.stats import Stats
from tests.utils import assert_log_contains

if hasattr(ase.md.nose_hoover_chain, "IsotropicMTKNPT"):
    from janus_core.calculations.md import NPT_MTK

    MTK_IMPORT_FAILED = False
else:
    NPT_MTK = None
    MTK_IMPORT_FAILED = True

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"

test_data = [
    (NVT, "nvt"),
    (NVE, "nve"),
    (NPT, "npt"),
    (NVT_NH, "nvt-nh"),
    (NPH, "nph"),
    (NVT_CSVR, "nvt-csvr"),
    pytest.param(
        NPT_MTK,
        "npt-mtk",
        marks=pytest.mark.skipif(
            MTK_IMPORT_FAILED, reason="Requires updated version of ASE"
        ),
    ),
]

ensembles_without_thermostat = (NVE, NPH)
ensembles_with_thermostat = (
    NVT,
    NPT,
    NVT_NH,
    NVT_CSVR,
    pytest.param(
        NPT_MTK,
        marks=pytest.mark.skipif(
            MTK_IMPORT_FAILED, reason="Requires updated version of ASE"
        ),
    ),
)


@pytest.mark.parametrize("ensemble, expected", test_data)
def test_init(ensemble, expected):
    """Test initialising ensembles."""
    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    dyn = ensemble(
        struct=single_point.struct,
    )
    assert dyn.ensemble == expected


def test_npt():
    """Test NPT molecular dynamics."""
    restart_path_1 = Path("Cl4Na4-npt-T300.0-p1.0-res-2.extxyz")
    restart_path_2 = Path("Cl4Na4-npt-T300.0-p1.0-res-4.extxyz")
    restart_final = Path("Cl4Na4-npt-T300.0-p1.0-final.extxyz")
    traj_path = Path("Cl4Na4-npt-T300.0-p1.0-traj.extxyz")
    stats_path = Path("Cl4Na4-npt-T300.0-p1.0-stats.dat")

    assert not restart_path_1.exists()
    assert not restart_path_2.exists()
    assert not restart_final.exists()
    assert not traj_path.exists()
    assert not stats_path.exists()

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
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
        restart_atoms_final = read(restart_final)
        assert isinstance(restart_atoms_final, Atoms)
        traj = read(traj_path, index=":")
        assert all(isinstance(image, Atoms) for image in traj)
        assert len(traj) == 4

        with open(stats_path, encoding="utf8") as stats_file:
            lines = stats_file.readlines()
            assert "Target_P [GPa]" in lines[0] and "Target_T [K]" in lines[0]
            assert len(lines) == 5
    finally:
        restart_path_1.unlink(missing_ok=True)
        restart_path_2.unlink(missing_ok=True)
        restart_final.unlink(missing_ok=True)
        traj_path.unlink(missing_ok=True)
        stats_path.unlink(missing_ok=True)


def test_nvt_nh():
    """Test NVT-Nosé–Hoover  molecular dynamics."""
    restart_path = Path("Cl4Na4-nvt-nh-T300.0-res-3.extxyz")
    restart_final_path = Path("Cl4Na4-nvt-nh-T300.0-final.extxyz")
    traj_path = Path("Cl4Na4-nvt-nh-T300.0-traj.extxyz")
    stats_path = Path("Cl4Na4-nvt-nh-T300.0-stats.dat")

    assert not restart_path.exists()
    assert not restart_final_path.exists()
    assert not traj_path.exists()
    assert not stats_path.exists()

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
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

        restart_final_atoms = read(restart_final_path)
        assert isinstance(restart_final_atoms, Atoms)

        traj = read(traj_path, index=":")
        assert all(isinstance(image, Atoms) for image in traj)
        assert len(traj) == 3

        with open(stats_path, encoding="utf8") as stats_file:
            lines = stats_file.readlines()
            assert " | T [K]" in lines[0]
            assert len(lines) == 4
    finally:
        restart_path.unlink(missing_ok=True)
        restart_final_path.unlink(missing_ok=True)
        traj_path.unlink(missing_ok=True)
        stats_path.unlink(missing_ok=True)


def test_nve(tmp_path):
    """Test NVE molecular dynamics."""
    file_prefix = tmp_path / "Cl4Na4-nve-T300.0"
    traj_path = tmp_path / "Cl4Na4-nve-T300.0-traj.extxyz"
    stats_path = tmp_path / "Cl4Na4-nve-T300.0-stats.dat"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
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
    restart_path = Path("Cl4Na4-nph-T300.0-p0.0-res-2.extxyz")
    restart_final_path = Path("Cl4Na4-nph-T300.0-p0.0-final.extxyz")
    traj_path = Path("Cl4Na4-nph-T300.0-p0.0-traj.extxyz")
    stats_path = Path("Cl4Na4-nph-T300.0-p0.0-stats.dat")

    assert not restart_path.exists()
    assert not restart_final_path.exists()
    assert not traj_path.exists()
    assert not stats_path.exists()

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
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

        restart_final_atoms = read(restart_final_path)
        assert isinstance(restart_final_atoms, Atoms)

        traj = read(traj_path, index=":")
        assert all(isinstance(image, Atoms) for image in traj)
        assert len(traj) == 1

        with open(stats_path, encoding="utf8") as stats_file:
            lines = stats_file.readlines()
            assert " | Epot/N [eV]" in lines[0]
            assert len(lines) == 2
    finally:
        restart_path.unlink(missing_ok=True)
        restart_final_path.unlink(missing_ok=True)
        traj_path.unlink(missing_ok=True)
        stats_path.unlink(missing_ok=True)


def test_nvt_csvr():
    """Test NVT CSVR molecular dynamics."""
    restart_path_1 = Path("NaCl-nvt-csvr-T300.0-res-2.extxyz")
    restart_path_2 = Path("NaCl-nvt-csvr-T300.0-res-4.extxyz")
    restart_final = Path("NaCl-nvt-csvr-T300.0-final.extxyz")
    traj_path = Path("NaCl-nvt-csvr-T300.0-traj.extxyz")
    stats_path = Path("NaCl-nvt-csvr-T300.0-stats.dat")

    assert not restart_path_1.exists()
    assert not restart_path_2.exists()
    assert not restart_final.exists()
    assert not traj_path.exists()
    assert not stats_path.exists()

    csvr = NVT_CSVR(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
        model_path=MODEL_PATH,
        temp=300.0,
        steps=4,
        traj_every=1,
        restart_every=2,
        stats_every=1,
        taut=10,
    )

    try:
        csvr.run()
        restart_atoms_1 = read(restart_path_1)
        assert isinstance(restart_atoms_1, Atoms)
        restart_atoms_2 = read(restart_path_2)
        assert isinstance(restart_atoms_2, Atoms)
        restart_atoms_final = read(restart_final)
        assert isinstance(restart_atoms_final, Atoms)
        traj = read(traj_path, index=":")
        assert all(isinstance(image, Atoms) for image in traj)
        assert len(traj) == 5

        with open(stats_path, encoding="utf8") as stats_file:
            lines = stats_file.readlines()
            assert "Target_T [K]" in lines[0]
            assert len(lines) == 6
    finally:
        restart_path_1.unlink(missing_ok=True)
        restart_path_2.unlink(missing_ok=True)
        restart_final.unlink(missing_ok=True)
        traj_path.unlink(missing_ok=True)
        stats_path.unlink(missing_ok=True)


@pytest.mark.skipif(MTK_IMPORT_FAILED, reason="Requires updated version of ASE")
def test_npt_mtk():
    """Test NPT MTK molecular dynamics."""
    restart_path_1 = Path("NaCl-npt-mtk-T300.0-p0.0001-res-2.extxyz")
    restart_path_2 = Path("NaCl-npt-mtk-T300.0-p0.0001-res-4.extxyz")
    restart_final = Path("NaCl-npt-mtk-T300.0-p0.0001-final.extxyz")
    traj_path = Path("NaCl-npt-mtk-T300.0-p0.0001-traj.extxyz")
    stats_path = Path("NaCl-npt-mtk-T300.0-p0.0001-stats.dat")

    assert not restart_path_1.exists()
    assert not restart_path_2.exists()
    assert not restart_final.exists()
    assert not traj_path.exists()
    assert not stats_path.exists()

    npt_mtk = NPT_MTK(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
        model_path=MODEL_PATH,
        temp=300.0,
        pressure=0.0001,
        steps=4,
        traj_every=1,
        restart_every=2,
        stats_every=1,
        thermostat_time=80.0,
        barostat_time=800.0,
        thermostat_chain=2,
        barostat_chain=2,
        thermostat_substeps=2,
        barostat_substeps=2,
    )

    try:
        npt_mtk.run()
        restart_atoms_1 = read(restart_path_1)
        assert isinstance(restart_atoms_1, Atoms)
        restart_atoms_2 = read(restart_path_2)
        assert isinstance(restart_atoms_2, Atoms)
        restart_atoms_final = read(restart_final)
        assert isinstance(restart_atoms_final, Atoms)
        traj = read(traj_path, index=":")
        assert all(isinstance(image, Atoms) for image in traj)
        assert len(traj) == 5

        with open(stats_path, encoding="utf8") as stats_file:
            lines = stats_file.readlines()
            assert "Target_T [K]" in lines[0] and "Target_P [GPa]" in lines[0]
            assert len(lines) == 6

    finally:
        restart_path_1.unlink(missing_ok=True)
        restart_path_2.unlink(missing_ok=True)
        restart_final.unlink(missing_ok=True)
        traj_path.unlink(missing_ok=True)
        stats_path.unlink(missing_ok=True)


def test_restart(tmp_path):
    """Test restarting molecular dynamics simulation."""
    file_prefix = tmp_path / "Cl4Na4-nvt-T300.0"
    traj_path = tmp_path / "Cl4Na4-nvt-T300.0-traj.extxyz"
    stats_path = tmp_path / "Cl4Na4-nvt-T300.0-stats.dat"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
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
        steps=8,
        traj_every=1,
        restart_every=4,
        stats_every=1,
        restart=True,
        restart_auto=False,
        file_prefix=file_prefix,
    )
    nvt_restart.run()
    assert nvt_restart.offset == 4

    with open(stats_path, encoding="utf8") as stats_file:
        lines = stats_file.readlines()
        assert " | Target_T [K]" in lines[0]
        # Includes step 0, and step 4 from restart
        assert len(lines) == 10
        assert int(lines[-1].split()[0]) == 8

    traj = read(traj_path, index=":")
    assert all(isinstance(image, Atoms) for image in traj)
    assert len(traj) == 9


def test_minimize(tmp_path):
    """Test geometry optimzation before dynamics."""
    file_prefix = tmp_path / "Cl4Na4-nvt-T300.0"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
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
        arch="mace",
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
        arch="mace",
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
    traj_path = tmp_path / "Cl4Na4-nvt-T300.0-traj.extxyz"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
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
        arch="mace",
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
        arch="mace",
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
    file_prefix = tmp_path / "NaCl-nvt"
    restart_path_1 = tmp_path / "NaCl-nvt-res-1.extxyz"
    restart_path_2 = tmp_path / "NaCl-nvt-res-2.extxyz"
    restart_path_3 = tmp_path / "NaCl-nvt-res-3.extxyz"
    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
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
    traj_path = tmp_path / "Cl4Na4-nvt-T300.0-traj.extxyz"
    stats_path = tmp_path / "Cl4Na4-nvt-T300.0-stats.dat"

    struct = read(DATA_PATH / "NaCl.cif")
    struct.calc = choose_calculator(arch="mace_mp", model=MODEL_PATH)

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


@pytest.mark.parametrize("ensemble, tag", test_data)
def test_stats(tmp_path, ensemble, tag):
    """Test stats file has correct structure and entries for all ensembles."""
    file_prefix = tmp_path / tag / "NaCl"
    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    md = ensemble(
        struct=single_point.struct,
        steps=2,
        stats_every=1,
        file_prefix=file_prefix,
    )
    md.run()

    stat_data = Stats(md.stats_file)

    etot_index = stat_data.labels.index("ETot/N")

    assert stat_data.columns == len(stat_data.labels)
    assert stat_data.columns == len(stat_data.units)
    assert stat_data.columns >= 16
    assert stat_data.units[etot_index] == "eV"


@pytest.mark.parametrize("ensemble", ensembles_with_thermostat)
def test_heating(tmp_path, ensemble):
    """Test heating with no MD."""
    file_prefix = tmp_path / "NaCl-heating"
    final_file = tmp_path / "NaCl-heating-final.extxyz"
    log_file = tmp_path / "NaCl.log"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    md = ensemble(
        struct=single_point.struct,
        temp=300.0,
        steps=0,
        traj_every=10,
        stats_every=10,
        file_prefix=file_prefix,
        temp_start=0.0,
        temp_end=20.0,
        temp_step=20,
        temp_time=0.5,
        log_kwargs={"filename": log_file},
    )
    md.run()
    assert_log_contains(
        log_file,
        includes=[
            "Beginning temperature ramp at 0.0K",
            "Temperature ramp complete at 20.0K",
        ],
        excludes=["Starting molecular dynamics simulation"],
    )

    assert final_file.exists()


@pytest.mark.parametrize("ensemble", ensembles_without_thermostat)
def test_no_thermostat_heating(tmp_path, ensemble):
    """Test that temperature ramp with no thermostat throws an error."""
    file_prefix = tmp_path / "NaCl-heating"
    final_file = tmp_path / "NaCl-heating-final.extxyz"
    log_file = tmp_path / "NaCl.log"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    with pytest.raises(ValueError, match="no thermostat"):
        md = ensemble(
            struct=single_point.struct,
            temp=300.0,
            steps=0,
            traj_every=10,
            stats_every=10,
            file_prefix=file_prefix,
            temp_start=0.0,
            temp_end=20.0,
            temp_step=20,
            temp_time=0.5,
            log_kwargs={"filename": log_file},
        )
        md.run()
    assert not final_file.exists()


@pytest.mark.parametrize("ensemble", ensembles_with_thermostat)
def test_noramp_heating(tmp_path, ensemble):
    """Test ValueError is thrown for invalid temperature ramp."""
    file_prefix = tmp_path / "NaCl-heating"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    with pytest.raises(ValueError):
        ensemble(
            struct=single_point.struct,
            file_prefix=file_prefix,
            temp_start=10,
            temp_end=10,
            temp_step=20,
        )


@pytest.mark.parametrize("ensemble", ensembles_with_thermostat)
def test_heating_md(tmp_path, ensemble):
    """Test heating followed by MD."""
    file_prefix = tmp_path / "NaCl-heating"
    stats_path = tmp_path / "NaCl-heating-stats.dat"
    log_file = tmp_path / "NaCl.log"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    md = ensemble(
        struct=single_point.struct,
        temp=25.0,
        steps=5,
        traj_every=100,
        stats_every=2,
        file_prefix=file_prefix,
        temp_start=10,
        temp_end=20,
        temp_step=10,
        temp_time=2,
        log_kwargs={"filename": log_file},
    )
    md.run()
    assert_log_contains(
        log_file,
        includes=[
            "Beginning temperature ramp at 10K",
            "Temperature ramp complete at 20K",
            "Starting molecular dynamics simulation",
            "Molecular dynamics simulation complete",
        ],
    )
    stat_data = Stats(stats_path)
    target_t_col = stat_data.labels.index("Target_T")

    is_ase_npt = isinstance(md.dyn, ASE_NPT)

    # ASE_NPT skips first row of output - ASE merge request submitted to fix:
    # https://gitlab.com/ase/ase/-/merge_requests/3598
    assert stat_data.rows == 4 if is_ase_npt else 5
    assert stat_data.data[0, target_t_col] == pytest.approx(10.0)
    if is_ase_npt:
        assert stat_data.data[1, target_t_col] == pytest.approx(20.0)
        assert stat_data.data[3, target_t_col] == pytest.approx(25.0)
    else:
        assert stat_data.data[2, target_t_col] == pytest.approx(20.0)
        assert stat_data.data[4, target_t_col] == pytest.approx(25.0)
    assert stat_data.labels[0] == "# Step"
    assert stat_data.units[0] == ""
    assert stat_data.units[target_t_col] == "K"


def test_heating_files():
    """Test default heating file names."""
    traj_heating_path = Path("Cl4Na4-nvt-T10-T20-traj.extxyz")
    stats_heating_path = Path("Cl4Na4-nvt-T10-T20-stats.dat")
    final_path = Path("Cl4Na4-nvt-T10-T20-final.extxyz")

    assert not traj_heating_path.exists()
    assert not stats_heating_path.exists()
    assert not final_path.exists()

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    nvt = NVT(
        struct=single_point.struct,
        temp=25.0,
        steps=0,
        traj_every=2,
        stats_every=2,
        temp_start=10,
        temp_end=20,
        temp_step=10,
        temp_time=2,
    )
    try:
        nvt.run()
        final_atoms = read(final_path, index=":")
        assert isinstance(final_atoms[0], Atoms)
        assert len(final_atoms) == 2

        traj = read(traj_heating_path, index=":")
        assert all(isinstance(image, Atoms) for image in traj)
        assert len(traj) == 3

        stats = Stats(source=stats_heating_path)
        assert stats.rows == 3
        assert stats.data[0, 16] == 10.0
        assert stats.data[1, 16] == 10.0
        assert stats.data[2, 16] == 20.0

    finally:
        traj_heating_path.unlink(missing_ok=True)
        stats_heating_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)


def test_heating_md_files():
    """Test default heating files when also running md."""
    traj_heating_path = Path("Cl4Na4-nvt-T10-T20-T25.0-traj.extxyz")
    stats_heating_path = Path("Cl4Na4-nvt-T10-T20-T25.0-stats.dat")
    final_path = Path("Cl4Na4-nvt-T10-T20-T25.0-final.extxyz")

    assert not traj_heating_path.exists()
    assert not stats_heating_path.exists()
    assert not final_path.exists()

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    nvt = NVT(
        struct=single_point.struct,
        temp=25.0,
        steps=2,
        traj_every=100,
        stats_every=2,
        temp_start=10,
        temp_end=20,
        temp_step=10,
        temp_time=2,
    )
    try:
        nvt.run()
        final_atoms = read(final_path, index=":")
        assert len(final_atoms) == 3
        assert isinstance(final_atoms[0], Atoms)

        traj = read(traj_heating_path, index=":")
        assert all(isinstance(image, Atoms) for image in traj)
        assert len(traj) == 1

        stats = Stats(source=stats_heating_path)
        assert stats.rows == 4
        assert stats.data[0, 16] == 10
        assert stats.data[2, 16] == 20
        assert stats.data[3, 16] == 25

    finally:
        traj_heating_path.unlink(missing_ok=True)
        stats_heating_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)


def test_ramp_negative(tmp_path):
    """Test ValueError is thrown for negative values in temperature ramp."""
    file_prefix = tmp_path / "NaCl-heating"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    with pytest.raises(ValueError):
        NVT(
            struct=single_point.struct,
            file_prefix=file_prefix,
            temp_start=-5,
            temp_end=10,
            temp_step=20,
            temp_time=10,
        )
    with pytest.raises(ValueError):
        NVT(
            struct=single_point.struct,
            file_prefix=file_prefix,
            temp_start=5,
            temp_end=-5,
            temp_step=20,
            temp_time=10,
        )


def test_cooling(tmp_path):
    """Test cooling with no MD."""
    file_prefix = tmp_path / "NaCl-cooling"
    final_path = tmp_path / "NaCl-cooling-final.extxyz"
    stats_cooling_path = tmp_path / "NaCl-cooling-stats.dat"
    log_file = tmp_path / "nvt.log"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    nvt = NVT(
        struct=single_point.struct,
        temp=300.0,
        steps=0,
        traj_every=10,
        stats_every=1,
        file_prefix=file_prefix,
        temp_start=20.0,
        temp_end=0.0,
        temp_step=10,
        temp_time=0.5,
        log_kwargs={"filename": log_file},
    )
    nvt.run()
    assert_log_contains(
        log_file,
        includes=[
            "Beginning temperature ramp at 20.0K",
            "Temperature ramp complete at 0.0K",
        ],
        excludes=["Starting molecular dynamics simulation"],
    )
    assert final_path.exists()
    final_atoms = read(final_path, index=":")
    assert isinstance(final_atoms[0], Atoms)
    assert len(final_atoms) == 3

    stats = Stats(source=stats_cooling_path)
    assert stats.rows == 2
    assert stats.data[0, 16] == 20.0
    assert stats.data[1, 16] == 10.0


def test_ensemble_kwargs(tmp_path):
    """Test setting integrator kwargs."""
    log_file = tmp_path / "nvt.log"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    npt = NPT(
        struct=single_point.struct,
        ensemble_kwargs={"mask": (0, 1, 0)},
        log_kwargs={"filename": log_file},
    )

    expected_mask = [[False, False, False], [False, True, False], [False, False, False]]
    assert np.array_equal(npt.dyn.mask, expected_mask)


def test_invalid_struct():
    """Test setting invalid structure."""
    single_point = SinglePoint(
        struct_path=DATA_PATH / "benzene-traj.xyz",
        arch="mace_mp",
        calc_kwargs={"model": MODEL_PATH},
    )

    with pytest.raises(NotImplementedError):
        NPT(
            single_point.struct,
        )
    with pytest.raises(ValueError):
        NPT(
            "structure",
        )


def test_logging(tmp_path):
    """Test attaching logger to NVT and emissions are saved to info."""
    log_file = tmp_path / "md.log"
    file_prefix = tmp_path / "md"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace_mp",
        calc_kwargs={"model": MODEL_PATH},
    )

    assert "emissions" not in single_point.struct.info

    nvt = NVT(
        struct=single_point.struct,
        temp=300.0,
        steps=10,
        log_kwargs={"filename": log_file},
        file_prefix=file_prefix,
    )
    nvt.run()

    assert log_file.exists()
    assert "emissions" in single_point.struct.info
    assert single_point.struct.info["emissions"] > 0


def test_auto_restart(tmp_path):
    """Test auto restarting simulation."""
    # tmp_path for all files other than restart
    # Include T300.0 to test Path.stem vs Path.name
    final_path = tmp_path / "md-T300.0-final.extxyz"
    traj_path = tmp_path / "md-T300.0-traj.extxyz"
    stats_path = tmp_path / "md-T300.0-stats.dat"
    log_file = tmp_path / "md.log"

    # Predicted restart file, from defaults
    restart_path = Path(".").absolute() / "NaCl-nvt-T300.0-res-4.extxyz"
    assert not restart_path.exists()

    try:
        nvt = NVT(
            struct_path=DATA_PATH / "NaCl.cif",
            arch="mace_mp",
            calc_kwargs={"model": MODEL_PATH},
            temp=300.0,
            steps=4,
            stats_file=stats_path,
            stats_every=3,
            traj_file=traj_path,
            traj_every=1,
            final_file=final_path,
            restart_every=4,
            timestep=100,
        )
        nvt.run()

        assert nvt.dyn.nsteps == 4
        assert stats_path.exists()
        assert traj_path.exists()
        assert restart_path.exists()

        with open(stats_path, encoding="utf8") as stats_file:
            lines = stats_file.readlines()
        # Header and steps 0 & 3 only in stats
        assert len(lines) == 3
        assert int(lines[-1].split()[0]) == 3

        restart_struct = read(restart_path)
        traj_struct = read(traj_path, index=-1)

        # Prepare with auto restart from step 4
        nvt_restart = NVT(
            struct_path=DATA_PATH / "NaCl.cif",
            arch="mace_mp",
            calc_kwargs={"model": MODEL_PATH},
            temp=300.0,
            steps=7,
            stats_every=1,
            restart=True,
            restart_auto=True,
            stats_file=stats_path,
            traj_file=traj_path,
            traj_every=1,
            final_file=final_path,
            log_kwargs={"filename": log_file},
        )

        assert_log_contains(log_file, includes="Auto restart successful")

        # Check saved restart struct is same as last trajectory struct
        assert restart_struct.positions == pytest.approx(traj_struct.positions)
        # Check saved restart struct is starting structure for new sim
        assert restart_struct.positions == pytest.approx(nvt_restart.struct.positions)
        # Check starting structure for new sim is same as final structure of old sim
        assert nvt.struct.positions == pytest.approx(
            nvt_restart.struct.positions, rel=1e-6
        )

        # Offset from restart file, not stats
        assert nvt_restart.offset == 4

        nvt_restart.run()

        # Check outputs are appended
        with open(stats_path, encoding="utf8") as stats_file:
            lines = stats_file.readlines()
        # Header and steps 0 & 3, and 5, 6 and 7
        assert len(lines) == 6
        assert int(lines[-1].split()[0]) == 7

        final_traj = read(traj_path, index=":")
        assert len(final_traj) == 8

    finally:
        restart_path.unlink(missing_ok=True)


def test_auto_restart_restart_stem(tmp_path):
    """Test auto restarting simulation with restart stem defined."""
    file_prefix = tmp_path / "npt"
    traj_path = tmp_path / "npt-traj.extxyz"
    stats_path = tmp_path / "npt-stats.dat"
    log_file = tmp_path / "md.log"

    # Set restart stem and predict restart file
    restart_stem = tmp_path / "npt-test"
    restart_path = tmp_path / "npt-test-4.extxyz"

    npt = NPT(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace_mp",
        calc_kwargs={"model": MODEL_PATH},
        temp=300.0,
        steps=6,
        stats_every=3,
        traj_every=1,
        file_prefix=file_prefix,
        restart_stem=str(restart_stem),
        restart_every=4,
        timestep=10,
    )
    npt.run()

    assert npt.dyn.nsteps == 6
    assert stats_path.exists()
    assert traj_path.exists()
    assert restart_path.exists()

    with open(stats_path, encoding="utf8") as stats_file:
        lines = stats_file.readlines()
    # Header and steps 3, 6 only in stats
    assert len(lines) == 3
    assert int(lines[-1].split()[0]) == 6

    restart_struct = read(restart_path)
    # Initial structure not saved by npt, so 4th structure from traj
    traj_struct = read(traj_path, index=3)

    # Prepare with auto restart from step 4
    npt_restart = NPT(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace_mp",
        calc_kwargs={"model": MODEL_PATH},
        temp=300.0,
        steps=7,
        stats_every=1,
        restart_stem=restart_stem,
        restart=True,
        restart_auto=True,
        file_prefix=file_prefix,
        traj_every=1,
        log_kwargs={"filename": log_file},
    )

    assert_log_contains(log_file, includes="Auto restart successful")

    # Check saved restart struct is same as last trajectory struct
    assert restart_struct.positions == pytest.approx(traj_struct.positions)
    # Check saved restart struct is starting structure for new sim
    assert restart_struct.positions == pytest.approx(npt_restart.struct.positions)
    # Check starting structure for new sim is different to final structure of old sim
    assert npt.struct.positions != pytest.approx(npt_restart.struct.positions, rel=1e-6)

    # Offset from restart file, not stats
    assert npt_restart.offset == 4

    npt_restart.run()

    # Check outputs are appended
    with open(stats_path, encoding="utf8") as stats_file:
        lines = stats_file.readlines()
    # Header and steps 0, 3, 6 and 5, 6 and 7
    assert len(lines) == 6
    assert int(lines[-1].split()[0]) == 7

    final_traj = read(traj_path, index=":")
    assert len(final_traj) == 9


def test_set_info(tmp_path):
    """Test info is set at correct frequency."""
    file_prefix = tmp_path / "npt"
    traj_path = tmp_path / "npt-traj.extxyz"

    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        arch="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    npt = NPT(
        struct=single_point.struct,
        steps=10,
        temp=1000,
        stats_every=7,
        file_prefix=file_prefix,
        seed=2024,
        traj_every=10,
    )

    npt.run()
    final_struct = read(traj_path, index="-1")
    assert npt.struct.info["density"] == pytest.approx(2.120952627887493)
    assert final_struct.info["density"] == pytest.approx(2.120952627887493)
