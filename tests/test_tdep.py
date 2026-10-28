"""Tests for TDEP input generation."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from ase import Atoms
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
import numpy as np
import pytest
from typer.testing import CliRunner
import yaml

from janus_core.calculations.md import NPH, NPT, NVE, NVT, NVT_CSVR, NVT_NH
from janus_core.cli.janus import app
from janus_core.processing.tdep import build_tdep_inputs_from_nvt
from tests.utils import assert_log_contains, chdir, check_output_files

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"

# Two-atom primitive cell of NaCl.cif, so NaCl.cif is a non-diagonal supercell of it
UNIT_CELL = DATA_PATH / "NaCl-primitive.extxyz"

# Committed NVT output for 64-atom NaCl at 300 K, written by
# NVT(struct=SUPERCELL, arch="mace", model=MODEL_PATH, temp=300, timestep=1,
# steps=2000, seed=2024, traj_start=1100, traj_every=100, stats_every=100,
# write_kwargs={"invalidate_calc": True})
SUPERCELL = DATA_PATH / "NaCl-tdep-supercell.extxyz"
TRAJ = DATA_PATH / "NaCl-tdep-traj.extxyz"
STATS = DATA_PATH / "NaCl-tdep-stats.dat"

TDEP_FILES = [
    "infile.forces",
    "infile.meta",
    "infile.positions",
    "infile.ssposcar",
    "infile.stat",
    "infile.ucposcar",
]

# NVT of NaCl.cif with TDEP inputs, which need at least one MD step
MD_KWARGS = {
    "struct": DATA_PATH / "NaCl.cif",
    "arch": "mace",
    "model": MODEL_PATH,
    "steps": 100,
}
TDEP_KWARGS = {
    "tdep_compute": True,
    "tdep_unit_cell_file": DATA_PATH / "NaCl.cif",
    "tdep_supercell_file": DATA_PATH / "NaCl.cif",
}

# Heating from 0 K to 100 K in 10 K steps of 1 fs at 0.1 fs, which ends at step 90
RAMP = {
    "timestep": 0.1,
    "temp_start": 0,
    "temp_end": 100,
    "temp_step": 10,
    "temp_time": 1,
    "traj_every": 10,
    "stats_every": 10,
}

NACL = read(DATA_PATH / "NaCl.cif")
PRIMITIVE = bulk("NaCl", "rocksalt", a=5.64)

# NaCl.cif with two Na atoms swapped, so only their positions differ
NACL_SWAPPED = NACL.copy()
NACL_SWAPPED.positions[[0, 2]] = NACL.positions[[2, 0]]

# TDEP's programs are found on PATH, where its INSTALL.md puts them
TDEP_INSTALLED = all(
    shutil.which(program)
    for program in ("extract_forceconstants", "phonon_dispersion_relations")
)
requires_tdep = pytest.mark.skipif(not TDEP_INSTALLED, reason="TDEP not installed")

runner = CliRunner()


def test_nvt_tdep(tmp_path):
    """Test TDEP input files written through NVT post-processing."""
    output_dir = tmp_path / "tdep-inputs"
    log_file = tmp_path / "md-log.yml"
    nvt = NVT(
        **(MD_KWARGS | {"steps": 10, "traj_every": 2, "stats_every": 1}),
        file_prefix=tmp_path / "NaCl",
        log_kwargs={"filename": log_file},
        post_process_kwargs=TDEP_KWARGS
        | {"tdep_unit_cell_file": UNIT_CELL, "tdep_output_dir": output_dir},
    )
    assert nvt.output_files["tdep_inputs"] == output_dir
    assert not output_dir.exists()

    nvt.run()
    assert sorted(path.name for path in output_dir.iterdir()) == TDEP_FILES

    meta = (output_dir / "infile.meta").read_text(encoding="utf-8").splitlines()
    assert meta == [
        "8    # N atoms",
        "6    # N timesteps",
        "2.0    # timestep in fs",
        "300    # temperature in K",
    ]
    assert np.loadtxt(output_dir / "infile.positions").shape == (48, 3)
    assert np.loadtxt(output_dir / "infile.forces").shape == (48, 3)

    # Frames at steps 0, 2, ..., 10, each paired with the stats row of its step
    stat = np.loadtxt(output_dir / "infile.stat")
    assert stat[:, 1] == pytest.approx([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    assert_log_contains(log_file, includes="TDEP input files written to")


def test_build_inputs(tmp_path):
    """Test TDEP input files written from committed NVT output."""
    output_dir = tmp_path / "tdep-inputs"
    paths = build_tdep_inputs_from_nvt(
        unit_cell_file=UNIT_CELL,
        supercell=read(SUPERCELL),
        traj_file=TRAJ,
        stats_file=STATS,
        temperature=300.0,
        output_dir=output_dir,
        timestep=100.0,
    )
    assert sorted(paths) == TDEP_FILES
    assert sorted(path.name for path in output_dir.iterdir()) == TDEP_FILES

    meta = (output_dir / "infile.meta").read_text(encoding="utf-8").splitlines()
    assert meta == [
        "64    # N atoms",
        "10    # N timesteps",
        "100.0    # timestep in fs",
        "300.0    # temperature in K",
    ]
    assert np.loadtxt(output_dir / "infile.positions").shape == (640, 3)

    # The frames have no calculator, so the forces come from mace_forces
    trajectory = read(TRAJ, index=":")
    forces = np.concatenate([atoms.arrays["mace_forces"] for atoms in trajectory])
    assert np.loadtxt(output_dir / "infile.forces") == pytest.approx(forces, abs=1e-11)

    # Each frame is paired with the stats row of its step, out of 21 rows
    steps = [atoms.info["step"] for atoms in trajectory]
    stats = np.loadtxt(STATS)
    stats = stats[np.isin(stats[:, 0], steps)]
    stat = np.loadtxt(output_dir / "infile.stat")
    assert len(stats) == len(stat) == 10
    assert stat[:, 1] == pytest.approx(stats[:, 2])

    # Off-diagonal stresses in the order TDEP reads them: xy, yz, xz
    assert stat[:, 10:13] == pytest.approx(stats[:, [15, 13, 14]])


def test_build_inputs_calc_forces(tmp_path):
    """Test forces are taken from the calculator without an <arch>_forces array."""
    frames = read(TRAJ, index=":2")
    for atoms in frames:
        forces = atoms.arrays.pop("mace_forces")
        atoms.calc = SinglePointCalculator(atoms, forces=forces)
    write(tmp_path / "traj.extxyz", frames)

    output_dir = tmp_path / "tdep-inputs"
    build_tdep_inputs_from_nvt(
        unit_cell_file=UNIT_CELL,
        supercell=SUPERCELL,
        traj_file=tmp_path / "traj.extxyz",
        stats_file=STATS,
        temperature=300.0,
        output_dir=output_dir,
    )
    forces = np.concatenate([atoms.calc.results["forces"] for atoms in frames])
    assert np.loadtxt(output_dir / "infile.forces") == pytest.approx(forces, abs=1e-11)


@pytest.mark.parametrize(
    "case, match",
    [
        ("no_forces", "must contain 'forces' or '<arch>_forces'"),
        ("no_step", "must contain 'step'"),
        ("same_step", "Frames in traj_file must have unique steps"),
        ("repeated_stats_row", "Rows in stats_file must have unique steps"),
        ("missing_stats_row", "stats_file has no row for 1 trajectory step"),
        ("small_supercell", "number of atoms in supercell must match"),
    ],
)
def test_build_inputs_errors(tmp_path, case, match):
    """Test invalid MD output is rejected before any file is written."""
    frames = read(TRAJ, index=":3")
    stats = np.loadtxt(STATS)
    supercell = SUPERCELL
    if case == "no_forces":
        for atoms in frames:
            del atoms.arrays["mace_forces"]
    elif case == "no_step":
        del frames[1].info["step"]
    elif case == "same_step":
        frames[1].info["step"] = frames[0].info["step"]
    elif case == "repeated_stats_row":
        stats = np.vstack([stats, stats[-1]])
    elif case == "missing_stats_row":
        stats = stats[stats[:, 0] != frames[1].info["step"]]
    elif case == "small_supercell":
        supercell = UNIT_CELL
    write(tmp_path / "traj.extxyz", frames)
    header = STATS.read_text(encoding="utf-8").splitlines()[0]
    np.savetxt(tmp_path / "stats.dat", stats, header=header, comments="")

    output_dir = tmp_path / "tdep-inputs"
    with pytest.raises(ValueError, match=match):
        build_tdep_inputs_from_nvt(
            unit_cell_file=UNIT_CELL,
            supercell=supercell,
            traj_file=tmp_path / "traj.extxyz",
            stats_file=tmp_path / "stats.dat",
            temperature=300.0,
            output_dir=output_dir,
        )
    assert not output_dir.exists()


def test_default_output_dir(tmp_path):
    """Test the default TDEP output directory is not created before the MD."""
    with chdir(tmp_path):
        nvt = NVT(**MD_KWARGS, post_process_kwargs=TDEP_KWARGS)
        output_dir = nvt.output_files["tdep_inputs"]
        assert output_dir == tmp_path / "janus_results" / "NaCl-nvt-T300-tdep-inputs"
        assert not output_dir.exists()


@pytest.mark.parametrize("ensemble", [NVT_NH, NVT_CSVR])
def test_nvt_ensembles(tmp_path, ensemble):
    """Test TDEP inputs are allowed for the other NVT ensembles."""
    md = ensemble(
        **MD_KWARGS, file_prefix=tmp_path / "NaCl", post_process_kwargs=TDEP_KWARGS
    )
    assert md.output_files["tdep_inputs"] == tmp_path / "NaCl-tdep-inputs"


@pytest.mark.parametrize("ensemble", [NVE, NPT, NPH])
def test_invalid_ensembles(tmp_path, ensemble):
    """Test TDEP inputs are rejected for ensembles other than NVT."""
    with pytest.raises(ValueError, match="can only be generated from NVT"):
        ensemble(
            **MD_KWARGS, file_prefix=tmp_path / "NaCl", post_process_kwargs=TDEP_KWARGS
        )


@pytest.mark.parametrize(
    "md_kwargs, tdep_kwargs, error, match",
    [
        ({}, {"tdep_unit_cell_file": None}, ValueError, "is required"),
        ({}, {"tdep_supercell_file": PRIMITIVE}, TypeError, "must be a path"),
        (
            {},
            {"tdep_unit_cell_file": DATA_PATH / "missing.cif"},
            FileNotFoundError,
            "not found",
        ),
        ({"plumed_input": "plumed.dat"}, {}, ValueError, "biased"),
        ({"minimize": True}, {}, ValueError, "filter_class"),
        ({"traj_every": 5, "stats_every": 10}, {}, ValueError, "positive multiple"),
        ({"traj_every": 0}, {}, ValueError, "positive multiple"),
        (RAMP | {"traj_start": 90}, {}, ValueError, "ramp ends at step 90"),
        ({"steps": 0}, {}, ValueError, "must be greater than 0"),
        (
            {"steps": 10, "traj_start": 20, "traj_every": 10, "stats_every": 10},
            {},
            ValueError,
            "No trajectory frames",
        ),
        (
            {},
            {"tdep_supercell_file": DATA_PATH / "NaCl-sg.cif"},
            ValueError,
            "same cell",
        ),
    ],
)
def test_invalid_settings(tmp_path, md_kwargs, tdep_kwargs, error, match):
    """Test invalid TDEP settings are rejected before the MD runs."""
    with pytest.raises(error, match=match):
        NVT(
            **(MD_KWARGS | md_kwargs),
            file_prefix=tmp_path / "NaCl",
            post_process_kwargs=TDEP_KWARGS | tdep_kwargs,
        )


@pytest.mark.parametrize(
    "md_kwargs",
    [
        {"minimize": True, "minimize_kwargs": {"filter_class": None}},
        RAMP | {"traj_start": 91},
        {"steps": 20, "traj_start": 20, "traj_every": 10, "stats_every": 10},
    ],
)
def test_valid_settings(tmp_path, md_kwargs):
    """Test settings at the limits of the TDEP checks are accepted."""
    nvt = NVT(
        **(MD_KWARGS | md_kwargs),
        file_prefix=tmp_path / "NaCl",
        post_process_kwargs=TDEP_KWARGS,
    )
    assert nvt.output_files["tdep_inputs"] == tmp_path / "NaCl-tdep-inputs"


@pytest.mark.parametrize(
    "unit_cell, supercell, match",
    [
        (NACL, NACL * (2, 1, 1), "has 16 atoms, but the MD structure has 8"),
        (NACL, NACL[NACL.numbers.argsort()], "same chemical symbols"),
        (NACL, NACL_SWAPPED, "same positions"),
        (Atoms("NaCl", positions=[[0, 0, 0], [2.82, 0, 0]]), NACL, "three cell"),
        (bulk("NaCl", "rocksalt", a=5.5, cubic=True), NACL, "volume ratio"),
        (NACL[:-1], NACL, r"must equal their volume ratio \(1\)"),
    ],
)
def test_invalid_cells(tmp_path, unit_cell, supercell, match):
    """Test unit cells and supercells that do not match the MD structure."""
    write(tmp_path / "unit_cell.extxyz", unit_cell)
    write(tmp_path / "supercell.extxyz", supercell)
    tdep_kwargs = {
        "tdep_unit_cell_file": tmp_path / "unit_cell.extxyz",
        "tdep_supercell_file": tmp_path / "supercell.extxyz",
    }
    with pytest.raises(ValueError, match=match):
        NVT(
            **MD_KWARGS,
            file_prefix=tmp_path / "NaCl",
            post_process_kwargs=TDEP_KWARGS | tdep_kwargs,
        )


def test_wrapped_supercell(tmp_path):
    """Test a supercell with an atom outside the cell is accepted."""
    supercell = NACL.copy()
    supercell.positions[0] += supercell.cell[0]
    write(tmp_path / "supercell.extxyz", supercell)

    nvt = NVT(
        **MD_KWARGS,
        file_prefix=tmp_path / "NaCl",
        post_process_kwargs=TDEP_KWARGS
        | {"tdep_supercell_file": tmp_path / "supercell.extxyz"},
    )
    assert nvt.output_files["tdep_inputs"] == tmp_path / "NaCl-tdep-inputs"


def test_primitive_unit_cell(tmp_path):
    """Test a primitive unit cell, whose orientation a CIF file does not keep."""
    write(tmp_path / "primitive.extxyz", PRIMITIVE)
    write(tmp_path / "primitive.cif", PRIMITIVE)

    NVT(
        **MD_KWARGS,
        file_prefix=tmp_path / "NaCl",
        post_process_kwargs=TDEP_KWARGS
        | {"tdep_unit_cell_file": tmp_path / "primitive.extxyz"},
    )
    with pytest.raises(ValueError, match="not an integer matrix"):
        NVT(
            **MD_KWARGS,
            file_prefix=tmp_path / "NaCl",
            post_process_kwargs=TDEP_KWARGS
            | {"tdep_unit_cell_file": tmp_path / "primitive.cif"},
        )


def test_output_dir_is_file(tmp_path):
    """Test an existing file at the TDEP output path is rejected."""
    output_file = tmp_path / "tdep-inputs"
    output_file.touch()
    with pytest.raises(FileExistsError, match="is an existing file"):
        NVT(
            **MD_KWARGS,
            file_prefix=tmp_path / "NaCl",
            post_process_kwargs=TDEP_KWARGS | {"tdep_output_dir": output_file},
        )


def test_md_cli(tmp_path):
    """Test TDEP input files written from the md command line."""
    file_prefix = tmp_path / "NaCl"
    output_dir = tmp_path / "NaCl-tdep-inputs"
    summary_path = tmp_path / "NaCl-md-summary.yml"
    post_process_kwargs = (
        "{'tdep_compute': True, "
        f"'tdep_unit_cell_file': '{UNIT_CELL.as_posix()}', "
        f"'tdep_supercell_file': '{(DATA_PATH / 'NaCl.cif').as_posix()}'}}"
    )

    result = runner.invoke(
        app,
        [
            "md",
            "--ensemble",
            "nvt",
            "--struct",
            DATA_PATH / "NaCl.cif",
            "--arch",
            "mace_mp",
            "--model",
            MODEL_PATH,
            "--file-prefix",
            file_prefix,
            "--steps",
            2,
            "--traj-every",
            1,
            "--stats-every",
            1,
            "--post-process-kwargs",
            post_process_kwargs,
            "--no-tracker",
        ],
    )
    assert result.exit_code == 0
    assert sorted(path.name for path in output_dir.iterdir()) == TDEP_FILES

    with open(summary_path, encoding="utf8") as file:
        summary = yaml.safe_load(file)
    check_output_files(summary=summary, output_files={"tdep_inputs": output_dir})


def run_tdep(directory: Path) -> np.ndarray:
    """
    Run TDEP on the input files in a directory and read the phonon dispersion.

    Parameters
    ----------
    directory
        Directory containing the TDEP input files.

    Returns
    -------
    np.ndarray
        Frequencies in THz, with one row per q-point and one column per branch.
    """
    # TDEP exits with 0 on many errors, so a missing output file fails the test
    subprocess.run(
        ["extract_forceconstants", "-rc2", "5"],
        cwd=directory,
        check=True,
        capture_output=True,
    )
    shutil.copy(directory / "outfile.forceconstant", directory / "infile.forceconstant")
    subprocess.run(
        ["phonon_dispersion_relations"], cwd=directory, check=True, capture_output=True
    )
    return np.loadtxt(directory / "outfile.dispersion_relations")[:, 1:]


@requires_tdep
def test_tdep_phonons(tmp_path):
    """Test phonons from TDEP run on inputs written from committed NVT output."""
    build_tdep_inputs_from_nvt(
        unit_cell_file=UNIT_CELL,
        supercell=SUPERCELL,
        traj_file=TRAJ,
        stats_file=STATS,
        temperature=300.0,
        output_dir=tmp_path,
        timestep=100.0,
    )
    frequencies = run_tdep(tmp_path)

    # Six branches for two atoms, and no imaginary modes (written as negative)
    assert frequencies.shape[1] == 6
    assert frequencies.min() > -1e-3

    # Reference values from TDEP 6ccdf13, the commit built in CI
    gamma = frequencies[0]
    assert gamma[:3] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert gamma[3:] == pytest.approx([5.7205173069] * 3, rel=1e-4)
    assert frequencies.max() == pytest.approx(6.2202119257, rel=1e-4)


@requires_tdep
def test_tdep_nvt(tmp_path):
    """Test TDEP runs on the inputs written by a short NVT simulation."""
    output_dir = tmp_path / "tdep-inputs"
    NVT(
        **(MD_KWARGS | {"seed": 2024, "traj_every": 10, "stats_every": 10}),
        file_prefix=tmp_path / "NaCl",
        post_process_kwargs=TDEP_KWARGS
        | {"tdep_unit_cell_file": UNIT_CELL, "tdep_output_dir": output_dir},
    ).run()
    frequencies = run_tdep(output_dir)

    # Fresh MD output varies between platforms, so only check physical bounds
    assert frequencies.shape[1] == 6
    assert frequencies.min() > -0.1
    assert 4.0 < frequencies.max() < 8.0
