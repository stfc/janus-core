"""Utilities for generating TDEP input files."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import read, write
import numpy as np

from janus_core.helpers.janus_types import PathLike


def build_tdep_inputs_from_nvt(
    *,
    unit_cell_file: PathLike,
    supercell: Atoms | PathLike,
    traj_file: PathLike,
    stats_file: PathLike,
    temperature: float,
    output_dir: PathLike,
    timestep: float = 0.0,
) -> dict[str, Path]:
    """
    Build TDEP input files from NVT molecular dynamics outputs.

    Parameters
    ----------
    unit_cell_file
        Path to the unit-cell structure file used to generate ``infile.ucposcar``.
    supercell
        Supercell structure or path used to generate ``infile.ssposcar``.
    traj_file
        Path to the MD trajectory file.
    stats_file
        Path to the MD statistics file.
    temperature
        MD temperature in K, written to ``infile.meta``.
    output_dir
        Directory where the TDEP input files will be written.
    timestep
        Time interval between exported configurations in fs. Default is ``0.0``.

    Returns
    -------
    dict[str, Path]
        Mapping from TDEP standard filenames to their output paths.
    """
    unit_cell_file = Path(unit_cell_file)
    traj_file = Path(traj_file)
    stats_file = Path(stats_file)
    output_dir = Path(output_dir)

    if isinstance(supercell, Atoms):
        supercell_atoms = supercell
    else:
        supercell = Path(supercell)
        supercell_atoms = read(supercell)

    output_dir.mkdir(parents=True, exist_ok=True)

    ucposcar_path = output_dir / "infile.ucposcar"
    ssposcar_path = output_dir / "infile.ssposcar"
    positions_path = output_dir / "infile.positions"
    forces_path = output_dir / "infile.forces"
    stat_path = output_dir / "infile.stat"
    meta_path = output_dir / "infile.meta"
    out_fmt = "{:.12f}".format
    trajectory = read(traj_file, index=":")
    n_atoms = len(trajectory[0])
    n_steps = len(trajectory)
    unit_cell = read(unit_cell_file)
    stats_data = np.atleast_2d(np.loadtxt(stats_file, comments="#"))

    if len(supercell_atoms) != n_atoms:
        raise ValueError(
            "The number of atoms in supercell must match the number of atoms "
            "in traj_file."
        )

    if len(stats_data) != n_steps:
        raise ValueError(
            "The number of rows in stats_file must match the number of frames "
            "in traj_file."
        )

    meta_path.write_text(
        f"""\
{n_atoms}    # N atoms
{n_steps}    # N timesteps
{timestep}    # timestep in fs
{temperature}    # temperature in K
""",
        encoding="utf-8",
    )

    write(ucposcar_path, unit_cell, format="vasp")
    write(ssposcar_path, supercell_atoms, format="vasp")

    with open(positions_path, "w", encoding="utf-8") as file:
        file.writelines(
            " ".join(map(out_fmt, position)) + "\n"
            for atoms in trajectory
            for position in atoms.get_scaled_positions()
        )

    with open(forces_path, "w", encoding="utf-8") as file:
        file.writelines(
            f"{force[0]:25.12f} {force[1]:25.12f} {force[2]:25.12f}\n"
            for atoms in trajectory
            for force in atoms.get_forces()
        )

    time_fs = stats_data[:, 2]
    e_pot = stats_data[:, 3] * n_atoms
    e_kin = stats_data[:, 4] * n_atoms
    temperature_data = stats_data[:, 5]
    e_tot = stats_data[:, 6] * n_atoms
    pressure = stats_data[:, 9]
    stress_xx = stats_data[:, 10]
    stress_yy = stats_data[:, 11]
    stress_zz = stats_data[:, 12]
    stress_yz = stats_data[:, 13]
    stress_xz = stats_data[:, 14]
    stress_xy = stats_data[:, 15]

    with open(stat_path, "w", encoding="utf-8") as file:
        file.writelines(
            f"{i:d} "
            + " ".join(
                out_fmt(dat[i])
                for dat in (
                    time_fs,
                    e_tot,
                    e_pot,
                    e_kin,
                    temperature_data,
                    pressure,
                    stress_xx,
                    stress_yy,
                    stress_zz,
                    stress_xz,
                    stress_yz,
                    stress_xy,
                )
            )
            + "\n"
            for i in range(len(time_fs))
        )

    return {
        "infile.ucposcar": ucposcar_path,
        "infile.ssposcar": ssposcar_path,
        "infile.positions": positions_path,
        "infile.forces": forces_path,
        "infile.stat": stat_path,
        "infile.meta": meta_path,
    }
