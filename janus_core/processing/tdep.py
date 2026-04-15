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

    trajectory = read(traj_file, index=":")
    n_atoms = len(trajectory[0])
    n_steps = len(trajectory)

    with open(meta_path, "w", encoding="utf-8") as file:
        file.write(f"{n_atoms}    # N atoms\n")
        file.write(f"{n_steps}    # N timesteps\n")
        file.write(f"{timestep}    # timestep in fs\n")
        file.write(f"{temperature}    # temperature in K\n")

    unit_cell = read(unit_cell_file)
    if len(supercell_atoms) != n_atoms:
        raise ValueError(
            "The number of atoms in supercell must match the number of atoms "
            "in traj_file."
        )

    write(ucposcar_path, unit_cell, format="vasp")
    write(ssposcar_path, supercell_atoms, format="vasp")

    with open(positions_path, "w", encoding="utf-8") as file:
        for atoms in trajectory:
            for position in atoms.get_scaled_positions():
                file.write(
                    f"{position[0]:.12f} {position[1]:.12f} {position[2]:.12f}\n"
                )

    with open(forces_path, "w", encoding="utf-8") as file:
        for atoms in trajectory:
            for force in atoms.get_forces():
                file.write(f"{force[0]:25.12f}{force[1]:25.12f}{force[2]:25.12f}\n")

    stats_data = np.atleast_2d(np.loadtxt(stats_file, comments="#"))

    if len(stats_data) != n_steps:
        raise ValueError(
            "The number of rows in stats_file must match the number of frames "
            "in traj_file."
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
        for i in range(len(time_fs)):
            file.write(
                f"{i:d} "
                f"{time_fs[i]:.12f} "
                f"{e_tot[i]:.12f} "
                f"{e_pot[i]:.12f} "
                f"{e_kin[i]:.12f} "
                f"{temperature_data[i]:.12f} "
                f"{pressure[i]:.12f} "
                f"{stress_xx[i]:.12f} "
                f"{stress_yy[i]:.12f} "
                f"{stress_zz[i]:.12f} "
                f"{stress_xz[i]:.12f} "
                f"{stress_yz[i]:.12f} "
                f"{stress_xy[i]:.12f}\n"
            )

    return {
        "infile.ucposcar": ucposcar_path,
        "infile.ssposcar": ssposcar_path,
        "infile.positions": positions_path,
        "infile.forces": forces_path,
        "infile.stat": stat_path,
        "infile.meta": meta_path,
    }
