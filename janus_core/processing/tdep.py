"""Utilities for generating TDEP input files."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import read, write
import numpy as np
from numpy import float64
from numpy.typing import NDArray

from janus_core.helpers.janus_types import PathLike


def _frame_forces(atoms: Atoms) -> NDArray[float64]:
    """
    Get the forces stored in a trajectory frame.

    Parameters
    ----------
    atoms
        Frame read from the MD trajectory file.

    Returns
    -------
    NDArray[float64]
        Forces on each atom. Taken from the frame's calculator if present,
        otherwise from the ``<arch>_forces`` array written by janus-core.
    """
    if atoms.calc is not None and "forces" in atoms.calc.results:
        return atoms.get_forces()
    arch = atoms.info.get("arch")
    if arch is not None and f"{arch}_forces" in atoms.arrays:
        return atoms.arrays[f"{arch}_forces"]
    raise ValueError("Frames in traj_file must contain 'forces' or '<arch>_forces'.")


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
        Path to the MD statistics file. Each frame in ``traj_file`` is paired with
        the row written at the same MD step.
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
    forces = [_frame_forces(atoms) for atoms in trajectory]

    if len(supercell_atoms) != n_atoms:
        raise ValueError(
            "The number of atoms in supercell must match the number of atoms "
            "in traj_file."
        )

    # Pair each frame with the stats row written at the same MD step
    if any("step" not in atoms.info for atoms in trajectory):
        raise ValueError("Frames in traj_file must contain 'step' in their info.")
    frame_steps = [int(atoms.info["step"]) for atoms in trajectory]
    if len(set(frame_steps)) != n_steps:
        raise ValueError("Frames in traj_file must have unique steps.")

    stats_rows = {int(step): row for row, step in enumerate(stats_data[:, 0])}
    if len(stats_rows) != len(stats_data):
        raise ValueError("Rows in stats_file must have unique steps.")

    missing = [step for step in frame_steps if step not in stats_rows]
    if missing:
        raise ValueError(
            f"stats_file has no row for {len(missing)} trajectory step(s), "
            f"e.g. step {missing[0]}. traj_every must be a multiple of stats_every."
        )
    stats_data = stats_data[[stats_rows[step] for step in frame_steps]]

    # Only create the output directory once all checks have passed
    output_dir.mkdir(parents=True, exist_ok=True)

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
            for frame_forces in forces
            for force in frame_forces
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
                    # Off-diagonal order follows TDEP's reader (xy, yz, xz),
                    # not its docs/files.md (xz, yz, xy)
                    stress_xy,
                    stress_yz,
                    stress_xz,
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
