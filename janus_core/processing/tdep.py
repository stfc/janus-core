"""Utilities for generating TDEP input files."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import read, write
import numpy as np
from numpy import float64
from numpy.typing import NDArray

from janus_core.helpers.janus_types import PathLike
from janus_core.helpers.stats import Stats


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
        Forces on each atom. Taken from the ``<arch>_forces`` array written by
        janus-core if present, otherwise from the results stored in the frame's
        calculator, so they are never recalculated.
    """
    arch = atoms.info.get("arch")
    if arch is not None and f"{arch}_forces" in atoms.arrays:
        return atoms.arrays[f"{arch}_forces"]
    if atoms.calc is not None and "forces" in atoms.calc.results:
        return atoms.calc.results["forces"]
    raise ValueError("Frames in traj_file must contain 'forces' or '<arch>_forces'.")


def _validate_inputs(
    trajectory: list[Atoms],
    supercell: Atoms,
    stats_steps: list[int],
    start_step: int,
) -> None:
    """
    Check that MD outputs can be converted to TDEP input files.

    Parameters
    ----------
    trajectory
        Frames read from the MD trajectory file.
    supercell
        Supercell written to ``infile.ssposcar``.
    stats_steps
        MD step of each row in the MD statistics file.
    start_step
        First MD step whose frame is used.

    Raises
    ------
    ValueError
        If the frames to use cannot be paired with the supercell and with the rows
        of the MD statistics file.
    """
    if any("step" not in atoms.info for atoms in trajectory):
        raise ValueError("Frames in traj_file must contain 'step' in their info.")
    frame_steps = [
        atoms.info["step"] for atoms in trajectory if atoms.info["step"] >= start_step
    ]
    if not frame_steps:
        raise ValueError(f"traj_file has no frames from step {start_step}.")
    if len(set(frame_steps)) != len(frame_steps):
        raise ValueError("Frames in traj_file must have unique steps.")

    if len(supercell) != len(trajectory[0]):
        raise ValueError(
            "The number of atoms in supercell must match the number of atoms "
            "in traj_file."
        )

    if len(set(stats_steps)) != len(stats_steps):
        raise ValueError("Rows in stats_file must have unique steps.")
    if missing := sorted(set(frame_steps).difference(stats_steps)):
        raise ValueError(
            f"stats_file has no row for {len(missing)} trajectory step(s), "
            f"e.g. step {missing[0]}. traj_every must be a multiple of stats_every."
        )


def _snap_to_unit_cell(unit_cell: Atoms, supercell: Atoms) -> tuple[Atoms, Atoms]:
    """
    Place the supercell atoms exactly on images of the unit-cell atoms.

    TDEP requires the two cells to agree to machine precision, but positions read from
    extxyz files, for example, are rounded to 8 decimals.

    Parameters
    ----------
    unit_cell
        Unit cell written to ``infile.ucposcar``.
    supercell
        Supercell of ``unit_cell`` written to ``infile.ssposcar``.

    Returns
    -------
    tuple[Atoms, Atoms]
        Copy of ``unit_cell``, with fractional coordinates within 1e-8 of a cell
        boundary moved onto it, and copy of ``supercell``, with each atom moved onto
        the nearest image of a unit-cell atom of the same element.
    """
    unit_cell = unit_cell.copy()
    supercell = supercell.copy()

    # Coordinates this close to a cell boundary are meant to be on it
    sites = unit_cell.get_scaled_positions(wrap=False)
    on_boundary = np.abs(sites - np.rint(sites)) < 1e-8
    sites[on_boundary] = np.rint(sites[on_boundary])
    unit_cell.set_scaled_positions(sites)

    # Offsets of each supercell atom from each unit-cell atom, in fractional
    # coordinates of the unit cell, using the integer supercell matrix as TDEP does
    matrix = np.rint(supercell.cell @ np.linalg.inv(unit_cell.cell))
    offsets = (supercell.get_scaled_positions(wrap=False) @ matrix)[:, None] - sites
    translations = np.rint(offsets)
    distances = np.linalg.norm((offsets - translations) @ unit_cell.cell, axis=2)
    distances[supercell.numbers[:, None] != unit_cell.numbers] = np.inf
    if distances.min(axis=1).max() > 1e-3:
        raise ValueError(
            "Each supercell atom must be within 1e-3 Å of an image of a unit-cell atom "
            "of the same element"
        )

    nearest = distances.argmin(axis=1)
    images = sites[nearest] + translations[np.arange(len(supercell)), nearest]
    supercell.set_scaled_positions(images @ np.linalg.inv(matrix))
    return unit_cell, supercell


def build_tdep_inputs_from_nvt(
    *,
    unit_cell_file: PathLike,
    supercell: Atoms | PathLike,
    traj_file: PathLike,
    stats_file: PathLike,
    temperature: float,
    output_dir: PathLike,
    timestep: float = 0.0,
    start_step: int = 0,
) -> dict[str, Path]:
    """
    Build TDEP input files from NVT molecular dynamics outputs.

    Parameters
    ----------
    unit_cell_file
        Path to the unit-cell structure file used to generate ``infile.ucposcar``.
        Fractional coordinates within 1e-8 of a cell boundary are moved onto it.
    supercell
        Supercell structure or path used to generate ``infile.ssposcar``. Each atom
        is moved onto the nearest image of a unit-cell atom of the same element,
        which must be within 1e-3 Å.
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
    start_step
        First MD step whose frame is used, so that earlier frames, such as those
        saved during a temperature ramp, are skipped. Default is ``0``.

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
    unit_cell = read(unit_cell_file)
    stats = Stats(stats_file)
    stats_steps = [int(step) for step in stats["Step"]]
    _validate_inputs(trajectory, supercell_atoms, stats_steps, start_step)
    unit_cell, supercell_atoms = _snap_to_unit_cell(unit_cell, supercell_atoms)

    # Pair each frame from start_step with the stats row written at the same MD step
    trajectory = [atoms for atoms in trajectory if atoms.info["step"] >= start_step]
    stats_rows = {step: row for row, step in enumerate(stats_steps)}
    rows = [stats_rows[atoms.info["step"]] for atoms in trajectory]
    forces = [_frame_forces(atoms) for atoms in trajectory]
    n_atoms = len(trajectory[0])
    n_steps = len(trajectory)

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

    time_fs = stats["Time"][rows]
    e_pot = stats["Epot/N"][rows] * n_atoms
    e_kin = stats["EKin/N"][rows] * n_atoms
    temperature_data = stats["T"][rows]
    e_tot = stats["ETot/N"][rows] * n_atoms
    pressure = stats["P"][rows]
    stress_xx = stats["Pxx"][rows]
    stress_yy = stats["Pyy"][rows]
    stress_zz = stats["Pzz"][rows]
    stress_yz = stats["Pyz"][rows]
    stress_xz = stats["Pxz"][rows]
    stress_xy = stats["Pxy"][rows]

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
