"""Module for functions operating on structure symmetry."""

from __future__ import annotations

from ase import Atoms
from ase.spacegroup.symmetrize import refine_symmetry
from spglib import get_spacegroup
from spglib.error import SpglibError


def spacegroup(
    struct: Atoms, sym_tolerance: float = 0.001, angle_tolerance: float = -1.0
) -> str | None:
    """
    Determine the spacegroup for a structure.

    Parameters
    ----------
    struct
        Structure as an ase Atoms object.
    sym_tolerance
        Atom displacement tolerance for spglib symmetry determination, in Å.
        Default is 0.001.
    angle_tolerance
        Angle precision for spglib symmetry determination, in degrees. Default is -1.0,
        which means an internally optimized routine is used to judge symmetry.

    Returns
    -------
    str | None
        Spacegroup name, or None if symmetry cannot be determined.
    """
    try:
        return get_spacegroup(
            cell=(
                struct.get_cell(),
                struct.get_scaled_positions(),
                struct.get_atomic_numbers(),
            ),
            symprec=sym_tolerance,
            angle_tolerance=angle_tolerance,
        )
    except SpglibError:
        return None


def snap_symmetry(struct: Atoms, sym_tolerance: float = 0.001) -> None:
    """
    Symmetrize structure's cell vectors and atomic positions.

    Parameters
    ----------
    struct
        Structure as an ase Atoms object.
    sym_tolerance
        Atom displacement tolerance for spglib symmetry determination, in Å.
        Default is 0.001.
    """
    refine_symmetry(struct, symprec=sym_tolerance, verbose=False)
