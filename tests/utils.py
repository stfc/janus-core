"""Utility functions for tests"""

from pathlib import Path
from typing import Union

from ase import Atoms
from ase.io import read


def read_atoms(path: Path) -> Union[Atoms, None]:
    """
    Read Atoms structure file, and delete file regardless of success.

    Parameters
    ----------
    path : Path
        Path to file containing Atoms structure to be read.

    Returns
    -------
    Union[Atoms, None]
        Atoms structure read from file, or None is any Exception is thrown.
    """
    assert path.exists()
    try:
        atoms = read(path)
        assert isinstance(atoms, Atoms)
    finally:
        # Ensure file is still deleted if read fails
        path.unlink()

    return atoms if atoms else None
