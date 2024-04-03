"""Utility functions for tests."""

from pathlib import Path
from typing import Optional, Union

from ase import Atoms
from ase.io import read
import yaml

from janus_core.janus_types import MaybeSequence, PathLike


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


def check_log_contents(
    log_path: PathLike,
    contains: Optional[MaybeSequence[str]] = None,
    excludes: Optional[MaybeSequence[str]] = None,
) -> None:
    """
    Check messages are present or not within a yaml-formatted log file.

    Parameters
    ----------
    log_path : PathLike
        Path to log file to check messsages of.
    contains : MaybeSequence[str]
        Messages that must appear in the log file. Default is None.
    excludes : MaybeSequence[str]
        Messages that must not appear in the log file. Default is None.
    """
    # Convert single strings to iterable
    contains = [contains] if isinstance(contains, str) else contains
    excludes = [excludes] if isinstance(excludes, str) else excludes

    # Read log file
    with open(log_path, encoding="utf8") as log_file:
        logs = yaml.safe_load(log_file)
    messages = "".join(log["message"] for log in logs)

    if contains:
        for msg in contains:
            assert msg in messages
    if excludes:
        for msg in excludes:
            assert msg not in messages
