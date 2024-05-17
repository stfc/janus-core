"""Module containing types used in Janus-Core."""

from collections.abc import Sequence
from enum import Enum
import logging
from pathlib import Path, PurePath
from typing import IO, Literal, Optional, TypedDict, TypeVar, Union

from ase import Atoms
from ase.eos import EquationOfState
import numpy as np
from numpy.typing import NDArray

# General

T = TypeVar("T")
MaybeList = Union[T, list[T]]
MaybeSequence = Union[T, Sequence[T]]
PathLike = Union[str, Path]


# ASE Arg types


class ASEReadArgs(TypedDict, total=False):
    """Main arguments for ase.io.read."""

    filename: Union[str, PurePath, IO]
    index: Union[int, slice, str]
    format: Optional[str]
    parallel: bool
    do_not_split_by_at_sign: bool


class ASEWriteArgs(TypedDict, total=False):
    """Main arguments for ase.io.write."""

    filename: Union[str, PurePath, IO]
    images: MaybeSequence[Atoms]
    format: Optional[str]
    parallel: bool
    append: bool


class ASEOptArgs(TypedDict, total=False):
    """Main arugments for ase optimisers."""

    restart: Optional[bool]
    logfile: Optional[PathLike]
    trajectory: Optional[str]


# eos_names from ase.eos
EoSNames = Literal[
    "sj",
    "taylor",
    "murnaghan",
    "birch",
    "birchmurnaghan",
    "pouriertarantola",
    "vinet",
    "antonschmidt",
    "p3",
]


# Janus specific
Architectures = Literal["mace", "mace_mp", "mace_off", "m3gnet", "chgnet"]
Devices = Literal["cpu", "cuda", "mps"]
Ensembles = Literal["nph", "npt", "nve", "nvt", "nvt-nh"]


class LogLevel(Enum):  # numpydoc ignore=PR01
    """Supported options for logger levels."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR


class CalcResults(TypedDict, total=False):
    """Return type from calculations."""

    energy: MaybeList[float]
    forces: MaybeList[NDArray[np.float64]]
    stress: MaybeList[NDArray[np.float64]]


class EoSResults(TypedDict, total=False):
    """Return type from calculations."""

    eos: EquationOfState
    bulk_modulus: float
    v_0: float
    e_0: float
