"""Module containing types used in Janus-Core."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from enum import Enum
import logging
from pathlib import Path, PurePath
from typing import (
    IO,
    Literal,
    Optional,
    Protocol,
    TypedDict,
    TypeVar,
    Union,
    runtime_checkable,
)

from ase import Atoms
from ase.eos import EquationOfState
import numpy as np
from numpy.typing import NDArray

# General

T = TypeVar("T")
MaybeList = Union[T, list[T]]
MaybeSequence = Union[T, Sequence[T]]
PathLike = Union[str, Path]
StartStopStep = tuple[Optional[int], Optional[int], int]
SliceLike = Union[slice, range, int, StartStopStep]

# ASE Arg types


class ASEReadArgs(TypedDict, total=False):
    """Main arguments for ase.io.read."""

    filename: str | PurePath | IO
    index: int | slice | str
    format: str | None
    parallel: bool
    do_not_split_by_at_sign: bool


class ASEWriteArgs(TypedDict, total=False):
    """Main arguments for ase.io.write."""

    filename: str | PurePath | IO
    images: MaybeSequence[Atoms]
    format: str | None
    parallel: bool
    append: bool


class ASEOptArgs(TypedDict, total=False):
    """Main arguments for ase optimisers."""

    restart: bool | None
    logfile: PathLike | None
    trajectory: str | None


class PostProcessKwargs(TypedDict, total=False):
    """Main arguments for MD post-processing."""

    # RDF
    rdf_compute: bool
    rdf_rmax: float
    rdf_nbins: int
    rdf_elements: MaybeSequence[str | int]
    rdf_by_elements: bool
    rdf_start: int
    rdf_stop: int | None
    rdf_step: int
    rdf_output_file: str | None
    # VAF
    vaf_compute: bool
    vaf_velocities: bool
    vaf_fft: bool
    vaf_atoms: Sequence[Sequence[int]]
    vaf_start: int
    vaf_stop: int | None
    vaf_step: int
    vaf_output_file: PathLike | None


@runtime_checkable
class Observable(Protocol):
    """Signature for correlation observable getter."""

    def __call__(self, atoms: Atoms, *args, **kwargs) -> float:
        """
        Call the getter.

        Parameters
        ----------
        atoms : Atoms
            Atoms object to extract values from.
        *args : tuple
            Additional positional arguments passed to getter.
        **kwargs : dict
            Additional kwargs passed getter.
        """


class CorrelationKwargs(TypedDict, total=True):
    """Arguments for on-the-fly correlations <ab>."""

    #: observable a in <ab>, with optional args and kwargs
    a: Observable | tuple[Observable, tuple, dict]
    #: observable b in <ab>, with optional args and kwargs
    b: Observable | tuple[Observable, tuple, dict]
    #: name used for correlation in output
    name: str
    #: blocks used in multi-tau algorithm
    blocks: int
    #: points per block
    points: int
    #: averaging between blocks
    averaging: int
    #: frequency to update the correlation (steps)
    update_frequency: int


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
Architectures = Literal[
    "mace", "mace_mp", "mace_off", "m3gnet", "chgnet", "alignn", "sevennet"
]
Devices = Literal["cpu", "cuda", "mps", "xpu"]
Ensembles = Literal["nph", "npt", "nve", "nvt", "nvt-nh"]
Properties = Literal["energy", "stress", "forces", "hessian"]
PhononCalcs = Literal["bands", "dos", "pdos", "thermal"]


class OutputKwargs(ASEWriteArgs, total=False):
    """Main keyword arguments for `output_structs`."""

    set_info: bool
    write_results: bool
    properties: Collection[Properties]
    invalidate_calc: bool


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
