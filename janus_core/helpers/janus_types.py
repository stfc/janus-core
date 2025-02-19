"""Module containing types used in Janus-Core."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from enum import Enum
import logging
from pathlib import Path, PurePath
from typing import IO, TYPE_CHECKING, Literal, TypedDict, TypeVar

from ase import Atoms
from ase.eos import EquationOfState
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from janus_core.processing.observables import Observable

# General

T = TypeVar("T")
MaybeList = T | list[T]
MaybeSequence = T | Sequence[T]
PathLike = str | Path
StartStopStep = tuple[int | None, int | None, int]
SliceLike = slice | range | int | StartStopStep

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
    vaf_output_files: Sequence[PathLike] | None


class CorrelationKwargs(TypedDict, total=True):
    """Arguments for on-the-fly correlations <ab>."""

    #: observable a in <ab>, with optional args and kwargs
    a: Observable
    #: observable b in <ab>, with optional args and kwargs
    b: Observable
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
    "mace",
    "mace_mp",
    "mace_off",
    "m3gnet",
    "chgnet",
    "alignn",
    "sevennet",
    "nequip",
    "dpa3",
    "orb",
]
Devices = Literal["cpu", "cuda", "mps", "xpu"]
Ensembles = Literal["nph", "npt", "nve", "nvt", "nvt-nh", "nvt-csvr", "npt-mtk"]
Properties = Literal["energy", "stress", "forces", "hessian"]
PhononCalcs = Literal["bands", "dos", "pdos", "thermal"]
Interpolators = Literal["ase", "pymatgen"]


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
