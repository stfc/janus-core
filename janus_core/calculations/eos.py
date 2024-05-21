"""Equation of State."""

from logging import Logger
from typing import Any, Optional

from ase import Atoms
from ase.eos import EquationOfState
from ase.units import kJ
from numpy import float64, linspace
from numpy.typing import NDArray

from janus_core.calculations.geom_opt import optimize
from janus_core.helpers.janus_types import EoSNames, EoSResults, PathLike
from janus_core.helpers.log import config_logger
from janus_core.helpers.utils import none_to_dict


def _calc_volumes_energies(
    struct: Atoms,
    min_volume: float = 0.95,
    max_volume: float = 1.05,
    n_lattice: int = 7,
    minimize_all: bool = False,
    minimize_kwargs: Optional[dict[str, Any]] = None,
    logger: Optional[Logger] = None,
) -> tuple[NDArray[float64], list[float], list[float]]:
    """
    Calculate volumes and energies for all lattice constants.

    Parameters
    ----------
    struct : Atoms
        Structure.
    min_volume : float
        Minimum volume constant scale factor. Default is 0.95.
    max_volume : float
        Maximum volume constant scale factor. Default is 1.05.
    n_lattice : int
        Number of lattice constants to use. Default is 7.
    minimize_all : bool
        Whether to optimize geometry for all generated structures. Default is False.
    minimize_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to optimize. Default is None.
        chemical formula of the structure.
    logger : Optional[Logger]
        Logger if log file has been specified.

    Returns
    -------
    tuple[NDArray[float64], list[float], list[float]]
        Tuple of lattice scalars and lists of the corresponding volumes and energies.
    """
    if logger:
        logger.info("Starting calculations for configurations")

    cell = struct.get_cell()

    lattice_scalars = linspace(min_volume, max_volume, n_lattice) ** (1 / 3)
    volumes = []
    energies = []
    for lattice_scalar in lattice_scalars:
        c_struct = struct.copy()
        c_struct.calc = struct.calc
        c_struct.set_cell(cell * lattice_scalar, scale_atoms=True)

        # Minimize new structure
        if minimize_all:
            if logger:
                logger.info("Minimising lattice scalar = %s", lattice_scalar)
            optimize(c_struct, **minimize_kwargs)

        volumes.append(c_struct.get_volume())
        energies.append(c_struct.get_potential_energy())

    if logger:
        logger.info("Calculations for configurations complete")

    return lattice_scalars, volumes, energies


def calc_eos(
    # pylint: disable=too-many-locals,too-many-arguments,too-many-branches
    struct: Atoms,
    struct_name: Optional[str] = None,
    min_volume: float = 0.95,
    max_volume: float = 1.05,
    n_lattice: int = 7,
    eos_type: EoSNames = "birchmurnaghan",
    minimize: bool = True,
    minimize_all: bool = False,
    minimize_kwargs: Optional[dict[str, Any]] = None,
    write_results: bool = True,
    file_prefix: Optional[PathLike] = None,
    log_kwargs: Optional[dict[str, Any]] = None,
) -> EoSResults:
    """
    Calculate equation of state.

    Parameters
    ----------
    struct : Atoms
        Structure.
    struct_name : Optional[str]
        Name of structure. Default is None.
    min_volume : float
        Minimum volume scale factor. Default is 0.95.
    max_volume : float
        Maximum volume scale factor. Default is 1.05.
    n_lattice : int
        Number of lattice constants to use. Default is 7.
    eos_type : EoSNames
        Type of fit for equation of state. Default is "birchmurnaghan".
    minimize : bool
        Whether to minimize initial structure before calculations. Default is True.
    minimize_all : bool
        Whether to optimize geometry for all generated structures. Default is False.
    minimize_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to optimize. Default is None.
    write_results : bool
        True to write out results of equation of state calculations. Default is True.
    file_prefix : Optional[PathLike]
        Prefix for output filenames. Default is inferred from structure name, or
        chemical formula of the structure.
    log_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to `config_logger`. Default is {}.

    Returns
    -------
    EoSResults
        Dictionary containing equation of state ASE object, and fitted minimum
        bulk_modulus, volume, and energy.
    """
    [minimize_kwargs, log_kwargs] = none_to_dict([minimize_kwargs, log_kwargs])
    log_kwargs.setdefault("name", __name__)
    logger = config_logger(**log_kwargs)

    struct_name = struct_name if struct_name else struct.get_chemical_formula()
    file_prefix = file_prefix if file_prefix else struct_name

    if not struct.calc:
        raise ValueError("Please attach a calculator to `struct`.")

    # Ensure lattice constants span correct range
    if n_lattice <= 1:
        raise ValueError("`n_lattice` must be greater than 1.")
    if not 0 < min_volume < 1:
        raise ValueError("`min_volume` must be between 0 and 1.")
    if max_volume <= 1:
        raise ValueError("`max_volume` must be greater than 1.")

    if minimize:
        if logger:
            logger.info("Minimising initial structure")
            minimize_kwargs["log_kwargs"] = {
                "filename": log_kwargs["filename"],
                "name": logger.name,
                "filemode": "a",
            }
        optimize(struct, **minimize_kwargs)

    # Set constant volume for geometry optimization of generated structures
    if "filter_kwargs" in minimize_kwargs:
        minimize_kwargs["filter_kwargs"]["constant_volume"] = True
    else:
        minimize_kwargs["filter_kwargs"] = {"constant_volume": True}

    lattice_scalars, volumes, energies = _calc_volumes_energies(
        struct,
        min_volume,
        max_volume,
        n_lattice,
        minimize_all,
        minimize_kwargs,
        logger,
    )

    if write_results:
        with open(f"{file_prefix}-eos-raw.dat", "w", encoding="utf8") as out:
            print("#Lattice Scalar | Energy [eV] | Volume [Å^3] ", file=out)
            for eos_data in zip(lattice_scalars, energies, volumes):
                print(*eos_data, file=out)

    eos = EquationOfState(volumes, energies, eos_type)

    if logger:
        logger.info("Starting of fitting equation of state")

    v_0, e_0, bulk_modulus = eos.fit()
    # transform bulk modulus unit in GPa
    bulk_modulus *= 1.0e24 / kJ

    if logger:
        logger.info("Equation of state fitting complete")

    if write_results:
        with open(f"{file_prefix}-eos-fit.dat", "w", encoding="utf8") as out:
            print("#Bulk modulus [GPa] | Energy [eV] | Volume [Å^3] ", file=out)
            print(bulk_modulus, e_0, v_0, file=out)

    results = {
        "eos": eos,
        "bulk_modulus": bulk_modulus,
        "e_0": e_0,
        "v_0": v_0,
    }

    return results
