"""Equation of State."""

from logging import Logger
from typing import Any, Optional

from ase import Atoms
from ase.eos import EquationOfState
from ase.units import kJ
from codecarbon import OfflineEmissionsTracker
from numpy import float64, linspace
from numpy.typing import NDArray

from janus_core.calculations.geom_opt import optimize
from janus_core.helpers.janus_types import EoSNames, EoSResults, OutputKwargs, PathLike
from janus_core.helpers.log import config_logger, config_tracker
from janus_core.helpers.utils import none_to_dict, output_structs


def _calc_volumes_energies(  # pylint: disable=too-many-locals
    struct: Atoms,
    *,
    min_volume: float = 0.95,
    max_volume: float = 1.05,
    n_volumes: int = 7,
    minimize_all: bool = False,
    minimize_kwargs: Optional[dict[str, Any]] = None,
    write_structures: bool = False,
    write_kwargs: Optional[OutputKwargs] = None,
    logger: Optional[Logger] = None,
    tracker: Optional[OfflineEmissionsTracker] = None,
) -> tuple[NDArray[float64], list[float], list[float]]:
    """
    Calculate volumes and energies for all lattice constants.

    Parameters
    ----------
    struct : Atoms
        Structure.
    min_volume : float
        Minimum volume scale factor. Default is 0.95.
    max_volume : float
        Maximum volume scale factor. Default is 1.05.
    n_volumes : int
        Number of volumes to use. Default is 7.
    minimize_all : bool
        Whether to optimize geometry for all generated structures. Default is False.
    minimize_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to optimize. Default is None.
        chemical formula of the structure.
    write_structures : bool
        True to write out all genereated structures. Default is False.
    write_kwargs : Optional[OutputKwargs],
        Keyword arguments to pass to ase.io.write to save generated structures.
        Default is {}.
    logger : Optional[Logger]
        Logger if log file has been specified.
    tracker : Optional[OfflineEmissionsTracker]
        Tracker if logging is enabled.

    Returns
    -------
    tuple[NDArray[float64], list[float], list[float]]
        Tuple of lattice scalars and lists of the corresponding volumes and energies.
    """
    [minimize_kwargs, write_kwargs] = none_to_dict([minimize_kwargs, write_kwargs])

    if logger:
        logger.info("Starting calculations for configurations")
        tracker.start_task("Calculate configurations")

    cell = struct.get_cell()

    lattice_scalars = linspace(min_volume, max_volume, n_volumes) ** (1 / 3)
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

        # Always append first original structure
        write_kwargs["append"] = True
        # Write structures, but no need to set info c_struct is not used elsewhere
        output_structs(
            images=c_struct,
            write_results=write_structures,
            set_info=False,
            write_kwargs=write_kwargs,
        )

    if logger:
        tracker.stop_task()
        logger.info("Calculations for configurations complete")

    return lattice_scalars, volumes, energies


def calc_eos(
    # pylint: disable=too-many-locals,too-many-arguments,too-many-branches
    struct: Atoms,
    struct_name: Optional[str] = None,
    min_volume: float = 0.95,
    max_volume: float = 1.05,
    n_volumes: int = 7,
    eos_type: EoSNames = "birchmurnaghan",
    minimize: bool = True,
    minimize_all: bool = False,
    minimize_kwargs: Optional[dict[str, Any]] = None,
    write_results: bool = True,
    write_structures: bool = False,
    write_kwargs: Optional[OutputKwargs] = None,
    file_prefix: Optional[PathLike] = None,
    log_kwargs: Optional[dict[str, Any]] = None,
    tracker_kwargs: Optional[dict[str, Any]] = None,
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
    n_volumes : int
        Number of volumes to use. Default is 7.
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
    write_structures : bool
        True to write out all genereated structures. Default is False.
    write_kwargs : Optional[OutputKwargs],
        Keyword arguments to pass to ase.io.write to save generated structures.
        Default is {}.
    file_prefix : Optional[PathLike]
        Prefix for output filenames. Default is inferred from structure name, or
        chemical formula of the structure.
    log_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to `config_logger`. Default is {}.
    tracker_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to `config_tracker`. Default is {}.

    Returns
    -------
    EoSResults
        Dictionary containing equation of state ASE object, and fitted minimum
        bulk_modulus, volume, and energy.
    """
    [minimize_kwargs, write_kwargs, log_kwargs, tracker_kwargs] = none_to_dict(
        [minimize_kwargs, write_kwargs, log_kwargs, tracker_kwargs]
    )

    log_kwargs.setdefault("name", __name__)
    logger = config_logger(**log_kwargs)
    tracker = config_tracker(logger, **tracker_kwargs)

    struct_name = struct_name if struct_name else struct.get_chemical_formula()
    file_prefix = file_prefix if file_prefix else struct_name

    write_kwargs.setdefault("filename", f"{file_prefix}-generated.extxyz")

    if (
        (minimize or minimize_all)
        and "write_results" in minimize_kwargs
        and minimize_kwargs["write_results"]
    ):
        raise ValueError(
            "Optimized structures can be saved by setting `write_structures`"
        )

    if not struct.calc:
        raise ValueError("Please attach a calculator to `struct`.")

    # Ensure lattice constants span correct range
    if n_volumes <= 1:
        raise ValueError("`n_volumes` must be greater than 1.")
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

        # Optionally write structure to file
        output_structs(
            images=struct, write_results=write_structures, write_kwargs=write_kwargs
        )

    # Set constant volume for geometry optimization of generated structures
    if "filter_kwargs" in minimize_kwargs:
        minimize_kwargs["filter_kwargs"]["constant_volume"] = True
    else:
        minimize_kwargs["filter_kwargs"] = {"constant_volume": True}

    lattice_scalars, volumes, energies = _calc_volumes_energies(
        struct=struct,
        min_volume=min_volume,
        max_volume=max_volume,
        n_volumes=n_volumes,
        minimize_all=minimize_all,
        minimize_kwargs=minimize_kwargs,
        write_structures=write_structures,
        write_kwargs=write_kwargs,
        logger=logger,
        tracker=tracker,
    )

    if write_results:
        with open(f"{file_prefix}-eos-raw.dat", "w", encoding="utf8") as out:
            print("#Lattice Scalar | Energy [eV] | Volume [Å^3] ", file=out)
            for eos_data in zip(lattice_scalars, energies, volumes):
                print(*eos_data, file=out)

    eos = EquationOfState(volumes, energies, eos_type)

    if logger:
        logger.info("Starting of fitting equation of state")
        tracker.start_task("Fit EoS")

    v_0, e_0, bulk_modulus = eos.fit()
    # transform bulk modulus unit in GPa
    bulk_modulus *= 1.0e24 / kJ

    if logger:
        tracker.stop_task()
        tracker.stop()
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
