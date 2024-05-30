"""Geometry optimization."""

from logging import Logger
from pathlib import Path
from typing import Any, Callable, Optional
import warnings

from ase import Atoms
from ase.io import read, write
from ase.optimize import LBFGS
from ase.optimize.optimize import Optimizer

try:
    from ase.filters import FrechetCellFilter as DefaultFilter
except ImportError:
    from ase.constraints import ExpCellFilter as DefaultFilter

from numpy import linalg

from janus_core.helpers.janus_types import ASEOptArgs, ASEWriteArgs
from janus_core.helpers.log import config_logger
from janus_core.helpers.utils import none_to_dict, spacegroup


def set_optimizer(
    struct: Atoms,
    filter_func: Optional[Callable] = DefaultFilter,
    filter_kwargs: Optional[dict[str, Any]] = None,
    optimizer: Callable = LBFGS,
    opt_kwargs: Optional[ASEOptArgs] = None,
    logger: Optional[Logger] = None,
) -> tuple[Optimizer, Optional[Atoms]]:
    """
    Set optimizer for geometry optimization.

    Parameters
    ----------
    struct : Atoms
        Atoms object to optimize geometry for.
    filter_func : Optional[callable]
        Apply constraints to atoms through ASE filter function.
        Default is `FrechetCellFilter` if available otherwise `ExpCellFilter`.
    filter_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to filter_func. Default is {}.
    optimizer : callable
        ASE optimization function. Default is `LBFGS`.
    opt_kwargs : Optional[ASEOptArgs]
        Keyword arguments to pass to optimizer. Default is {}.
    logger : Optional[Logger]
        Logger instance. Default is None.

    Returns
    -------
    tuple[Optimizer, Optional[Atoms]]
        Optimizer and options the filtered atoms structure.
    """
    [filter_kwargs, opt_kwargs] = none_to_dict([filter_kwargs, opt_kwargs])
    filtered_struct = None
    if filter_func is not None:
        filtered_struct = filter_func(struct, **filter_kwargs)
        dyn = optimizer(filtered_struct, **opt_kwargs)
        if logger:
            logger.info("Using filter %s", filter_func.__name__)
            logger.info("Using optimizer %s", optimizer.__name__)
            if "hydrostatic_strain" in filter_kwargs:
                logger.info(
                    "hydrostatic_strain: %s", filter_kwargs["hydrostatic_strain"]
                )
            if "constant_volume" in filter_kwargs:
                logger.info("constant_volume: %s", filter_kwargs["constant_volume"])
            if "scalar_pressure" in filter_kwargs:
                logger.info("scalar_pressure: %s", filter_kwargs["scalar_pressure"])
    else:
        dyn = optimizer(struct, **opt_kwargs)

    return (dyn, filtered_struct)


def optimize(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
    struct: Atoms,
    fmax: float = 0.1,
    steps: int = 1000,
    symmetry_tolerance: float = 0.001,
    angle_tolerance: float = -1.0,
    filter_func: Optional[Callable] = DefaultFilter,
    filter_kwargs: Optional[dict[str, Any]] = None,
    optimizer: Callable = LBFGS,
    opt_kwargs: Optional[ASEOptArgs] = None,
    write_results: bool = False,
    write_kwargs: Optional[ASEWriteArgs] = None,
    traj_kwargs: Optional[ASEWriteArgs] = None,
    log_kwargs: Optional[dict[str, Any]] = None,
) -> Atoms:
    """
    Optimize geometry of input structure.

    Parameters
    ----------
    struct : Atoms
        Atoms object to optimize geometry for.
    fmax : float
        Set force convergence criteria for optimizer in units eV/Å.
        Default is 0.1.
    steps : int
        Set maximum number of optimization steps to run. Default is 1000.
    symmetry_tolerance : float
        Atom displacement tolerance for spglib symmetry determination, in Å.
        Default is 0.001.
    angle_tolerance : float
        Angle precision for spglib symmetry determination, in degrees. Default is -1.0,
        which means an internally optimized routine is used to judge symmetry.
    filter_func : Optional[callable]
        Apply constraints to atoms through ASE filter function.
        Default is `FrechetCellFilter` if available otherwise `ExpCellFilter`.
    filter_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to filter_func. Default is {}.
    optimizer : callable
        ASE optimization function. Default is `LBFGS`.
    opt_kwargs : Optional[ASEOptArgs]
        Keyword arguments to pass to optimizer. Default is {}.
    write_results : bool
        True to write out optimized structure. Default is False.
    write_kwargs : Optional[ASEWriteArgs],
        Keyword arguments to pass to ase.io.write to save optimized structure.
        Default is {}.
    traj_kwargs : Optional[ASEWriteArgs]
        Keyword arguments to pass to ase.io.write to save optimization trajectory.
        Must include "filename" keyword. Default is {}.
    log_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to `config_logger`. Default is {}.

    Returns
    -------
    struct: Atoms
        Structure with geometry optimized.
    """
    [opt_kwargs, write_kwargs, traj_kwargs, log_kwargs] = none_to_dict(
        [opt_kwargs, write_kwargs, traj_kwargs, log_kwargs]
    )

    write_kwargs.setdefault(
        "filename",
        Path(f"./{struct.get_chemical_formula()}-opt.xyz").absolute(),
    )

    if traj_kwargs and "filename" not in traj_kwargs:
        raise ValueError("'filename' must be included in `traj_kwargs`")

    if traj_kwargs and "trajectory" not in opt_kwargs:
        raise ValueError(
            "'trajectory' must be a key in `opt_kwargs` to save the trajectory."
        )

    if log_kwargs and "filename" not in log_kwargs:
        raise ValueError("'filename' must be included in `log_kwargs`")

    log_kwargs.setdefault("name", __name__)
    logger = config_logger(**log_kwargs)

    s_grp = spacegroup(struct, symmetry_tolerance, angle_tolerance)
    message = f"Before optimisation spacegroup {s_grp}"
    struct.info["initial_spacegroup"] = s_grp

    if logger:
        logger.info(message)

    dyn, filtered_struct = set_optimizer(
        struct, filter_func, filter_kwargs, optimizer, opt_kwargs, logger
    )

    if logger:
        logger.info("Starting geometry optimization")

    converged = dyn.run(fmax=fmax, steps=steps)

    s_grp = spacegroup(struct, symmetry_tolerance, angle_tolerance)
    message = f"After optimisation spacegroup {s_grp}"
    struct.info["final_spacegroup"] = s_grp

    # Calculate current maximum force
    if filter_func is not None:
        max_force = linalg.norm(filtered_struct.get_forces(), axis=1).max()
    else:
        max_force = linalg.norm(struct.get_forces(), axis=1).max()

    if logger:
        logger.info(message)
        logger.info("Max force: %.6f", max_force)

    if not converged:
        warnings.warn(
            f"Optimization has not converged after {steps} steps. "
            f"Current max force {max_force} > target force {fmax}"
        )

    # Write out optimized structure
    if write_results:
        write(images=struct, **write_kwargs)

    # Reformat trajectory file from binary
    if traj_kwargs:
        traj = read(opt_kwargs["trajectory"], index=":")
        write(images=traj, **traj_kwargs)

    if logger:
        logger.info("Geometry optimization complete")

    return struct
