"""Geometry optimization."""

from pathlib import Path
from typing import Any, Callable, Optional

from ase import Atoms
from ase.io import read, write
from ase.optimize import LBFGS

try:
    from ase.filters import FrechetCellFilter as DefaultFilter
except ImportError:
    from ase.constraints import ExpCellFilter as DefaultFilter

from janus_core.janus_types import ASEOptArgs, ASEOptRunArgs, ASEWriteArgs
from janus_core.log import config_logger


def optimize(  # pylint: disable=too-many-arguments
    atoms: Atoms,
    fmax: float = 0.1,
    dyn_kwargs: Optional[ASEOptRunArgs] = None,
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
    atoms : Atoms
        Atoms object to optimize geometry for.
    fmax : float
        Set force convergence criteria for optimizer in units eV/Å.
        Default is 0.1.
    dyn_kwargs : Optional[ASEOptRunArgs]
        Keyword arguments to pass to dyn.run. Default is {}.
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
    atoms: Atoms
        Structure with geometry optimized.
    """
    dyn_kwargs = dyn_kwargs if dyn_kwargs else {}
    filter_kwargs = filter_kwargs if filter_kwargs else {}
    opt_kwargs = opt_kwargs if opt_kwargs else {}
    write_kwargs = write_kwargs if write_kwargs else {}
    traj_kwargs = traj_kwargs if traj_kwargs else {}
    log_kwargs = log_kwargs if log_kwargs else {}

    write_kwargs.setdefault(
        "filename",
        Path(f"./{atoms.get_chemical_formula()}-opt.xyz").absolute(),
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

    if filter_func is not None:
        filtered_atoms = filter_func(atoms, **filter_kwargs)
        dyn = optimizer(filtered_atoms, **opt_kwargs)
        if logger:
            logger.info("Using filter %s", filter_func.__name__)
            logger.info("Using optimizer %s", optimizer.__name__)

    else:
        dyn = optimizer(atoms, **opt_kwargs)

    if logger:
        logger.info("Starting geometry optimization")

    dyn.run(fmax=fmax, **dyn_kwargs)

    # Write out optimized structure
    if write_results:
        write(images=atoms, **write_kwargs)

    # Reformat trajectory file from binary
    if traj_kwargs:
        traj = read(opt_kwargs["trajectory"], index=":")
        write(images=traj, **traj_kwargs)

    if logger:
        logger.info("Geometry optimization complete")

    return atoms
