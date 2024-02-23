"""Geometry optimization."""

from typing import Any, Optional

from ase import Atoms
from ase.io import read, write

try:
    from ase.filters import FrechetCellFilter as DefaultFilter
except ImportError:
    from ase.constraints import ExpCellFilter as DefaultFilter

from ase.optimize import LBFGS


def optimize(
    atoms: Atoms,
    fmax: float = 0.1,
    dyn_kwargs: Optional[dict[str, Any]] = None,
    filter_func: Optional[callable] = DefaultFilter,
    filter_kwargs: Optional[dict[str, Any]] = None,
    optimizer: callable = LBFGS,
    opt_kwargs: Optional[dict[str, Any]] = None,
    struct_kwargs: Optional[dict[str, Any]] = None,
    traj_kwargs: Optional[dict[str, Any]] = None,
) -> Atoms:
    """Optimize geometry of input structure.

    Parameters
    ----------
    atoms : Atoms
        Atoms object to optimize geometry for.
    fmax : float
        Set force convergence criteria for optimizer in units eV/Å.
    dyn_kwargs : Optional[dict[str, Any]]
        kwargs to pass to dyn.run. Default is {}.
    filter_func : Optional[callable]
        Apply constraints to atoms through ASE filter function.
        Default is `FrechetCellFilter` if available otherwise `ExpCellFilter`.
    filter_kwargs : Optional[dict[str, Any]]
        kwargs to pass to filter_func. Default is {}.
    optimzer : callable
        ASE optimization function. Default is `LBFGS`.
    opt_kwargs : Optional[dict[str, Any]]
        kwargs to pass to optimzer. Default is {}.
    struct_kwargs : Optional[dict[str, Any]]
        kwargs to pass to ase.io.write to save optimized structure.
        Must include "filename" keyword. Default is {}.
    traj_kwargs : Optional[dict[str, Any]]
        kwargs to pass to ase.io.write to save optimization trajectory.
        Must include "filename" keyword. Default is {}.

    Returns
    -------
    atoms: Atoms
        Structure with geometry optimized.
    """
    dyn_kwargs = dyn_kwargs if dyn_kwargs else {}
    filter_kwargs = filter_kwargs if filter_kwargs else {}
    opt_kwargs = opt_kwargs if opt_kwargs else {}
    struct_kwargs = struct_kwargs if struct_kwargs else {}
    traj_kwargs = traj_kwargs if traj_kwargs else {}

    if struct_kwargs and "filename" not in struct_kwargs:
        raise ValueError("'filename' must be included in struct_kwargs")

    if traj_kwargs and "filename" not in traj_kwargs:
        raise ValueError("'filename' must be included in traj_kwargs")

    if traj_kwargs and "trajectory" not in opt_kwargs:
        raise ValueError(
            "'trajectory' must be a key in opt_kwargs to save the trajectory."
        )

    if filter_func is not None:
        filtered_atoms = filter_func(atoms, **filter_kwargs)
        dyn = optimizer(filtered_atoms, **opt_kwargs)
    else:
        dyn = optimizer(atoms, **opt_kwargs)

    dyn.run(fmax=fmax, **dyn_kwargs)

    # Write out optimized structure
    if struct_kwargs:
        write(images=atoms, **struct_kwargs)

    # Reformat trajectory file from binary
    if traj_kwargs:
        traj = read(opt_kwargs["trajectory"], index=":")
        write(images=traj, **traj_kwargs)

    return atoms
