"""Geometry optimisation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import write

try:
    from ase.filters import FrechetCellFilter as DefaultFilter
except ImportError:
    from ase.constraints import ExpCellFilter as DefaultFilter

from ase.optimize import LBFGS


def optimize(
    atoms: Atoms,
    fmax: float = 0.1,
    dyn_kwargs: dict[str, Any] | None = None,
    filter_func: callable | None = DefaultFilter,
    filter_kwargs: dict[str, Any] | None = None,
    optimizer: callable = LBFGS,
    opt_kwargs: dict[str, Any] | None = None,
    save_path: Path | str | None = None,
    save_kwargs: dict[str, Any] | None = None,
) -> Atoms:
    """Optimize geometry of input structure.

    Parameters
    ----------
    atoms : Atoms
        Atoms object to optimize geometry for.
    fmax : float
        Set force convergence criteria for optimizer in units eV/Å.
    dyn_kwargs : dict[str, Any] | None
        kwargs to pass to dyn.run. Default is None.
    filter_func : callable | None
        Apply constraints to atoms through ASE filter function. Default is `FrechetCellFilter`.
    filter_kwargs : dict[str, Any] | None
        kwargs to pass to filter_func. Default is None.
    optimzer : callable
        ASE optimization function. Default is `LBFGS`.
    opt_kwargs : dict[str, Any] | None
        kwargs to pass to optimzer. Default is None.
    save_path : Path | str | None
        Path to save optimised structure. Default is None.
    save_kwargs : dict[str, Any] | None
        kwargs to pass to ase.io.write. Default is None.

    Returns
    -------
    atoms: Atoms
        Structure with geometry optimized.
    """
    dyn_kwargs = dyn_kwargs if dyn_kwargs else {}
    filter_kwargs = filter_kwargs if filter_kwargs else {}
    opt_kwargs = opt_kwargs if opt_kwargs else {}
    save_kwargs = save_kwargs if save_kwargs else {}

    if filter_func is not None:
        filtered_atoms = filter_func(atoms, **filter_kwargs)
        dyn = optimizer(filtered_atoms, **opt_kwargs)
    else:
        dyn = optimizer(atoms, **opt_kwargs)

    dyn.run(fmax=fmax, **dyn_kwargs)

    if save_path is not None:
        write(save_path, atoms, **save_kwargs)

    return atoms
