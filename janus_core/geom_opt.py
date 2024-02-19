"""Geometry optimisation."""

from ase import Atoms

try:
    from ase.filters import FrechetCellFilter as DefaultFilter
except ImportError:
    from ase.constraints import ExpCellFilter as DefaultFilter

from ase.optimize import LBFGS


#  pylint: disable=dangerous-default-value
#  Dictionaries are not modified within the function.
def optimize(
    atoms: Atoms,
    fmax: float = 0.1,
    dyn_kwargs: dict = {},
    filter_func: callable = DefaultFilter,
    filter_kwargs: dict = {},
    optimizer: callable = LBFGS,
    opt_kwargs: dict = {},
) -> None:
    """Optimize geometry of input structure.

    Parameters
    ----------
    atoms : Atoms
        Atoms object to optimize geometry for.
    fmax : float
        Set force convergence criteria for optimizer in units eV/Å.
    dyn_kwargs : dict
        kwargs to pass to dyn.run. Default is {}.
    filter_func : callable
        Apply constraints to atoms through ASE filter function. Default is `FrechetCellFilter`.
    filter_kwargs : dict
        kwargs to pass to filter_func. Default is {}.
    optimzer : callable
        ASE optimization function. Default is `LBFGS`.
    opt_kwargs : dict
        kwargs to pass to optimzer. Default is {}.

    Returns
    -------
    atoms: Atoms
        Structure with geometry optimized.
    """
    if filter_func is not None:
        filtered_atoms = filter_func(atoms, **filter_kwargs)
        dyn = optimizer(filtered_atoms, **opt_kwargs)
    else:
        dyn = optimizer(atoms, **opt_kwargs)

    dyn.run(fmax=fmax, **dyn_kwargs)
    return atoms
