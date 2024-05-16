"""Equation of State."""

from typing import Any, Optional

from ase import Atoms
from ase.eos import EquationOfState
from ase.units import kJ
import numpy as np

from janus_core.calculations.geom_opt import optimize
from janus_core.helpers.janus_types import EoSNames, PathLike
from janus_core.helpers.log import config_logger
from janus_core.helpers.utils import none_to_dict


def calc_eos(  # pylint: disable=too-many-locals
    struct: Atoms,
    struct_name: Optional[str] = None,
    min_lattice: float = 0.95,
    max_lattice: float = 1.05,
    n_lattice: int = 11,
    eos_type: EoSNames = "birchmurnaghan",
    minimize: bool = True,
    minimize_kwargs: Optional[dict[str, Any]] = None,
    file_prefix: Optional[PathLike] = None,
    log_kwargs: Optional[dict[str, Any]] = None,
) -> EquationOfState:
    """
    Calculate equation of state.

    Parameters
    ----------
    struct : Atoms
        Structure.
    struct_name : Optional[str]
        Name of structure. Default is None.
    min_lattice : float
        Minimum lattice constant scale factor. Default is 0.95.
    max_lattice : float
        Maximum lattice constant scale factor. Default is 1.05.
    n_lattice : int
        Number of lattice constants to use. Default is 11.
    eos_type : EoSNames
        Type of fit for equation of state. Default is "birchmurnaghan".
    minimize : bool
        Whether to optimize geometry. Default is True.
    minimize_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to optimize. Default is None.
    file_prefix : Optional[PathLike]
        Prefix for output filenames. Default is inferred from structure name, or
        chemical formula of the structure.
    log_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to `config_logger`. Default is {}.

    Returns
    -------
    tuple[EquationOfState, float, float, float]
        Tuple of equation of state ASE object for structure, and fitted minimum
        bulk_modulus, volume and, energy.
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
    if not 0 < min_lattice < 1:
        raise ValueError("`min_lattice` must be between 0 and 1.")
    if max_lattice <= 1:
        raise ValueError("`max_lattice` must be greater than 1.")

    if minimize:
        if logger:
            minimize_kwargs["log_kwargs"] = {
                "filename": log_kwargs["filename"],
                "name": logger.name,
                "filemode": "a",
            }
        optimize(struct, **minimize_kwargs)

    cell = struct.get_cell()

    lattice_scalars = np.linspace(min_lattice, max_lattice, n_lattice) ** (1 / 3)
    volumes = []
    energies = []
    for lattice_scalar in lattice_scalars:
        struct.set_cell(cell * lattice_scalar, scale_atoms=True)
        energies.append(struct.get_potential_energy())
        volumes.append(struct.get_volume())

    with open(f"{file_prefix}-eos-raw.dat", "w", encoding="utf8") as out:
        print("#Lattice Scalar | Energy [eV] | Volume [Å^3] ", file=out)
        for lattice_scalar, energy, volume in zip(lattice_scalars, energies, volumes):
            print(f"{lattice_scalar} {energy} {volume}", file=out)

    eos = EquationOfState(volumes, energies, eos_type)

    v_0, e_0, bulk_modulus = eos.fit()
    bulk_modulus *= 1.0e24 / kJ
    with open(f"{file_prefix}-eos-fit.dat", "w", encoding="utf8") as out:
        print("#B [GPa] | Energy [eV] | Volume [Å^3] ", file=out)
        print(f"{bulk_modulus} {e_0} {v_0}", file=out)

    return eos, bulk_modulus, v_0, e_0
