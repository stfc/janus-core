"""Calculate MLIP descriptors for structures."""

from collections.abc import Sequence
from logging import Logger
from pathlib import Path
from typing import Any, Optional

from ase import Atoms
from ase.io import write
import numpy as np

from janus_core.helpers.janus_types import ASEWriteArgs, MaybeSequence
from janus_core.helpers.log import config_logger
from janus_core.helpers.utils import none_to_dict


def calc_descriptors(
    struct: MaybeSequence[Atoms],
    struct_name: Optional[str] = None,
    invariants_only: bool = True,
    calc_elements: bool = False,
    write_results: bool = False,
    write_kwargs: Optional[ASEWriteArgs] = None,
    log_kwargs: Optional[dict[str, Any]] = None,
) -> MaybeSequence[Atoms]:
    """
    Prepare and call calculation of MLIP descriptors for the given structure(s).

    Parameters
    ----------
    struct : MaybeSequence[Atoms]
        Structure(s) to calculate descriptors for.
    struct_name : Optional[str]
        Name of structure. Default is None.
    invariants_only : bool
        Whether only the invariant descriptors should be returned. Default is True.
    calc_elements : bool
        Whether to calculate mean descriptors for each element. Default is False.
    write_results : bool
        True to write out structure with results of calculations. Default is False.
    write_kwargs : Optional[ASEWriteArgs],
        Keyword arguments to pass to ase.io.write if saving structure with
        results of calculations. Default is {}.
    log_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to `config_logger`. Default is {}.

    Returns
    -------
    MaybeSequence[Atoms]
        Atoms object(s) with descriptors attached as info.
    """
    [write_kwargs, log_kwargs] = none_to_dict([write_kwargs, log_kwargs])
    log_kwargs.setdefault("name", __name__)
    logger = config_logger(**log_kwargs)

    # Set default name for output file
    if not struct_name:
        if isinstance(struct, Sequence):
            struct_name = struct[0].get_chemical_formula()
        else:
            struct_name = struct.get_chemical_formula()

    write_kwargs.setdefault(
        "filename",
        Path(f"./{struct_name}-descriptors.xyz").absolute(),
    )

    if isinstance(struct, Sequence):
        if any(not image.calc for image in struct):
            raise ValueError("Please attach a calculator to all images in `struct`.")
    else:
        if not struct.calc:
            raise ValueError("Please attach a calculator to `struct`.")

    if logger:
        logger.info("Starting descriptors calculation")

    if isinstance(struct, Sequence):
        for image in struct:
            image = _calc_descriptors(
                image,
                invariants_only=invariants_only,
                calc_elements=calc_elements,
                logger=logger,
            )
    else:
        struct = _calc_descriptors(
            struct,
            invariants_only=invariants_only,
            calc_elements=calc_elements,
            logger=logger,
        )

    if logger:
        logger.info("Descriptors calculation complete")

    if write_results:
        write(images=struct, **write_kwargs, write_info=True)

    return struct


def _calc_descriptors(
    struct: Atoms,
    invariants_only: bool = True,
    calc_elements: bool = False,
    logger: Optional[Logger] = None,
) -> None:
    """
    Calculate MLIP descriptors for the given structure(s).

    Parameters
    ----------
    struct : Atoms
        Structure(s) to calculate descriptors for.
    invariants_only : bool
        Whether only the invariant descriptors should be returned. Default is True.
    calc_elements : bool
        Whether to calculate mean descriptors for each element. Default is False.
    logger : Optional[Logger]
        Logger if log file has been specified.

    Returns
    -------
    MaybeSequence[Atoms]
        Atoms object(s) with array of descriptors attached.
    """
    if logger:
        logger.info("invariants_only: %s", invariants_only)
        logger.info("calc_elements: %s", calc_elements)

    # Calculate mean descriptor and save mean
    descriptors = struct.calc.get_descriptors(struct, invariants_only=invariants_only)
    descriptor = np.mean(descriptors)
    struct.info["descriptor"] = descriptor

    if calc_elements:
        elements = set(struct.get_chemical_symbols())
        for element in elements:
            pattern = [atom.index for atom in struct if atom.symbol == element]
            struct.info[f"{element}_descriptor"] = np.mean(descriptors[pattern, :])

    return struct
