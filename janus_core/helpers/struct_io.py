"""Module for functions for input and output of structures."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from copy import copy
import logging
from pathlib import Path
from typing import Any, get_args

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.io.formats import filetype

from janus_core.helpers.janus_types import (
    Architectures,
    ASEReadArgs,
    ASEWriteArgs,
    Devices,
    MaybeSequence,
    PathLike,
    Properties,
)
from janus_core.helpers.mlip_calculators import choose_calculator
from janus_core.helpers.utils import build_file_dir, none_to_dict


def results_to_info(
    struct: Atoms,
    *,
    properties: Collection[Properties] = (),
    invalidate_calc: bool = False,
) -> None:
    """
    Copy or move MLIP calculated results to Atoms.info dict.

    Parameters
    ----------
    struct
        Atoms object to copy or move calculated results to info dict.
    properties
        Properties to copy from results to info dict. Default is ().
    invalidate_calc
        Whether to remove all calculator results after copying properties to info dict.
        Default is False.
    """
    if not properties:
        properties = get_args(Properties)

    if struct.calc and "model" in struct.calc.parameters:
        struct.info["model"] = struct.calc.parameters["model"]

    # Only add to info if MLIP calculator with "arch" parameter set
    if struct.calc and "arch" in struct.calc.parameters:
        arch = struct.calc.parameters["arch"]
        struct.info["arch"] = arch

        for key in properties & struct.calc.results.keys():
            tag = f"{arch}_{key}"
            value = struct.calc.results[key]
            if key == "forces":
                struct.arrays[tag] = value
            else:
                struct.info[tag] = value

        # Remove all calculator results
        if invalidate_calc:
            struct.calc.results = {}


def attach_calculator(
    struct: MaybeSequence[Atoms],
    arch: Architectures,
    *,
    device: Devices = "cpu",
    model: PathLike | None = None,
    calc_kwargs: dict[str, Any] | None = None,
) -> None:
    """
    Configure calculator and attach to structure(s).

    Parameters
    ----------
    struct
        ASE Atoms structure(s) to attach calculators to.
    arch
        MLIP architecture to use for calculations.
    device
        Device to run model on. Default is "cpu".
    model
        MLIP model label, path to model, or loaded model. Default is `None`.
    calc_kwargs
        Keyword arguments to pass to the selected calculator. Default is {}.
    """
    (calc_kwargs,) = none_to_dict(calc_kwargs)

    calculator = choose_calculator(
        arch=arch,
        device=device,
        model=model,
        **calc_kwargs,
    )

    if isinstance(struct, Sequence):
        for image in struct:
            image.calc = copy(calculator)
    else:
        struct.calc = calculator


def read_structs(
    struct: MaybeSequence[Atoms] | PathLike,
    *,
    read_kwargs: ASEReadArgs | None = None,
    sequence_allowed: bool = True,
    logger: logging.Logger | None = None,
) -> tuple[MaybeSequence[Atoms], PathLike | None]:
    """
    Read input structures.

    Parameters
    ----------
    struct
        ASE Atoms structure(s), or filepath to structure(s) to simulate.
    read_kwargs
        Keyword arguments to pass to ase.io.read. Default is {}.
    sequence_allowed
        Whether a sequence of Atoms objects is allowed. Default is True.
    logger
        Logger if log file has been specified. Default is None.

    Returns
    -------
    tuple[MaybeSequence[Atoms], PathLike | None]
        Loaded structure(s) and filepath.
    """
    (read_kwargs,) = none_to_dict(read_kwargs)

    struct_path = struct if isinstance(struct, PathLike) else None

    # Read from file
    if struct_path:
        if logger:
            logger.info("Reading structures from file.")

        if not Path(struct_path).exists():
            raise ValueError("`struct` file could not be found")

        struct = read(struct_path, **read_kwargs)
        if logger:
            logger.info("Structures read from file.")

        # Return single Atoms object if reading only one image
        if len(struct) == 1:
            struct = struct[0]

    # Check struct is valid type
    if not isinstance(struct, Atoms | Sequence) or isinstance(struct, str):
        raise ValueError("`struct` must be an ASE Atoms object or sequence of Atoms")

    # Check struct is valid length
    if not sequence_allowed and isinstance(struct, Sequence):
        raise NotImplementedError(
            "Only one Atoms object at a time can be used for this calculation"
        )

    return struct, struct_path


def input_structs(
    struct: MaybeSequence[Atoms] | PathLike,
    *,
    read_kwargs: ASEReadArgs | None = None,
    sequence_allowed: bool = True,
    arch: Architectures | None = None,
    device: Devices = "cpu",
    model: PathLike | None = None,
    calc_kwargs: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> tuple[MaybeSequence[Atoms], PathLike | None]:
    """
    Read input structures and/or attach MLIP calculators.

    Parameters
    ----------
    struct
        ASE Atoms structure(s), or filepath to structure(s) to simulate.
    read_kwargs
        Keyword arguments to pass to ase.io.read. Default is {}.
    sequence_allowed
        Whether a sequence of Atoms objects is allowed. Default is True.
    arch
        MLIP architecture to use for calculations. Default is None.
    device
        Device to run model on. Default is "cpu".
    model
        MLIP model label, path to model, or loaded model. Default is `None`.
    calc_kwargs
        Keyword arguments to pass to the selected calculator. Default is {}.
    logger
        Logger if log file has been specified. Default is None.

    Returns
    -------
    tuple[MaybeSequence[Atoms], PathLike | None]
        Structure(s) with attached MLIP calculators.
    """
    (read_kwargs,) = none_to_dict(read_kwargs)

    struct, struct_path = read_structs(
        struct=struct,
        read_kwargs=read_kwargs,
        sequence_allowed=sequence_allowed,
        logger=logger,
    )

    if arch:
        if logger:
            logger.info("Attaching calculator to structure.")
        attach_calculator(
            struct=struct,
            arch=arch,
            device=device,
            model=model,
            calc_kwargs=calc_kwargs,
        )
        if logger:
            logger.info("Calculator attached to structure.")
    elif struct.calc is None:
        raise ValueError("A calculator must be attached to `struct`")
    elif isinstance(struct.calc, SinglePointCalculator):
        raise ValueError("The attached calculator cannot be used for new calculations.")

    return struct, struct_path


def output_structs(
    images: MaybeSequence[Atoms],
    *,
    struct_path: PathLike | None = None,
    set_info: bool = True,
    write_results: bool = False,
    properties: Collection[Properties] = (),
    invalidate_calc: bool = False,
    write_kwargs: ASEWriteArgs | None = None,
    config_type: str = "",
) -> None:
    """
    Copy or move calculated results to Atoms.info dict and/or write structures to file.

    Parameters
    ----------
    images
        Atoms object or a list of Atoms objects to interact with.
    struct_path
        Path of structure being simulated. If specified, the file stem is added to the
        info dict under "system_name". Default is None.
    set_info
        True to set info dict from calculated results. Default is True.
    write_results
        True to write out structure with results of calculations. Default is False.
    properties
        Properties to copy from calculated results to info dict. Default is ().
    invalidate_calc
        Whether to remove all calculator results after copying properties to info dict.
        Default is False.
    write_kwargs
        Keyword arguments passed to ase.io.write. Default is {}.
    config_type
        Label for calculation that generated configurations, added to the Atoms info if
        it does not exist already. Default is no label.
    """
    # Separate kwargs for output_structs from kwargs for ase.io.write
    # This assumes values passed via kwargs have priority over passed parameters
    (write_kwargs,) = none_to_dict(write_kwargs)
    set_info = write_kwargs.pop("set_info", set_info)
    properties = write_kwargs.pop("properties", properties)
    invalidate_calc = write_kwargs.pop("invalidate_calc", invalidate_calc)

    if isinstance(images, Atoms):
        images = (images,)

    if set_info:
        for image in images:
            results_to_info(
                image, properties=properties, invalidate_calc=invalidate_calc
            )
    else:
        # Label architecture even if not copying results to info
        for image in images:
            if image.calc and "arch" in image.calc.parameters:
                image.info["arch"] = image.calc.parameters["arch"]
            if image.calc and "model" in image.calc.parameters:
                image.info["model"] = image.calc.parameters["model"]

    # Add labels for system and configuration type
    for image in images:
        if struct_path and "system_name" not in image.info:
            image.info["system_name"] = Path(struct_path).stem
        if config_type:
            image.info.setdefault("config_type", config_type)

    if write_results:
        # Check required filename is specified
        if "filename" not in write_kwargs:
            raise ValueError(
                "`filename` must be specified in `write_kwargs` to write results"
            )

        # Get format of file to be written
        write_format = write_kwargs.get(
            "format", filetype(write_kwargs["filename"], read=False)
        )

        # write_results is only a valid kwarg for extxyz
        if write_format == "extxyz":
            write_kwargs.setdefault("write_results", not invalidate_calc)
        else:
            write_kwargs.pop("write_results", None)

        build_file_dir(write_kwargs["filename"])
        write(images=images, **write_kwargs)
