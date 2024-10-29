"""Module for functions for input and output of structures."""

from collections.abc import Collection, Sequence
from copy import copy
import logging
from pathlib import Path
from typing import Any, Optional, get_args

from ase import Atoms
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
    struct : Atoms
        Atoms object to copy or move calculated results to info dict.
    properties : Collection[Properties]
        Properties to copy from results to info dict. Default is ().
    invalidate_calc : bool
        Whether to remove all calculator results after copying properties to info dict.
        Default is False.
    """
    if not properties:
        properties = get_args(Properties)

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
    *,
    arch: Architectures = "mace_mp",
    device: Devices = "cpu",
    model_path: Optional[PathLike] = None,
    calc_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    """
    Configure calculator and attach to structure(s).

    Parameters
    ----------
    struct : Optional[MaybeSequence[Atoms]]
        ASE Atoms structure(s) to attach calculators to.
    arch : Architectures
        MLIP architecture to use for calculations. Default is "mace_mp".
    device : Devices
        Device to run model on. Default is "cpu".
    model_path : Optional[PathLike]
        Path to MLIP model. Default is `None`.
    calc_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to the selected calculator. Default is {}.
    """
    calc_kwargs = calc_kwargs if calc_kwargs else {}

    calculator = choose_calculator(
        arch=arch,
        device=device,
        model_path=model_path,
        **calc_kwargs,
    )

    if isinstance(struct, Sequence):
        for image in struct:
            image.calc = copy(calculator)
    else:
        struct.calc = calculator


def input_structs(
    struct: Optional[MaybeSequence[Atoms]] = None,
    *,
    struct_path: Optional[PathLike] = None,
    read_kwargs: Optional[ASEReadArgs] = None,
    sequence_allowed: bool = True,
    arch: Architectures = "mace_mp",
    device: Devices = "cpu",
    model_path: Optional[PathLike] = None,
    calc_kwargs: Optional[dict[str, Any]] = None,
    set_calc: Optional[bool] = None,
    logger: Optional[logging.Logger] = None,
) -> MaybeSequence[Atoms]:
    """
    Read input structures and/or attach MLIP calculators.

    Parameters
    ----------
    struct : Optional[MaybeSequence[Atoms]]
        ASE Atoms structure(s) to simulate. Required if `struct_path` is None. Default
        is None.
    struct_path : Optional[PathLike]
        Path of structure to simulate. Required if `struct` is None. Default is None.
    read_kwargs : ASEReadArgs
        Keyword arguments to pass to ase.io.read. Default is {}.
    sequence_allowed : bool
        Whether a sequence of Atoms objects is allowed. Default is True.
    arch : Architectures
        MLIP architecture to use for calculations. Default is "mace_mp".
    device : Devices
        Device to run model on. Default is "cpu".
    model_path : Optional[PathLike]
        Path to MLIP model. Default is `None`.
    calc_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to the selected calculator. Default is {}.
    set_calc : Optional[bool]
        Whether to set (new) calculators for structures.  Default is True if
        `struct_path` is specified or `struct` is missing calculators, else default is
        False.
    logger : Optional[logging.Logger]
        Logger if log file has been specified. Default is None.

    Returns
    -------
    MaybeSequence[Atoms]
        Structure(s) with attached MLIP calculators.
    """
    read_kwargs = read_kwargs if read_kwargs else {}

    # Validate parameters
    if not struct and not struct_path:
        raise ValueError(
            "Please specify either the ASE Atoms structure (`struct`) "
            "or a path to the structure file (`struct_path`)"
        )

    if struct and struct_path:
        raise ValueError(
            "You cannot specify both the ASE Atoms structure (`struct`) "
            "and a path to the structure file (`struct_path`)"
        )

    # Read from file
    if struct_path:
        if logger:
            logger.info("Reading structures from file.")
        struct = read(struct_path, **read_kwargs)
        if logger:
            logger.info("Structures read from file.")

        # Return single Atoms object if reading only one image
        if len(struct) == 1:
            struct = struct[0]

    # Check struct is valid type
    if not isinstance(struct, (Atoms, Sequence)) or isinstance(struct, str):
        raise ValueError("`struct` must be an ASE Atoms object or sequence of Atoms")

    # Check struct is valid length
    if not sequence_allowed and isinstance(struct, Sequence):
        raise NotImplementedError(
            "Only one Atoms object at a time can be used for this calculation"
        )

    # If set_calc not specified, set to True if reading file or calculators not attached
    if set_calc is None:
        if struct_path:
            set_calc = True

        elif isinstance(struct, Atoms):
            set_calc = struct.calc is None

        elif isinstance(struct, Sequence):
            set_calc = any(image.calc is None for image in struct)

    if set_calc:
        if logger:
            logger.info("Attaching calculator to structure.")
        attach_calculator(
            struct=struct,
            arch=arch,
            device=device,
            model_path=model_path,
            calc_kwargs=calc_kwargs,
        )
        if logger:
            logger.info("Calculator attached to structure.")

    return struct


def output_structs(
    images: MaybeSequence[Atoms],
    *,
    struct_path: Optional[PathLike] = None,
    set_info: bool = True,
    write_results: bool = False,
    properties: Collection[Properties] = (),
    invalidate_calc: bool = False,
    write_kwargs: Optional[ASEWriteArgs] = None,
) -> None:
    """
    Copy or move calculated results to Atoms.info dict and/or write structures to file.

    Parameters
    ----------
    images : MaybeSequence[Atoms]
        Atoms object or a list of Atoms objects to interact with.
    struct_path : Optional[PathLike]
        Path of structure being simulated. If specified, the file stem is added to the
        info dict under "system_name". Default is None.
    set_info : bool
        True to set info dict from calculated results. Default is True.
    write_results : bool
        True to write out structure with results of calculations. Default is False.
    properties : Collection[Properties]
        Properties to copy from calculated results to info dict. Default is ().
    invalidate_calc : bool
        Whether to remove all calculator results after copying properties to info dict.
        Default is False.
    write_kwargs : Optional[ASEWriteArgs]
        Keyword arguments passed to ase.io.write. Default is {}.
    """
    # Separate kwargs for output_structs from kwargs for ase.io.write
    # This assumes values passed via kwargs have priority over passed parameters
    write_kwargs = write_kwargs if write_kwargs else {}
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

    # Add label for system
    for image in images:
        if struct_path and "system_name" not in image.info:
            image.info["system_name"] = Path(struct_path).stem

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
        if write_format in ("xyz", "extxyz"):
            write_kwargs.setdefault("write_results", not invalidate_calc)
        else:
            write_kwargs.pop("write_results", None)
        write(images=images, **write_kwargs)
