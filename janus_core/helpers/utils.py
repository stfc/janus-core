"""Utility functions for janus_core."""

from abc import ABC
from collections.abc import Collection, Generator, Iterable, Sequence
from copy import copy
from io import StringIO
import logging
from pathlib import Path
from typing import Any, Literal, Optional, TextIO, get_args

from ase import Atoms
from ase.io import read, write
from ase.io.formats import filetype
from spglib import get_spacegroup

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


class FileNameMixin(ABC):  # noqa: B024 (abstract-base-class-without-abstract-method)
    """
    Provide mixin functions for standard filename handling.

    Parameters
    ----------
    struct : MaybeSequence[Atoms]
        Structure from which to derive the default name. If `struct` is a sequence,
        the first structure will be used.
    struct_path : Optional[PathLike]
        Path to file containing structures.
    file_prefix : Optional[PathLike]
        Default prefix to use.
    *additional
        Components to add to default file_prefix (joined by hyphens).

    Methods
    -------
    _get_default_prefix(file_prefix, struct)
        Return a prefix from the provided file_prefix or from chemical formula of
        struct.
    _build_filename(suffix, *additional, filename, prefix_override)
         Return a standard format filename if filename not provided.
    """

    def __init__(
        self,
        struct: MaybeSequence[Atoms],
        struct_path: Optional[PathLike],
        file_prefix: Optional[PathLike],
        *additional,
    ):
        """
        Provide mixin functions for standard filename handling.

        Parameters
        ----------
        struct : MaybeSequence[Atoms]
            Structure(s) from which to derive the default name. If `struct` is a
            sequence, the first structure will be used.
        struct_path : Optional[PathLike]
            Path to file structures were read from. Used as default prefix is not None.
        file_prefix : Optional[PathLike]
            Default prefix to use.
        *additional
            Components to add to default file_prefix (joined by hyphens).
        """
        self.file_prefix = Path(
            self._get_default_prefix(file_prefix, struct, struct_path, *additional)
        )

    @staticmethod
    def _get_default_prefix(
        file_prefix: Optional[PathLike],
        struct: MaybeSequence[Atoms],
        struct_path: Optional[PathLike],
        *additional,
    ) -> str:
        """
        Determine the default prefix from the structure  or provided file_prefix.

        Parameters
        ----------
        file_prefix : Optional[PathLike]
            Given file_prefix.
        struct : MaybeSequence[Atoms]
            Structure(s) from which to derive the default name. If `struct` is a
            sequence, the first structure will be used.
        struct_path : Optional[PathLike]
            Path to file containing structures.
        *additional
            Components to add to default file_prefix (joined by hyphens).

        Returns
        -------
        str
            File prefix.
        """
        if file_prefix is not None:
            return str(file_prefix)

        # Prefer file stem, otherwise use formula
        if struct_path is not None:
            struct_name = Path(struct_path).stem
        elif isinstance(struct, Sequence):
            struct_name = struct[0].get_chemical_formula()
        else:
            struct_name = struct.get_chemical_formula()

        return "-".join((struct_name, *filter(None, additional)))

    def _build_filename(
        self,
        suffix: str,
        *additional,
        filename: Optional[PathLike] = None,
        prefix_override: Optional[str] = None,
    ) -> Path:
        """
        Set filename using the file prefix and suffix if not specified otherwise.

        Parameters
        ----------
        suffix : str
            Default suffix to use if `filename` is not specified.
        *additional
            Extra components to add to suffix (joined with hyphens).
        filename : Optional[PathLike]
            Filename to use, if specified. Default is None.
        prefix_override : Optional[str]
            Replace file_prefix if not None.

        Returns
        -------
        Path
            Filename specified, or default filename.
        """
        if filename:
            built_filename = Path(filename)
        else:
            prefix = str(
                prefix_override if prefix_override is not None else self.file_prefix
            )
            built_filename = Path("-".join((prefix, *filter(None, additional), suffix)))

        # Make directory if it does not exist
        built_filename.parent.mkdir(parents=True, exist_ok=True)
        return built_filename


def spacegroup(
    struct: Atoms, sym_tolerance: float = 0.001, angle_tolerance: float = -1.0
) -> str:
    """
    Determine the spacegroup for a structure.

    Parameters
    ----------
    struct : Atoms
        Structure as an ase Atoms object.
    sym_tolerance : float
        Atom displacement tolerance for spglib symmetry determination, in Å.
        Default is 0.001.
    angle_tolerance : float
        Angle precision for spglib symmetry determination, in degrees. Default is -1.0,
        which means an internally optimized routine is used to judge symmetry.

    Returns
    -------
    str
        Spacegroup name.
    """
    return get_spacegroup(
        cell=(
            struct.get_cell(),
            struct.get_scaled_positions(),
            struct.get_atomic_numbers(),
        ),
        symprec=sym_tolerance,
        angle_tolerance=angle_tolerance,
    )


def none_to_dict(dictionaries: Sequence[Optional[dict]]) -> Generator[dict, None, None]:
    """
    Ensure dictionaries that may be None are dictionaires.

    Parameters
    ----------
    dictionaries : Sequence[dict]
        Sequence of dictionaries that be be None.

    Yields
    ------
    dict
        Input dictionaries or ``{}`` if empty or `None`.
    """
    yield from (dictionary if dictionary else {} for dictionary in dictionaries)


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


def write_table(
    fmt: Literal["ascii", "csv"],
    file: Optional[TextIO] = None,
    units: Optional[dict[str, str]] = None,
    formats: Optional[dict[str, str]] = None,
    *,
    print_header: bool = True,
    **columns,
) -> Optional[StringIO]:
    """
    Dump a table in a standard format.

    Table columns are passed as kwargs, with the column header being
    the keyword name and data the value.

    Each header also supports an optional "<header>_units" or
    "<header>_format" kwarg to define units and format for the column.
    These can also be passed explicitly through the respective
    dictionaries where the key is the "header".

    Parameters
    ----------
    fmt : {'ascii', 'csv'}
        Format to write table in.
    file : TextIO
        File to dump to. If unspecified function returns
        io.StringIO object simulating file.
    units : dict[str, str]
        Units as ``{key: unit}``:

        key
            To align with those in `columns`.
        unit
            String defining unit.

        Units which do not match any columns in `columns` are
        ignored.
    formats : dict[str, str]
        Output formats as ``{key: format}``:

        key
            To align with those in `columns`.
        format
            Python magic format string to use.
    print_header : bool
        Whether to print the header or just append formatted data.
    **columns : dict[str, Any]
        Dictionary of columns names to data with optional
        "<header>_units"/"<header>_format" to define units/format.

        See: Examples.

    Returns
    -------
    Optional[StringIO]
        If no file given write columns to StringIO.

    Notes
    -----
    Passing "kwarg_units" or "kwarg_format" takes priority over
    those passed in the `units`/`formats` dicts.

    Examples
    --------
    >>> data = write_table(fmt="ascii", single=(1, 2, 3),
    ...                    double=(2,4,6), double_units="THz")
    >>> print(*data, sep="", end="")
    # single | double [THz]
    1 2
    2 4
    3 6
    >>> data = write_table(fmt="csv", a=(3., 5.), b=(11., 12.),
    ...                    units={'a': 'eV', 'b': 'Ha'})
    >>> print(*data, sep="", end="")
    a [eV],b [Ha]
    3.0,11.0
    5.0,12.0
    >>> data = write_table(fmt="csv", a=(3., 5.), b=(11., 12.),
    ...                    formats={"a": ".3f"})
    >>> print(*data, sep="", end="")
    a,b
    3.000,11.0
    5.000,12.0
    >>> data = write_table(fmt="ascii", single=(1, 2, 3),
    ...                    double=(2,4,6), double_units="THz",
    ...                    print_header=False)
    >>> print(*data, sep="", end="")
    1 2
    2 4
    3 6
    """
    units = units if units else {}
    units.update(
        {
            key.removesuffix("_units"): val
            for key, val in columns.items()
            if key.endswith("_units")
        }
    )

    formats = formats if formats else {}
    formats.update(
        {
            key.removesuffix("_format"): val
            for key, val in columns.items()
            if key.endswith("_format")
        }
    )

    columns = {
        key: val if isinstance(val, Iterable) else (val,)
        for key, val in columns.items()
        if not key.endswith("_units")
    }

    if print_header:
        header = [
            f"{datum}" + (f" [{unit}]" if (unit := units.get(datum, "")) else "")
            for datum in columns
        ]
    else:
        header = ()

    dump_loc = file if file is not None else StringIO()

    write_fmt = [formats.get(key, "") for key in columns]

    if fmt == "ascii":
        _dump_ascii(dump_loc, header, columns, write_fmt)
    elif fmt == "csv":
        _dump_csv(dump_loc, header, columns, write_fmt)

    if file is None:
        dump_loc.seek(0)
        return dump_loc
    return None


def _dump_ascii(
    file: TextIO,
    header: list[str],
    columns: dict[str, Sequence[Any]],
    formats: Sequence[str],
):
    """
    Dump data as an ASCII style table.

    Parameters
    ----------
    file : TextIO
        File to dump to.
    header : list[str]
        Column name information.
    columns : dict[str, Sequence[Any]]
        Column data by key (ordered with header info).
    formats : Sequence[str]
        Python magic string formats to apply
        (must align with header info).

    See Also
    --------
    write_table : Main entry function.
    """
    if header:
        print(f"# {' | '.join(header)}", file=file)

    for cols in zip(*columns.values()):
        print(*map(format, cols, formats), file=file)


def _dump_csv(
    file: TextIO,
    header: list[str],
    columns: dict[str, Sequence[Any]],
    formats: Sequence[str],
):
    """
    Dump data as a csv style table.

    Parameters
    ----------
    file : TextIO
        File to dump to.
    header : list[str]
        Column name information.
    columns : dict[str, Sequence[Any]]
        Column data by key (ordered with header info).
    formats : Sequence[str]
        Python magic string formats to apply
        (must align with header info).

    See Also
    --------
    write_table : Main entry function.
    """
    if header:
        print(",".join(header), file=file)

    for cols in zip(*columns.values()):
        print(",".join(map(format, cols, formats)), file=file)
