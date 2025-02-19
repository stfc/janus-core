"""Utility functions for janus_core."""

from __future__ import annotations

from abc import ABC
from collections.abc import Generator, Iterable, Sequence
from io import StringIO
from logging import Logger
from pathlib import Path
from typing import Any, Literal, TextIO

from ase import Atoms
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.style import Style

from janus_core.helpers.janus_types import (
    MaybeSequence,
    PathLike,
    SliceLike,
    StartStopStep,
)


class FileNameMixin(ABC):  # noqa: B024 (abstract-base-class-without-abstract-method)
    """
    Provide mixin functions for standard filename handling.

    Parameters
    ----------
    struct
        Structure from which to derive the default name. If `struct` is a sequence,
        the first structure will be used.
    struct_path
        Path to file containing structures.
    file_prefix
        Default prefix to use.
    *additional
        Components to add to default file_prefix (joined by hyphens).
    """

    def __init__(
        self,
        struct: MaybeSequence[Atoms],
        struct_path: PathLike | None,
        file_prefix: PathLike | None,
        *additional,
    ) -> None:
        """
        Provide mixin functions for standard filename handling.

        Parameters
        ----------
        struct
            Structure(s) from which to derive the default name. If `struct` is a
            sequence, the first structure will be used.
        struct_path
            Path to file structures were read from. Used as default prefix is not None.
        file_prefix
            Default prefix to use.
        *additional
            Components to add to default file_prefix (joined by hyphens).
        """
        self.file_prefix = Path(
            self._get_default_prefix(file_prefix, struct, struct_path, *additional)
        )

    @staticmethod
    def _get_default_prefix(
        file_prefix: PathLike | None,
        struct: MaybeSequence[Atoms],
        struct_path: PathLike | None,
        *additional,
    ) -> str:
        """
        Determine the default prefix from the structure  or provided file_prefix.

        Parameters
        ----------
        file_prefix
            Given file_prefix.
        struct
            Structure(s) from which to derive the default name. If `struct` is a
            sequence, the first structure will be used.
        struct_path
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
        filename: PathLike | None = None,
        prefix_override: str | None = None,
    ) -> Path:
        """
        Set filename using the file prefix and suffix if not specified otherwise.

        Parameters
        ----------
        suffix
            Default suffix to use if `filename` is not specified.
        *additional
            Extra components to add to suffix (joined with hyphens).
        filename
            Filename to use, if specified. Default is None.
        prefix_override
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


def none_to_dict(*dictionaries: Sequence[dict | None]) -> Generator[dict, None, None]:
    """
    Ensure dictionaries that may be None are dictionaries.

    Parameters
    ----------
    *dictionaries
        Sequence of dictionaries that could be None.

    Yields
    ------
    dict
        Input dictionaries or ``{}`` if empty or `None`.
    """
    yield from (dictionary if dictionary else {} for dictionary in dictionaries)


def write_table(
    fmt: Literal["ascii", "csv"],
    file: TextIO | None = None,
    units: dict[str, str] | None = None,
    formats: dict[str, str] | None = None,
    *,
    print_header: bool = True,
    **columns,
) -> StringIO | None:
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
    fmt
        Format to write table in.
    file
        File to dump to. If unspecified function returns
        io.StringIO object simulating file.
    units
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
    StringIO | None
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
    (units,) = none_to_dict(units)
    units.update(
        {
            key.removesuffix("_units"): val
            for key, val in columns.items()
            if key.endswith("_units")
        }
    )

    (formats,) = none_to_dict(formats)
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
) -> None:
    """
    Dump data as an ASCII style table.

    Parameters
    ----------
    file
        File to dump to.
    header
        Column name information.
    columns
        Column data by key (ordered with header info).
    formats
        Python magic string formats to apply
        (must align with header info).

    See Also
    --------
    write_table : Main entry function.
    """
    if header:
        print(f"# {' | '.join(header)}", file=file)

    for cols in zip(*columns.values(), strict=True):
        print(*map(format, cols, formats), file=file)


def _dump_csv(
    file: TextIO,
    header: list[str],
    columns: dict[str, Sequence[Any]],
    formats: Sequence[str],
) -> None:
    """
    Dump data as a csv style table.

    Parameters
    ----------
    file
        File to dump to.
    header
        Column name information.
    columns
        Column data by key (ordered with header info).
    formats
        Python magic string formats to apply
        (must align with header info).

    See Also
    --------
    write_table : Main entry function.
    """
    if header:
        print(",".join(header), file=file)

    for cols in zip(*columns.values(), strict=True):
        print(",".join(map(format, cols, formats)), file=file)


def track_progress(sequence: Sequence | Iterable, description: str) -> Iterable:
    """
    Track the progress of iterating over a sequence.

    This is done by displaying a progress bar in the console using the rich library.
    The function is an iterator over the sequence, updating the progress bar each
    iteration.

    Parameters
    ----------
    sequence
        The sequence to iterate over. Must support "len".
    description
        The text to display to the left of the progress bar.

    Yields
    ------
    Iterable
        An iterable of the values in the sequence.
    """
    text_column = TextColumn("{task.description}")
    bar_column = BarColumn(
        bar_width=None,
        complete_style=Style(color="#FBBB10"),
        finished_style=Style(color="#E38408"),
    )
    completion_column = MofNCompleteColumn()
    time_column = TimeRemainingColumn()
    progress = Progress(
        text_column,
        bar_column,
        completion_column,
        time_column,
        expand=True,
        auto_refresh=False,
    )

    with progress:
        yield from progress.track(sequence, description=description)


def check_files_exist(config: dict, req_file_keys: Sequence[PathLike]) -> None:
    """
    Check files specified in a dictionary read from a configuration file exist.

    Parameters
    ----------
    config
        Dictionary read from configuration file.
    req_file_keys
        Files that must exist if defined in the configuration file.

    Raises
    ------
    FileNotFoundError
        If a key from `req_file_keys` is in the configuration file, but the
        file corresponding to the configuration value do not exist.
    """
    for file_key in config.keys() & req_file_keys:
        # Only check if file key is in the configuration file
        if not Path(config[file_key]).exists():
            raise FileNotFoundError(f"{config[file_key]} does not exist")


def validate_slicelike(maybe_slicelike: SliceLike) -> None:
    """
    Raise an exception if slc is not a valid SliceLike.

    Parameters
    ----------
    maybe_slicelike
        Candidate to test.

    Raises
    ------
    ValueError
        If maybe_slicelike is not SliceLike.
    """
    if isinstance(maybe_slicelike, slice | range | int):
        return
    if isinstance(maybe_slicelike, tuple) and len(maybe_slicelike) == 3:
        start, stop, step = maybe_slicelike
        if (
            (start is None or isinstance(start, int))
            and (stop is None or isinstance(stop, int))
            and isinstance(step, int)
        ):
            return

    raise ValueError(f"{maybe_slicelike} is not a valid SliceLike")


def slicelike_to_startstopstep(index: SliceLike) -> StartStopStep:
    """
    Standarize `SliceLike`s into tuple of `start`, `stop`, `step`.

    Parameters
    ----------
    index
        `SliceLike` to standardize.

    Returns
    -------
    StartStopStep
        Standardized `SliceLike` as `start`, `stop`, `step` triplet.
    """
    validate_slicelike(index)
    if isinstance(index, int):
        if index == -1:
            return (index, None, 1)
        return (index, index + 1, 1)

    if isinstance(index, slice | range):
        return (index.start, index.stop, index.step)

    return index


def selector_len(slc: SliceLike | list, selectable_length: int) -> int:
    """
    Calculate the length of a selector applied to an indexable of a given length.

    Parameters
    ----------
    slc
        The applied SliceLike or list for selection.
    selectable_length
        The length of the selectable object.

    Returns
    -------
    int
        Length of the result of applying slc.
    """
    if isinstance(slc, int):
        return 1
    if isinstance(slc, list):
        return len(slc)
    start, stop, step = slicelike_to_startstopstep(slc)
    if stop is None:
        stop = selectable_length
    return len(range(start, stop, step))


def set_log_tracker(
    attach_logger: bool, log_kwargs: dict, track_carbon: bool
) -> tuple[bool, bool]:
    """
    Set attach_logger and track_carbon default values.

    Parameters
    ----------
    attach_logger
        Whether to attach a logger.
    log_kwargs
        Keyword arguments to pass to `config_logger`.
    track_carbon
        Whether to track carbon emissions of calculation.

    Returns
    -------
    tuple[bool, bool]
        Default values for attach_logger and track_carbon.
    """
    if "filename" in log_kwargs:
        attach_logger = True
    else:
        attach_logger = attach_logger if attach_logger else False

    if not attach_logger:
        if track_carbon:
            raise ValueError("Carbon tracking requires logging to be enabled")
        track_carbon = False
    else:
        track_carbon = track_carbon if track_carbon is not None else True

    return attach_logger, track_carbon


def set_minimize_logging(
    logger: Logger | None,
    minimize_kwargs: dict[str, Any],
    log_kwargs: dict[str, Any],
    track_carbon: bool,
) -> None:
    """
    Set kwargs to be passed to GeomOpt when logging has been set up.

    Parameters
    ----------
    logger
        Logger, which may already be set up.
    minimize_kwargs
        Kwargs to set for GeomOpt.
    log_kwargs
        Kwargs used in setting up Logger.
    track_carbon
        Whether carbon emissions are being tracked.
    """
    if logger:
        minimize_kwargs["log_kwargs"] = {
            "filename": log_kwargs["filename"],
            "name": logger.name,
            "filemode": "a",
        }
        minimize_kwargs["track_carbon"] = track_carbon
