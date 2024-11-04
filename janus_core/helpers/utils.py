"""Utility functions for janus_core."""

from __future__ import annotations

from abc import ABC
from collections.abc import Generator, Iterable, Sequence
from io import StringIO
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

from janus_core.helpers.janus_types import MaybeSequence, PathLike


class FileNameMixin(ABC):  # noqa: B024 (abstract-base-class-without-abstract-method)
    """
    Provide mixin functions for standard filename handling.

    Parameters
    ----------
    struct : MaybeSequence[Atoms]
        Structure from which to derive the default name. If `struct` is a sequence,
        the first structure will be used.
    struct_path : PathLike | None
        Path to file containing structures.
    file_prefix : PathLike | None
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
        struct_path: PathLike | None,
        file_prefix: PathLike | None,
        *additional,
    ) -> None:
        """
        Provide mixin functions for standard filename handling.

        Parameters
        ----------
        struct : MaybeSequence[Atoms]
            Structure(s) from which to derive the default name. If `struct` is a
            sequence, the first structure will be used.
        struct_path : PathLike | None
            Path to file structures were read from. Used as default prefix is not None.
        file_prefix : PathLike | None
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
        file_prefix : PathLike | None
            Given file_prefix.
        struct : MaybeSequence[Atoms]
            Structure(s) from which to derive the default name. If `struct` is a
            sequence, the first structure will be used.
        struct_path : PathLike | None
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
        suffix : str
            Default suffix to use if `filename` is not specified.
        *additional
            Extra components to add to suffix (joined with hyphens).
        filename : PathLike | None
            Filename to use, if specified. Default is None.
        prefix_override : str | None
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
    *dictionaries : Sequence[dict | None]
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
) -> None:
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


def track_progress(sequence: Sequence | Iterable, description: str) -> Iterable:
    """
    Track the progress of iterating over a sequence.

    This is done by displaying a progress bar in the console using the rich library.
    The function is an iterator over the sequence, updating the progress bar each
    iteration.

    Parameters
    ----------
    sequence : Iterable
        The sequence to iterate over. Must support "len".
    description : str
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
    config : dict
        Dictionary read from configuration file.
    req_file_keys : Sequence[Pathlike]
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
