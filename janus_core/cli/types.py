"""Module containing types used for Janus-Core CLI."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, get_args

from click import Choice
from typer import Option

from janus_core.helpers.janus_types import Architectures, Devices

if TYPE_CHECKING:
    from janus_core.helpers.janus_types import ASEReadArgs


def parse_dict_class(value: str | ASEReadArgs) -> TyperDict:
    """
    Convert string input into a dictionary.

    Parameters
    ----------
    value
        String representing dictionary to be parsed.

    Returns
    -------
    TyperDict
        Parsed string as a dictionary.
    """
    if isinstance(value, dict):
        return TyperDict(value)
    return TyperDict(ast.literal_eval(value))


class TyperDict:
    """
    Custom dictionary for typer.

    Parameters
    ----------
    value
        Value of string representing a dictionary.
    """

    def __init__(self, value: str) -> None:
        """
        Initialise class.

        Parameters
        ----------
        value
            Value of string representing a dictionary.
        """
        self.value = value

    def __str__(self) -> str:
        """
        Return string representation of class.

        Returns
        -------
        str
            Class name and value of string representing a dictionary.
        """
        return f"<TyperDict: value={self.value}>"


StructPath = Annotated[Path, Option(help="Path of structure to simulate.")]

Architecture = Annotated[
    str | None,
    Option(
        click_type=Choice(get_args(Architectures)),
        help="MLIP architecture to use for calculations.",
    ),
]
Device = Annotated[
    str | None,
    Option(
        click_type=Choice(get_args(Devices)),
        help="Device to run calculations on.",
    ),
]
ModelPath = Annotated[str | None, Option(help="Path to MLIP model.")]

FilePrefix = Annotated[
    Path | None,
    Option(
        help=(
            """
                Prefix for output files, including directories. Default directory is
                ./janus_results, and default filename prefix is inferred from the
                input stucture filename.
                """
        )
    ),
]

ReadKwargsAll = Annotated[
    TyperDict | None,
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to ase.io.read. Must be passed as a dictionary
            wrapped in quotes, e.g. "{'key': value}". By default,
            read_kwargs['index'] = ':', so all structures are read.
            """
        ),
        metavar="DICT",
    ),
]

ReadKwargsLast = Annotated[
    TyperDict | None,
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to ase.io.read. Must be passed as a dictionary
            wrapped in quotes, e.g. "{'key': value}". By default,
            read_kwargs['index'] = -1, so only the last structure is read.
            """
        ),
        metavar="DICT",
    ),
]

CalcKwargs = Annotated[
    TyperDict | None,
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to selected calculator. Must be passed as a
            dictionary wrapped in quotes, e.g. "{'key': value}". For the default
            architecture ('mace_mp'), "{'model': 'small'}" is set unless overwritten.
            """
        ),
        metavar="DICT",
    ),
]

WriteKwargs = Annotated[
    TyperDict | None,
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to ase.io.write when saving results. Must be
            passed as a dictionary wrapped in quotes, e.g. "{'key': value}".
            """
        ),
        metavar="DICT",
    ),
]

OptKwargs = Annotated[
    TyperDict | None,
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to optimizer. Must be passed as a dictionary
            wrapped in quotes, e.g. "{'key': value}".
            """
        ),
        metavar="DICT",
    ),
]

MinimizeKwargs = Annotated[
    TyperDict | None,
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to optimizer. Must be passed as a dictionary
            wrapped in quotes, e.g. "{'key': value}".
            """
        ),
        metavar="DICT",
    ),
]

DoSKwargs = Annotated[
    TyperDict | None,
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to run_total_dos. Must be passed as a dictionary
            wrapped in quotes, e.g. "{'key' : value}".
            """
        ),
        metavar="DICT",
    ),
]

PDoSKwargs = Annotated[
    TyperDict | None,
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to run_projected_dos. Must be passed as a
            dictionary wrapped in quotes, e.g. "{'key' : value}".
            """
        ),
        metavar="DICT",
    ),
]

EnsembleKwargs = Annotated[
    TyperDict | None,
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to ensemble initialization. Must be passed as a
            dictionary wrapped in quotes, e.g. "{'key': value}".
            """
        ),
        metavar="DICT",
    ),
]

DisplacementKwargs = Annotated[
    TyperDict | None,
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to generate_displacements. Must be passed as a
            dictionary wrapped in quotes, e.g. "{'key' : value}".
            """
        ),
        metavar="DICT",
    ),
]

PostProcessKwargs = Annotated[
    TyperDict | None,
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to post-processer. Must be passed as a dictionary
            wrapped in quotes, e.g. "{'key': value}".
            """
        ),
        metavar="DICT",
    ),
]

NebKwargs = Annotated[
    TyperDict | None,
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to neb_method. Must be passed as a dictionary
            wrapped in quotes, e.g. "{'key': value}".
            """
        ),
        metavar="DICT",
    ),
]

NebOptKwargs = Annotated[
    TyperDict | None,
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to neb_optimizer. Must be passed as a dictionary
            wrapped in quotes, e.g. "{'key': value}".
            """
        ),
        metavar="DICT",
    ),
]

InterpolationKwargs = Annotated[
    TyperDict | None,
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to interpolator. Must be passed as a dictionary
            wrapped in quotes, e.g. "{'key': value}".
            """
        ),
        metavar="DICT",
    ),
]

PostProcessKwargs = Annotated[
    TyperDict | None,
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to post-processer. Must be passed as a dictionary
            wrapped in quotes, e.g. "{'key': value}".
            """
        ),
        metavar="DICT",
    ),
]

CorrelationKwargs = Annotated[
    TyperDict | None,
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to md for on-the-fly correlations. Must be
            passed as a list of dictionaries wrapped in quotes, e.g.
            "[{'key' : values}]".
            """
        ),
        metavar="DICT",
    ),
]

LogPath = Annotated[
    Path | None,
    Option(help=("Path to save logs to. Default is inferred from `file_prefix`")),
]

Summary = Annotated[
    Path | None,
    Option(
        help=(
            "Path to save summary of inputs, start/end time, and carbon emissions. "
            "Default is inferred from `file_prefix`."
        )
    ),
]
