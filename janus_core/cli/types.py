"""Module containing types used for Janus-Core CLI."""

import ast
from pathlib import Path
from typing import Annotated, Optional, Union

from typer import Option

from janus_core.helpers.janus_types import ASEReadArgs


def parse_dict_class(value: Union[str, ASEReadArgs]):
    """
    Convert string input into a dictionary.

    Parameters
    ----------
    value : str
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
    value : str
        Value of string representing a dictionary.
    """

    def __init__(self, value: str):
        """
        Initialise class.

        Parameters
        ----------
        value : str
            Value of string representing a dictionary.
        """
        self.value = value

    def __str__(self):
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
    Optional[str], Option(help="MLIP architecture to use for calculations.")
]
Device = Annotated[Optional[str], Option(help="Device to run calculations on.")]
ModelPath = Annotated[Optional[str], Option(help="Path to MLIP model.")]

ReadKwargsAll = Annotated[
    Optional[TyperDict],
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to ase.io.read. Must be passed as a dictionary
            wrapped in quotes, e.g. "{'key' : value}". By default,
            read_kwargs['index'] = ':', so all structures are read.
            """
        ),
        metavar="DICT",
    ),
]

ReadKwargsLast = Annotated[
    Optional[TyperDict],
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to ase.io.read. Must be passed as a dictionary
            wrapped in quotes, e.g. "{'key' : value}". By default,
            read_kwargs['index'] = -1, so only the last structure is read.
            """
        ),
        metavar="DICT",
    ),
]

CalcKwargs = Annotated[
    Optional[TyperDict],
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to selected calculator. Must be passed as a
            dictionary wrapped in quotes, e.g. "{'key' : value}". For the default
            architecture ('mace_mp'), "{'model':'small'}" is set unless overwritten.
            """
        ),
        metavar="DICT",
    ),
]

WriteKwargs = Annotated[
    Optional[TyperDict],
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to ase.io.write when saving results. Must be
            passed as a dictionary wrapped in quotes, e.g. "{'key' : value}".
            """
        ),
        metavar="DICT",
    ),
]

OptKwargs = Annotated[
    Optional[TyperDict],
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to optimizer. Must be passed as a dictionary
            wrapped in quotes, e.g. "{'key' : value}".
            """
        ),
        metavar="DICT",
    ),
]

MinimizeKwargs = Annotated[
    Optional[TyperDict],
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to optimizer. Must be passed as a dictionary
            wrapped in quotes, e.g. "{'key' : value}".
            """
        ),
        metavar="DICT",
    ),
]

EnsembleKwargs = Annotated[
    Optional[TyperDict],
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to ensemble initialization. Must be passed as a
            dictionary wrapped in quotes, e.g. "{'key' : value}".
            """
        ),
        metavar="DICT",
    ),
]

PostProcessKwargs = Annotated[
    Optional[TyperDict],
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to post-processer. Must be passed as a dictionary
            wrapped in quotes, e.g. "{'key' : value}".
            """
        ),
        metavar="DICT",
    ),
]

LogPath = Annotated[
    Optional[Path],
    Option(
        help=(
            "Path to save logs to. Default is inferred from the name of the structure "
            "file."
        )
    ),
]

Summary = Annotated[
    Optional[Path],
    Option(
        help=(
            "Path to save summary of inputs, start/end time, and carbon emissions. "
            "Default is inferred from the name of the structure file."
        )
    ),
]
