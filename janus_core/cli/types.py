"""Module containing types used for Janus-Core CLI."""

import ast
from pathlib import Path
from typing import Annotated, Union

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


class TyperDict:  #  pylint: disable=too-few-public-methods
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
        String representation of class.

        Returns
        -------
        str
            Class name and value of string representing a dictionary.
        """
        return f"<TyperDict: value={self.value}>"


StructPath = Annotated[Path, Option(help="Path of structure to simulate.")]

Architecture = Annotated[str, Option(help="MLIP architecture to use for calculations.")]
Device = Annotated[str, Option(help="Device to run calculations on.")]
ModelPath = Annotated[str, Option(help="Path to MLIP model.  [default: None]")]

ReadKwargs = Annotated[
    TyperDict,
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to ase.io.read. Must be passed as a dictionary
            wrapped in quotes, e.g. "{'key' : value}".  [default: "{}"]
            """
        ),
        metavar="DICT",
    ),
]

CalcKwargs = Annotated[
    TyperDict,
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
    TyperDict,
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to ase.io.write when saving results. Must be
            passed as a dictionary wrapped in quotes, e.g. "{'key' : value}".
             [default: "{}"]
            """
        ),
        metavar="DICT",
    ),
]

OptKwargs = Annotated[
    TyperDict,
    Option(
        parser=parse_dict_class,
        help=(
            """
            Keyword arguments to pass to optimizer. Must be passed as a dictionary
            wrapped in quotes, e.g. "{'key' : value}".  [default: "{}"]
            """
        ),
        metavar="DICT",
    ),
]

MinimizeKwargs = Annotated[
    TyperDict,
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
    TyperDict,
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
    TyperDict,
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

LogPath = Annotated[Path, Option(help="Path to save logs to.")]

Summary = Annotated[
    Path, Option(help="Path to save summary of inputs and start/end time.")
]
