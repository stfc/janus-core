"""Set up commandline interface."""

import ast
from pathlib import Path
from typing import Annotated

import typer

from janus_core.single_point import SinglePoint

app = typer.Typer()


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


def parse_dict_class(value: str):
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
    return TyperDict(ast.literal_eval(value))


@app.command()
def singlepoint(
    struct_path: Annotated[
        Path, typer.Option("--struct", help="Path of structure to simulate")
    ],
    architecture: Annotated[
        str, typer.Option("--arch", help="MLIP architecture to use for calculations")
    ] = "mace_mp",
    device: Annotated[str, typer.Option(help="Device to run calculations on")] = "cpu",
    properties: Annotated[
        list[str],
        typer.Option(
            "--property",
            help=(
                "Properties to calculate. If not specified, 'energy', 'forces' "
                "and 'stress' will be returned."
            ),
        ),
    ] = None,
    read_kwargs: Annotated[
        TyperDict,
        typer.Option(
            parser=parse_dict_class,
            help="Keyword arguments to pass to ase.io.read  [default: {}]",
            metavar="DICT",
        ),
    ] = None,
    calc_kwargs: Annotated[
        TyperDict,
        typer.Option(
            parser=parse_dict_class,
            help="Keyword arguments to pass to selected calculator  [default: {}]",
            metavar="DICT",
        ),
    ] = None,
    write_kwargs: Annotated[
        TyperDict,
        typer.Option(
            parser=parse_dict_class,
            help=(
                "Keyword arguments to pass to ase.io.write when saving "
                "results [default: {}]"
            ),
            metavar="DICT",
        ),
    ] = None,
    log_file: Annotated[
        str, typer.Option("--log", help="File to save logs")
    ] = "singlepoint.log",
):
    """
    Perform single point calculations and save to file.

    Parameters
    ----------
    struct_path : Path
        Path of structure to simulate.
    architecture : Optional[str]
        MLIP architecture to use for single point calculations.
        Default is "mace_mp".
    device : Optional[str]
        Device to run model on. Default is "cpu".
    properties : Optional[str]
        Physical properties to calculate. Default is "energy".
    read_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ase.io.read. Default is {}.
    calc_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to the selected calculator. Default is {}.
    write_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ase.io.write when saving results. Default is {}.
    log_file : Optional[str]
        Name of log file to write logs to. Default is "singlepoint.log".
    """
    read_kwargs = read_kwargs.value if read_kwargs else {}
    calc_kwargs = calc_kwargs.value if calc_kwargs else {}
    write_kwargs = write_kwargs.value if write_kwargs else {}

    if not isinstance(read_kwargs, dict):
        raise ValueError("read_kwargs must be a dictionary")
    if not isinstance(calc_kwargs, dict):
        raise ValueError("calc_kwargs must be a dictionary")
    if not isinstance(write_kwargs, dict):
        raise ValueError("write_kwargs must be a dictionary")

    s_point = SinglePoint(
        struct_path=struct_path,
        architecture=architecture,
        device=device,
        read_kwargs=read_kwargs,
        calc_kwargs=calc_kwargs,
        log_file=log_file,
    )
    s_point.run_single_point(
        properties=properties, write_results=True, write_kwargs=write_kwargs
    )


@app.command()
def test(name: str):
    """
    Dummy alternative CLI command.

    Parameters
    ----------
    name : str
        Name of person.
    """
    print(f"Hello, {name}!")
