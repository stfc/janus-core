"""Set up commandline interface."""

import ast
from pathlib import Path
from typing import Annotated

import typer

from janus_core.geom_opt import optimize
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


# Shared type aliases
StructPath = Annotated[
    Path, typer.Option("--struct", help="Path of structure to simulate")
]
Architecture = Annotated[
    str, typer.Option("--arch", help="MLIP architecture to use for calculations")
]
Device = Annotated[str, typer.Option(help="Device to run calculations on")]
ReadKwargs = Annotated[
    TyperDict,
    typer.Option(
        parser=parse_dict_class,
        help="Keyword arguments to pass to ase.io.read  [default: {}]",
        metavar="DICT",
    ),
]
CalcKwargs = Annotated[
    TyperDict,
    typer.Option(
        parser=parse_dict_class,
        help="Keyword arguments to pass to selected calculator  [default: {}]",
        metavar="DICT",
    ),
]
WriteKwargs = Annotated[
    TyperDict,
    typer.Option(
        parser=parse_dict_class,
        help=(
            "Keyword arguments to pass to ase.io.write when saving "
            "results [default: {}]"
        ),
        metavar="DICT",
    ),
]
LogFile = Annotated[Path, typer.Option("--log", help="Path to save logs to")]


@app.command()
def singlepoint(
    struct_path: StructPath,
    architecture: Architecture = "mace_mp",
    device: Device = "cpu",
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
    read_kwargs: ReadKwargs = None,
    calc_kwargs: CalcKwargs = None,
    write_kwargs: WriteKwargs = None,
    log_file: LogFile = "singlepoint.log",
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
    log_file : Optional[Path]
        Path to write logs to. Default is "singlepoint.log".
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
        log_kwargs={"filename": log_file, "filemode": "w"},
    )
    s_point.run(properties=properties, write_results=True, write_kwargs=write_kwargs)


@app.command()
def geomopt(  # pylint: disable=too-many-arguments,too-many-locals
    struct_path: StructPath,
    fmax: Annotated[
        float, typer.Option("--max-force", help="Maximum force for convergence")
    ] = 0.1,
    architecture: Architecture = "mace_mp",
    device: Device = "cpu",
    fully_opt: Annotated[
        bool,
        typer.Option(
            "--fully-opt",
            help="Fully optimize the cell vectors, angles, and atomic positions",
        ),
    ] = False,
    vectors_only: Annotated[
        bool,
        typer.Option("--vectors-only", help="Allow only cell vectors to change"),
    ] = False,
    read_kwargs: ReadKwargs = None,
    calc_kwargs: CalcKwargs = None,
    write_kwargs: WriteKwargs = None,
    traj_file: Annotated[
        str, typer.Option("--traj", help="Path to save optimization frames to")
    ] = None,
    log_file: LogFile = "geomopt.log",
):
    """
    Perform geometry optimization and save optimized structure to file.

    Parameters
    ----------
    struct_path : Path
        Path of structure to simulate.
    fmax : float
        Set force convergence criteria for optimizer in units eV/Å.
        Default is 0.1.
    architecture : Optional[str]
        MLIP architecture to use for geometry optimization.
        Default is "mace_mp".
    device : Optional[str]
        Device to run model on. Default is "cpu".
    fully_opt : bool
        Whether to optimize the cell as well as atomic positions. Default is False.
    vectors_only : bool
        Whether to allow only hydrostatic deformations. Default is False.
    read_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ase.io.read. Default is {}.
    calc_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to the selected calculator. Default is {}.
    write_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ase.io.write when saving results. Default is {}.
    traj_file : Optional[str]
        Path to save optimization trajectory to. Default is None.
    log_file : Optional[Path]
        Path to write logs to. Default is "geomopt.log".
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

    if not fully_opt and vectors_only:
        raise ValueError("--vectors-only requires --fully-opt to be set")

    # Set up single point calculator
    s_point = SinglePoint(
        struct_path=struct_path,
        architecture=architecture,
        device=device,
        read_kwargs=read_kwargs,
        calc_kwargs=calc_kwargs,
        log_kwargs={"filename": log_file, "filemode": "w"},
    )

    opt_kwargs = {"trajectory": traj_file} if traj_file else None
    traj_kwargs = {"filename": traj_file} if traj_file else None
    filter_kwargs = {"hydrostatic_strain": vectors_only} if fully_opt else None

    # Use default filter if passed --fully-opt, otherwise override with None
    fully_opt_dict = {} if fully_opt else {"filter_func": None}

    # Run geometry optimization and save output structure
    optimize(
        s_point.struct,
        fmax=fmax,
        filter_kwargs=filter_kwargs,
        **fully_opt_dict,
        opt_kwargs=opt_kwargs,
        write_results=True,
        write_kwargs=write_kwargs,
        traj_kwargs=traj_kwargs,
        log_kwargs={"filename": log_file, "filemode": "a"},
    )
