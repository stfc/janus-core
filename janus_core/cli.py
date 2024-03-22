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


def parse_typer_dicts(typer_dicts: list[TyperDict]) -> list[dict]:
    """
    Convert list of TyperDict objects to list of dictionaries.

    Parameters
    ----------
    typer_dicts : list[TyperDict]
        List of TyperDict objects to convert.

    Returns
    -------
    list[dict]
        List of converted dictionaries.

    Raises
    ------
    ValueError
        If items in list are not converted to dicts.
    """
    for i, typer_dict in enumerate(typer_dicts):
        typer_dicts[i] = typer_dict.value if typer_dict else {}
        if not isinstance(typer_dicts[i], dict):
            raise ValueError(f"{typer_dicts[i]} must be passed as a dictionary")
    return typer_dicts


# Shared type aliases
StructPath = Annotated[
    Path, typer.Option("--struct", help="Path of structure to simulate.")
]
Architecture = Annotated[
    str, typer.Option("--arch", help="MLIP architecture to use for calculations.")
]
Device = Annotated[str, typer.Option(help="Device to run calculations on.")]
ReadKwargs = Annotated[
    TyperDict,
    typer.Option(
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
    typer.Option(
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
    typer.Option(
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
LogFile = Annotated[Path, typer.Option("--log", help="Path to save logs to.")]


@app.command(help="Perform single point calculations and save to file.")
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
    [read_kwargs, calc_kwargs, write_kwargs] = parse_typer_dicts(
        [read_kwargs, calc_kwargs, write_kwargs]
    )

    s_point = SinglePoint(
        struct_path=struct_path,
        architecture=architecture,
        device=device,
        read_kwargs=read_kwargs,
        calc_kwargs=calc_kwargs,
        log_kwargs={"filename": log_file, "filemode": "w"},
    )
    s_point.run(properties=properties, write_results=True, write_kwargs=write_kwargs)


@app.command(
    help="Perform geometry optimization and save optimized structure to file.",
)
def geomopt(  # pylint: disable=too-many-arguments,too-many-locals
    struct_path: StructPath,
    fmax: Annotated[
        float, typer.Option("--max-force", help="Maximum force for convergence.")
    ] = 0.1,
    steps: Annotated[
        int, typer.Option("--steps", help="Maximum number of optimization steps.")
    ] = 1000,
    architecture: Architecture = "mace_mp",
    device: Device = "cpu",
    vectors_only: Annotated[
        bool,
        typer.Option(
            "--vectors-only",
            help=("Optimize cell vectors, as well as atomic positions."),
        ),
    ] = False,
    fully_opt: Annotated[
        bool,
        typer.Option(
            "--fully-opt",
            help="Fully optimize the cell vectors, angles, and atomic positions.",
        ),
    ] = False,
    opt_file: Annotated[
        Path,
        typer.Option(
            "--opt",
            help=(
                "Path to save optimized structure. Default is inferred from name "
                "of structure file."
            ),
        ),
    ] = None,
    traj_file: Annotated[
        str,
        typer.Option(
            "--traj", help="Path if saving optimization frames.  [default: None]"
        ),
    ] = None,
    read_kwargs: ReadKwargs = None,
    calc_kwargs: CalcKwargs = None,
    opt_kwargs: Annotated[
        TyperDict,
        typer.Option(
            parser=parse_dict_class,
            help=(
                """
                Keyword arguments to pass to optimizer. Must be passed as a dictionary
                wrapped in quotes, e.g. "{'key' : value}".  [default: "{}"]
                """
            ),
            metavar="DICT",
        ),
    ] = None,
    write_kwargs: WriteKwargs = None,
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
    steps : int
        Set maximum number of optimization steps to run. Default is 1000.
    architecture : Optional[str]
        MLIP architecture to use for geometry optimization.
        Default is "mace_mp".
    device : Optional[str]
        Device to run model on. Default is "cpu".
    vectors_only : bool
        Whether to optimize cell vectors, as well as atomic positions, by setting
        `hydrostatic_strain` in the filter function. Default is False.
    fully_opt : bool
        Whether to fully optimize the cell vectors, angles, and atomic positions.
        Default is False.
    opt_file : Optional[Path]
        Path to save optimized structure. Default is inferred from name of
        structure file.
    traj_file : Optional[str]
        Path if saving optimization frames. Default is None.
    read_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ase.io.read. Default is {}.
    calc_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to the selected calculator. Default is {}.
    opt_kwargs : Optional[ASEOptArgs]
        Keyword arguments to pass to optimizer. Default is {}.
    write_kwargs : Optional[ASEWriteArgs]
        Keyword arguments to pass to ase.io.write when saving optimized structure.
        Default is {}.
    log_file : Optional[Path]
        Path to write logs to. Default is "geomopt.log".
    """
    [read_kwargs, calc_kwargs, opt_kwargs, write_kwargs] = parse_typer_dicts(
        [read_kwargs, calc_kwargs, opt_kwargs, write_kwargs]
    )

    # Set up single point calculator
    s_point = SinglePoint(
        struct_path=struct_path,
        architecture=architecture,
        device=device,
        read_kwargs=read_kwargs,
        calc_kwargs=calc_kwargs,
        log_kwargs={"filename": log_file, "filemode": "w"},
    )

    # Check optimized structure path not duplicated
    if "filename" in write_kwargs:
        raise ValueError("'filename' must be passed through the --opt option")

    # Check trajectory path not duplicated
    if "trajectory" in opt_kwargs:
        raise ValueError("'trajectory' must be passed through the --traj option")

    # Set default filname for writing optimized structure if not specified
    if opt_file:
        write_kwargs["filename"] = opt_file
    else:
        write_kwargs["filename"] = f"{s_point.struct_name}-opt.xyz"

    # Set same trajectory filenames to overwrite saved binary with xyz
    opt_kwargs["trajectory"] = traj_file if traj_file else None
    traj_kwargs = {"filename": traj_file} if traj_file else None

    # Set hydrostatic_strain
    # If not passed --fully-opt or --vectors-only, will be unused
    filter_kwargs = {"hydrostatic_strain": vectors_only}

    # Use default filter if passed --fully-opt or --vectors-only
    # Otherwise override with None
    fully_opt_dict = {} if (fully_opt or vectors_only) else {"filter_func": None}

    # Run geometry optimization and save output structure
    optimize(
        s_point.struct,
        fmax=fmax,
        steps=steps,
        filter_kwargs=filter_kwargs,
        **fully_opt_dict,
        opt_kwargs=opt_kwargs,
        write_results=True,
        write_kwargs=write_kwargs,
        traj_kwargs=traj_kwargs,
        log_kwargs={"filename": log_file, "filemode": "a"},
    )
