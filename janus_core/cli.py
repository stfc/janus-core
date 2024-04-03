"""Set up commandline interface."""

import ast
import datetime
import logging
from pathlib import Path
from typing import Annotated, Optional, get_args

from ase import Atoms
import typer
import yaml

from janus_core.geom_opt import optimize
from janus_core.janus_types import Ensembles
from janus_core.md import NPH, NPT, NVE, NVT, NVT_NH
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


def _parse_dict_class(value: str):
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


def _parse_typer_dicts(typer_dicts: list[TyperDict]) -> list[dict]:
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


def _dict_paths_to_strs(dictionary: dict) -> None:
    """
    Recursively iterate over dictionary, converting Path values to strings.

    Parameters
    ----------
    dictionary : dict
        Dictionary to be converted.
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            _dict_paths_to_strs(value)
        elif isinstance(value, Path):
            dictionary[key] = str(value)


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
        parser=_parse_dict_class,
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
        parser=_parse_dict_class,
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
        parser=_parse_dict_class,
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
Summary = Annotated[
    Path, typer.Option(help="Path to save summary of inputs and start/end time.")
]


@app.command(help="Perform single point calculations and save to file.")
def singlepoint(  # pylint: disable=too-many-locals
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
    out_file: Annotated[
        Path,
        typer.Option(
            "--out",
            help=(
                "Path to save structure with calculated results. Default is inferred "
                "from name of structure file."
            ),
        ),
    ] = None,
    read_kwargs: ReadKwargs = None,
    calc_kwargs: CalcKwargs = None,
    write_kwargs: WriteKwargs = None,
    log_file: LogFile = "singlepoint.log",
    summary: Summary = "singlepoint_summary.yml",
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
    out_file : Optional[Path]
        Path to save structure with calculated results. Default is inferred from name
        of the structure file.
    read_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ase.io.read. Default is {}.
    calc_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to the selected calculator. Default is {}.
    write_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ase.io.write when saving results. Default is {}.
    log_file : Optional[Path]
        Path to write logs to. Default is "singlepoint.log".
    summary : Path
        Path to save summary of inputs and start/end time. Default is
        singlepoint_summary.yml.
    """
    [read_kwargs, calc_kwargs, write_kwargs] = _parse_typer_dicts(
        [read_kwargs, calc_kwargs, write_kwargs]
    )

    # Check filename for results not duplicated
    if "filename" in write_kwargs:
        raise ValueError("'filename' must be passed through the --out option")

    # Default filname for saving results determined in SinglePoint if not specified
    if out_file:
        write_kwargs["filename"] = out_file

    singlepoint_kwargs = {
        "struct_path": struct_path,
        "architecture": architecture,
        "device": device,
        "read_kwargs": read_kwargs,
        "calc_kwargs": calc_kwargs,
        "log_kwargs": {"filename": log_file, "filemode": "w"},
    }

    # Initialise singlepoint structure and calculator
    s_point = SinglePoint(**singlepoint_kwargs)

    # Store inputs for yaml summary
    inputs = singlepoint_kwargs.copy()

    # Store only filename as filemode is not set by user
    del inputs["log_kwargs"]
    inputs["log"] = log_file

    if isinstance(s_point.struct, Atoms):
        inputs["struct"] = {
            "n_atoms": len(s_point.struct),
            "struct_path": struct_path,
            "struct_name": s_point.struct_name,
            "formula": s_point.struct.get_chemical_formula(),
        }
    else:
        inputs["traj"] = {
            "length": len(s_point.struct),
            "struct_path": struct_path,
            "struct_name": s_point.struct_name,
            "struct": {
                "n_atoms": len(s_point.struct[0]),
                "formula": s_point.struct[0].get_chemical_formula(),
            },
        }

    inputs["run"] = {
        "properties": properties,
        "write_kwargs": write_kwargs,
    }

    # Convert all paths to strings in inputs nested dictionary
    _dict_paths_to_strs(inputs)

    # Save summary information before singlepoint calculation begins
    save_info = [
        {"command": "janus singlepoint"},
        {"start_time": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")},
        {"inputs": inputs},
    ]
    with open(summary, "w", encoding="utf8") as outfile:
        yaml.dump(save_info, outfile, default_flow_style=False)

    # Run singlepoint calculation
    s_point.run(properties=properties, write_results=True, write_kwargs=write_kwargs)

    # Save time after simulation has finished
    with open(summary, "a", encoding="utf8") as outfile:
        yaml.dump(
            [{"end_time": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")}],
            outfile,
            default_flow_style=False,
        )
    logging.shutdown()


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
    out_file: Annotated[
        Path,
        typer.Option(
            "--out",
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
            parser=_parse_dict_class,
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
    summary: Summary = "geomopt_summary.yml",
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
    out_file : Optional[Path]
        Path to save optimized structure, or last structure if optimization did not
        converge. Default is inferred from name of structure file.
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
    summary : Path
        Path to save summary of inputs and start/end time. Default is
        geomopt_summary.yml.
    """
    [read_kwargs, calc_kwargs, opt_kwargs, write_kwargs] = _parse_typer_dicts(
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
        raise ValueError("'filename' must be passed through the --out option")

    # Check trajectory path not duplicated
    if "trajectory" in opt_kwargs:
        raise ValueError("'trajectory' must be passed through the --traj option")

    # Set default filname for writing optimized structure if not specified
    if out_file:
        write_kwargs["filename"] = out_file
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

    # Dictionary of inputs for optimize function
    optimize_kwargs = {
        "struct": s_point.struct,
        "fmax": fmax,
        "steps": steps,
        "filter_kwargs": filter_kwargs,
        **fully_opt_dict,
        "opt_kwargs": opt_kwargs,
        "write_results": True,
        "write_kwargs": write_kwargs,
        "traj_kwargs": traj_kwargs,
        "log_kwargs": {"filename": log_file, "filemode": "a"},
    }

    # Store inputs for yaml summary
    inputs = optimize_kwargs.copy()

    # Store only filename as filemode is not set by user
    del inputs["log_kwargs"]
    inputs["log"] = log_file

    inputs["struct"] = {
        "n_atoms": len(s_point.struct),
        "struct_path": struct_path,
        "struct_name": s_point.struct_name,
        "formula": s_point.struct.get_chemical_formula(),
    }

    inputs["calc"] = {
        "architecture": architecture,
        "device": device,
        "read_kwargs": read_kwargs,
        "calc_kwargs": calc_kwargs,
    }

    # Convert all paths to strings in inputs nested dictionary
    _dict_paths_to_strs(inputs)

    # Save summary information before optimization begins
    save_info = [
        {"command": "janus geomopt"},
        {"start_time": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")},
        {"inputs": inputs},
    ]
    with open(summary, "w", encoding="utf8") as outfile:
        yaml.dump(save_info, outfile, default_flow_style=False)

    # Run geometry optimization and save output structure
    optimize(**optimize_kwargs)

    # Time after optimization has finished
    with open(summary, "a", encoding="utf8") as outfile:
        yaml.dump(
            [{"end_time": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")}],
            outfile,
            default_flow_style=False,
        )
    logging.shutdown()


@app.command(
    help="Run molecular dynamics simulation, and save trajectory and statistics.",
)
def md(  # pylint: disable=too-many-arguments,too-many-locals,invalid-name
    ensemble: Annotated[str, typer.Option(help="Name of thermodynamic ensemble.")],
    struct_path: StructPath,
    steps: Annotated[int, typer.Option(help="Number of steps in simulation.")] = 0,
    timestep: Annotated[
        float, typer.Option(help="Timestep for integrator, in fs.")
    ] = 1.0,
    temp: Annotated[float, typer.Option(help="Temperature, in K.")] = 300.0,
    thermostat_time: Annotated[
        float,
        typer.Option(
            help="Thermostat time for NPT, NVT Nosé-Hoover, or NPH simulation, in fs."
        ),
    ] = 50.0,
    barostat_time: Annotated[
        float, typer.Option(help="Barostat time for NPT simulation, in fs.")
    ] = 75.0,
    bulk_modulus: Annotated[
        float, typer.Option(help="Bulk modulus for NPT or NPH simulation, in GPa.")
    ] = 2.0,
    pressure: Annotated[
        float, typer.Option(help="Pressure fpr NPT or NPH simulation, in bar.")
    ] = 0.0,
    friction: Annotated[
        float, typer.Option(help="Friction coefficient for NVT simulation, in fs^-1.")
    ] = 0.005,
    architecture: Architecture = "mace_mp",
    device: Device = "cpu",
    read_kwargs: ReadKwargs = None,
    calc_kwargs: CalcKwargs = None,
    equil_steps: Annotated[
        int,
        typer.Option(
            help=("Maximum number of steps at which to perform optimization and reset")
        ),
    ] = 0,
    minimize: Annotated[
        bool, typer.Option(help="Whether to minimize structure during equilibration.")
    ] = False,
    minimize_every: Annotated[
        int,
        typer.Option(
            help=(
                """
                Frequency of minimizations. Default disables minimization after
                beginning dynamics.
                """
            )
        ),
    ] = -1,
    minimize_kwargs: Annotated[
        TyperDict,
        typer.Option(
            parser=_parse_dict_class,
            help=(
                """
                Keyword arguments to pass to optimizer. Must be passed as a dictionary
                wrapped in quotes, e.g. "{'key' : value}".
                """
            ),
            metavar="DICT",
        ),
    ] = None,
    rescale_velocities: Annotated[
        bool, typer.Option(help="Whether to rescale velocities during equilibration.")
    ] = False,
    remove_rot: Annotated[
        bool, typer.Option(help="Whether to remove rotation during equilibration.")
    ] = False,
    rescale_every: Annotated[
        int, typer.Option(help="Frequency to rescale velocities during equilibration.")
    ] = 10,
    file_prefix: Annotated[
        Optional[Path],
        typer.Option(
            help=(
                """
                Prefix for output filenames. Default is inferred from structure,
                ensemble, and temperature.
                """
            ),
        ),
    ] = None,
    restart: Annotated[bool, typer.Option(help="Whether restarting dynamics.")] = False,
    restart_stem: Annotated[
        Optional[Path],
        typer.Option(
            help="Stem for restart file name. Default inferred from `file_prefix`."
        ),
    ] = None,
    restart_every: Annotated[
        int, typer.Option(help="Frequency of steps to save restart info.")
    ] = 1000,
    rotate_restart: Annotated[
        bool, typer.Option(help="Whether to rotate restart files.")
    ] = False,
    restarts_to_keep: Annotated[
        int, typer.Option(help="Restart files to keep if rotating.")
    ] = 4,
    stats_file: Annotated[
        Path,
        typer.Option(
            help=(
                """
                File to save thermodynamical statistics. Default inferred from
                `file_prefix`.
                """
            )
        ),
    ] = None,
    stats_every: Annotated[
        int, typer.Option(help="Frequency to output statistics.")
    ] = 100,
    traj_file: Annotated[
        Path,
        typer.Option(
            help="File to save trajectory. Default inferred from `file_prefix`."
        ),
    ] = None,
    traj_append: Annotated[
        bool, typer.Option(help="Whether to append trajectory.")
    ] = False,
    traj_start: Annotated[
        int, typer.Option(help="Step to start saving trajectory.")
    ] = 0,
    traj_every: Annotated[
        int, typer.Option(help="Frequency of steps to save trajectory.")
    ] = 100,
    log_file: LogFile = "md.log",
    seed: Annotated[
        Optional[int],
        typer.Option(help="Random seed for numpy.random and random functions."),
    ] = None,
    summary: Summary = "md_summary.yml",
):
    """
    Run molecular dynamics simulation, and save trajectory and statistics.

    Parameters
    ----------
    ensemble : str
        Name of thermodynamic ensemble.
    struct_path : Path
        Path of structure to simulate.
    steps : int
        Number of steps in simulation. Default is 0.
    timestep : float
        Timestep for integrator, in fs. Default is 1.0.
    temp : float
        Temperature, in K. Default is 300.
    thermostat_time : float
        Thermostat time, in fs. Default is 50.0.
    barostat_time : float
        Barostat time, in fs. Default is 75.0.
    bulk_modulus : float
        Bulk modulus, in GPa. Default is 2.0.
    pressure : float
        Pressure, in bar. Default is 0.0.
    friction : float
        Friction coefficient in fs^-1. Default is 0.005.
    architecture : Optional[str]
        MLIP architecture to use for molecular dynamics.
        Default is "mace_mp".
    device : Optional[str]
        Device to run model on. Default is "cpu".
    read_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ase.io.read. Default is {}.
    calc_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to the selected calculator. Default is {}.
    equil_steps : int
        Maximum number of steps at which to perform optimization and reset velocities.
        Default is 0.
    minimize : bool
        Whether to minimize structure during equilibration. Default is False.
    minimize_every : int
        Frequency of minimizations. Default is -1, which disables minimization after
        beginning dynamics.
    minimize_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to geometry optimizer. Default is {}.
    rescale_velocities : bool
        Whether to rescale velocities. Default is False.
    remove_rot : bool
        Whether to remove rotation. Default is False.
    rescale_every : int
        Frequency to rescale velocities. Default is 10.
    file_prefix : Optional[PathLike]
        Prefix for output filenames. Default is inferred from structure, ensemble,
        and temperature.
    restart : bool
        Whether restarting dynamics. Default is False.
    restart_stem : str
        Stem for restart file name. Default inferred from `file_prefix`.
    restart_every : int
        Frequency of steps to save restart info. Default is 1000.
    rotate_restart : bool
        Whether to rotate restart files. Default is False.
    restarts_to_keep : int
        Restart files to keep if rotating. Default is 4.
    stats_file : Optional[PathLike]
        File to save thermodynamical statistics. Default inferred from `file_prefix`.
    stats_every : int
        Frequency to output statistics. Default is 100.
    traj_file : Optional[PathLike]
        Trajectory file to save. Default inferred from `file_prefix`.
    traj_append : bool
        Whether to append trajectory. Default is False.
    traj_start : int
        Step to start saving trajectory. Default is 0.
    traj_every : int
        Frequency of steps to save trajectory. Default is 100.
    log_file : Optional[Path]
        Path to write logs to. Default is "md.log".
    seed : Optional[int]
        Random seed used by numpy.random and random functions, such as in Langevin.
        Default is None.
    summary : Path
        Path to save summary of inputs and start/end time. Default is md_summary.yml.
    """
    [read_kwargs, calc_kwargs, minimize_kwargs] = _parse_typer_dicts(
        [read_kwargs, calc_kwargs, minimize_kwargs]
    )

    if not ensemble in get_args(Ensembles):
        raise ValueError(f"ensemble must be in {get_args(Ensembles)}")

    # Set up single point calculator
    s_point = SinglePoint(
        struct_path=struct_path,
        architecture=architecture,
        device=device,
        read_kwargs=read_kwargs,
        calc_kwargs=calc_kwargs,
        log_kwargs={"filename": log_file, "filemode": "w"},
    )

    log_kwargs = {"filename": log_file, "filemode": "a"}

    dyn_kwargs = {
        "struct": s_point.struct,
        "struct_name": s_point.struct_name,
        "timestep": timestep,
        "steps": steps,
        "temp": temp,
        "thermostat_time": thermostat_time,
        "barostat_time": barostat_time,
        "bulk_modulus": bulk_modulus,
        "pressure": pressure,
        "friction": friction,
        "equil_steps": equil_steps,
        "minimize": minimize,
        "minimize_every": minimize_every,
        "minimize_kwargs": minimize_kwargs,
        "rescale_velocities": rescale_velocities,
        "remove_rot": remove_rot,
        "rescale_every": rescale_every,
        "file_prefix": file_prefix,
        "restart": restart,
        "restart_stem": restart_stem,
        "restart_every": restart_every,
        "rotate_restart": rotate_restart,
        "restarts_to_keep": restarts_to_keep,
        "stats_file": stats_file,
        "stats_every": stats_every,
        "traj_file": traj_file,
        "traj_append": traj_append,
        "traj_start": traj_start,
        "traj_every": traj_every,
        "log_kwargs": log_kwargs,
        "seed": seed,
    }

    # Instantiate MD ensemble
    if ensemble == "nvt":
        for key in ["thermostat_time", "barostat_time", "bulk_modulus", "pressure"]:
            del dyn_kwargs[key]
        dyn = NVT(**dyn_kwargs)

    if ensemble == "npt":
        del dyn_kwargs["friction"]
        dyn = NPT(**dyn_kwargs)

    if ensemble == "nph":
        for key in ["friction", "barostat_time"]:
            del dyn_kwargs[key]
        dyn = NPH(**dyn_kwargs)

    if ensemble == "nve":
        for key in [
            "thermostat_time",
            "barostat_time",
            "bulk_modulus",
            "pressure",
            "friction",
        ]:
            del dyn_kwargs[key]
        dyn = NVE(**dyn_kwargs)

    if ensemble == "nvt-nh":
        for key in ["barostat_time", "bulk_modulus", "pressure", "friction"]:
            del dyn_kwargs[key]
        dyn = NVT_NH(**dyn_kwargs)

    # Store inputs for yaml summary
    inputs = dyn_kwargs | {"ensemble": ensemble}

    # Store only filename as filemode is not set by user
    del inputs["log_kwargs"]
    inputs["log"] = log_file

    inputs["struct"] = {
        "n_atoms": len(s_point.struct),
        "struct_path": struct_path,
        "struct_name": s_point.struct_name,
        "formula": s_point.struct.get_chemical_formula(),
    }

    inputs["calc"] = {
        "architecture": architecture,
        "device": device,
        "read_kwargs": read_kwargs,
        "calc_kwargs": calc_kwargs,
    }

    # Convert all paths to strings in inputs nested dictionary
    _dict_paths_to_strs(inputs)

    # Save summary information before simulation begins
    save_info = [
        {"command": "janus md"},
        {"start_time": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")},
        {"inputs": inputs},
    ]
    with open(summary, "w", encoding="utf8") as outfile:
        yaml.dump(save_info, outfile, default_flow_style=False)

    # Run molecular dynamics
    dyn.run()

    # Save time after simulation has finished
    with open(summary, "a", encoding="utf8") as outfile:
        yaml.dump(
            [{"end_time": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")}],
            outfile,
            default_flow_style=False,
        )
    logging.shutdown()
