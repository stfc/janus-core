"""Set up md commandline interface."""

from pathlib import Path
from typing import Annotated, Optional, get_args

from typer import Context, Option, Typer
from typer_config import use_config

from janus_core.calculations.md import NPH, NPT, NVE, NVT, NVT_NH
from janus_core.calculations.single_point import SinglePoint
from janus_core.cli.types import (
    Architecture,
    CalcKwargs,
    Device,
    EnsembleKwargs,
    LogPath,
    MinimizeKwargs,
    ModelPath,
    PostProcessKwargs,
    ReadKwargs,
    StructPath,
    Summary,
    WriteKwargs,
)
from janus_core.cli.utils import (
    check_config,
    end_summary,
    parse_typer_dicts,
    save_struct_calc,
    start_summary,
    yaml_converter_callback,
)
from janus_core.helpers.janus_types import Ensembles
from janus_core.helpers.utils import dict_paths_to_strs

app = Typer()


@app.command()
@use_config(yaml_converter_callback)
def md(
    # pylint: disable=too-many-arguments,too-many-locals,invalid-name,duplicate-code
    # numpydoc ignore=PR02
    ctx: Context,
    ensemble: Annotated[str, Option(help="Name of thermodynamic ensemble.")],
    struct: StructPath,
    struct_name: Annotated[
        str,
        Option(
            help=(
                """
                Name of structure to simulate. Default is inferred from filepath or
                chemical formula.
                """
            )
        ),
    ] = None,
    steps: Annotated[int, Option(help="Number of steps in simulation.")] = 0,
    timestep: Annotated[float, Option(help="Timestep for integrator, in fs.")] = 1.0,
    temp: Annotated[float, Option(help="Temperature, in K.")] = 300.0,
    thermostat_time: Annotated[
        float,
        Option(
            help="Thermostat time for NPT, NVT Nosé-Hoover, or NPH simulation, in fs."
        ),
    ] = 50.0,
    barostat_time: Annotated[
        float, Option(help="Barostat time for NPT simulation, in fs.")
    ] = 75.0,
    bulk_modulus: Annotated[
        float, Option(help="Bulk modulus for NPT or NPH simulation, in GPa.")
    ] = 2.0,
    pressure: Annotated[
        float, Option(help="Pressure fpr NPT or NPH simulation, in GPa.")
    ] = 0.0,
    friction: Annotated[
        float, Option(help="Friction coefficient for NVT simulation, in fs^-1.")
    ] = 0.005,
    ensemble_kwargs: EnsembleKwargs = None,
    arch: Architecture = "mace_mp",
    device: Device = "cpu",
    model_path: ModelPath = None,
    read_kwargs: ReadKwargs = None,
    calc_kwargs: CalcKwargs = None,
    equil_steps: Annotated[
        int,
        Option(
            help=("Maximum number of steps at which to perform optimization and reset")
        ),
    ] = 0,
    minimize: Annotated[
        bool, Option(help="Whether to minimize structure during equilibration.")
    ] = False,
    minimize_every: Annotated[
        int,
        Option(
            help=(
                """
                Frequency of minimizations. Default disables minimization after
                beginning dynamics.
                """
            )
        ),
    ] = -1,
    minimize_kwargs: MinimizeKwargs = None,
    rescale_velocities: Annotated[
        bool, Option(help="Whether to rescale velocities during equilibration.")
    ] = False,
    remove_rot: Annotated[
        bool, Option(help="Whether to remove rotation during equilibration.")
    ] = False,
    rescale_every: Annotated[
        int, Option(help="Frequency to rescale velocities during equilibration.")
    ] = 10,
    file_prefix: Annotated[
        Optional[Path],
        Option(
            help=(
                """
                Prefix for output filenames. Default is inferred from structure,
                ensemble, and temperature.
                """
            ),
        ),
    ] = None,
    restart: Annotated[bool, Option(help="Whether restarting dynamics.")] = False,
    restart_stem: Annotated[
        Optional[Path],
        Option(help="Stem for restart file name. Default inferred from `file_prefix`."),
    ] = None,
    restart_every: Annotated[
        int, Option(help="Frequency of steps to save restart info.")
    ] = 1000,
    rotate_restart: Annotated[
        bool, Option(help="Whether to rotate restart files.")
    ] = False,
    restarts_to_keep: Annotated[
        int, Option(help="Restart files to keep if rotating.")
    ] = 4,
    final_file: Annotated[
        Path,
        Option(
            help=(
                """
                File to save final configuration at each temperature of similation.
                Default inferred from `file_prefix`.
                """
            )
        ),
    ] = None,
    stats_file: Annotated[
        Path,
        Option(
            help=(
                """
                File to save thermodynamical statistics. Default inferred from
                `file_prefix`.
                """
            )
        ),
    ] = None,
    stats_every: Annotated[int, Option(help="Frequency to output statistics.")] = 100,
    traj_file: Annotated[
        Path,
        Option(help="File to save trajectory. Default inferred from `file_prefix`."),
    ] = None,
    traj_append: Annotated[bool, Option(help="Whether to append trajectory.")] = False,
    traj_start: Annotated[int, Option(help="Step to start saving trajectory.")] = 0,
    traj_every: Annotated[
        int, Option(help="Frequency of steps to save trajectory.")
    ] = 100,
    temp_start: Annotated[
        Optional[float],
        Option(help="Temperature to start heating, in K.  [default: None]"),
    ] = None,
    temp_end: Annotated[
        Optional[float],
        Option(help="Maximum temperature for heating, in K.  [default: None]"),
    ] = None,
    temp_step: Annotated[
        float, Option(help="Size of temperature steps when heating, in K.")
    ] = None,
    temp_time: Annotated[
        float, Option(help="Time between heating steps, in fs.")
    ] = None,
    write_kwargs: WriteKwargs = None,
    post_process_kwargs: PostProcessKwargs = None,
    log: LogPath = "md.log",
    seed: Annotated[
        Optional[int],
        Option(help="Random seed for numpy.random and random functions."),
    ] = None,
    summary: Summary = "md_summary.yml",
):
    """
    Run molecular dynamics simulation, and save trajectory and statistics.

    Parameters
    ----------
    ctx : Context
        Typer (Click) Context. Automatically set.
    ensemble : str
        Name of thermodynamic ensemble.
    struct : Path
        Path of structure to simulate.
    struct_name : Optional[str]
        Name of structure to simulate. Default is inferred from filepath or chemical
        formula.
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
        Pressure, in GPa. Default is 0.0.
    friction : float
        Friction coefficient in fs^-1. Default is 0.005.
    ensemble_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ensemble initialization. Default is {}.
    arch : Optional[str]
        MLIP architecture to use for molecular dynamics.
        Default is "mace_mp".
    device : Optional[str]
        Device to run model on. Default is "cpu".
    model_path : Optional[str]
        Path to MLIP model. Default is `None`.
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
    final_file : Optional[PathLike]
        File to save final configuration at each temperature of similation. Default
        inferred from `file_prefix`.
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
    temp_start : Optional[float]
        Temperature to start heating, in K. Default is None, which disables
        heating.
    temp_end : Optional[float]
        Maximum temperature for heating, in K. Default is None, which disables
        heating.
    temp_step : Optional[float]
        Size of temperature steps when heating, in K. Default is None, which disables
        heating.
    temp_time : Optional[float]
        Time between heating steps, in fs. Default is None, which disables
        heating.
    write_kwargs : Optional[dict[str, Any]],
        Keyword arguments to pass to `output_structs` when saving trajectory and final
        files. Default is {}.
    post_process_kwargs : Optional[PostProcessKwargs]
        Kwargs to pass to post-processing.
    log : Optional[Path]
        Path to write logs to. Default is "md.log".
    seed : Optional[int]
        Random seed used by numpy.random and random functions, such as in Langevin.
        Default is None.
    summary : Path
        Path to save summary of inputs and start/end time. Default is md_summary.yml.
    config : Path
        Path to yaml configuration file to define the above options. Default is None.
    """
    # Check options from configuration file are all valid
    check_config(ctx)

    [
        read_kwargs,
        calc_kwargs,
        minimize_kwargs,
        ensemble_kwargs,
        write_kwargs,
        post_process_kwargs,
    ] = parse_typer_dicts(
        [
            read_kwargs,
            calc_kwargs,
            minimize_kwargs,
            ensemble_kwargs,
            write_kwargs,
            post_process_kwargs,
        ]
    )

    if not ensemble in get_args(Ensembles):
        raise ValueError(f"ensemble must be in {get_args(Ensembles)}")

    # Set up single point calculator
    s_point = SinglePoint(
        struct_path=struct,
        struct_name=struct_name,
        architecture=arch,
        device=device,
        model_path=model_path,
        read_kwargs=read_kwargs,
        calc_kwargs=calc_kwargs,
        log_kwargs={"filename": log, "filemode": "w"},
    )

    log_kwargs = {"filename": log, "filemode": "a"}

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
        "final_file": final_file,
        "stats_file": stats_file,
        "stats_every": stats_every,
        "traj_file": traj_file,
        "traj_append": traj_append,
        "traj_start": traj_start,
        "traj_every": traj_every,
        "temp_start": temp_start,
        "temp_end": temp_end,
        "temp_step": temp_step,
        "temp_time": temp_time,
        "write_kwargs": write_kwargs,
        "post_process_kwargs": post_process_kwargs,
        "log_kwargs": log_kwargs,
        "seed": seed,
        "ensemble_kwargs": ensemble_kwargs,
    }

    # Instantiate MD ensemble
    if ensemble == "nvt":
        for key in ["thermostat_time", "barostat_time", "bulk_modulus", "pressure"]:
            del dyn_kwargs[key]
        dyn = NVT(**dyn_kwargs)
    elif ensemble == "npt":
        del dyn_kwargs["friction"]
        dyn = NPT(**dyn_kwargs)
    elif ensemble == "nph":
        for key in ["friction", "barostat_time"]:
            del dyn_kwargs[key]
        dyn = NPH(**dyn_kwargs)
    elif ensemble == "nve":
        for key in [
            "thermostat_time",
            "barostat_time",
            "bulk_modulus",
            "pressure",
            "friction",
        ]:
            del dyn_kwargs[key]
        dyn = NVE(**dyn_kwargs)
    elif ensemble == "nvt-nh":
        for key in ["barostat_time", "bulk_modulus", "pressure", "friction"]:
            del dyn_kwargs[key]
        dyn = NVT_NH(**dyn_kwargs)
    else:
        raise ValueError(f"Unsupported Ensemble ({ensemble})")
    # Store inputs for yaml summary
    inputs = dyn_kwargs | {"ensemble": ensemble}

    # Store only filename as filemode is not set by user
    del inputs["log_kwargs"]
    inputs["log"] = log

    save_struct_calc(
        inputs, s_point, arch, device, model_path, read_kwargs, calc_kwargs
    )

    # Convert all paths to strings in inputs nested dictionary
    dict_paths_to_strs(inputs)

    # Save summary information before simulation begins
    start_summary(command="md", summary=summary, inputs=inputs)

    # Run molecular dynamics
    dyn.run()

    # Save time after simulation has finished
    end_summary(summary=summary)
