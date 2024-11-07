# ruff: noqa: I002, FA100
"""Set up md commandline interface."""

# Issues with future annotations and typer
# c.f. https://github.com/maxb2/typer-config/issues/295
# from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional, get_args

from typer import Context, Option, Typer
from typer_config import use_config

from janus_core.cli.types import (
    Architecture,
    CalcKwargs,
    Device,
    EnsembleKwargs,
    LogPath,
    MinimizeKwargs,
    ModelPath,
    PostProcessKwargs,
    ReadKwargsLast,
    StructPath,
    Summary,
    WriteKwargs,
)
from janus_core.cli.utils import yaml_converter_callback

app = Typer()


@app.command()
@use_config(yaml_converter_callback)
def md(
    # numpydoc ignore=PR02
    ctx: Context,
    ensemble: Annotated[str, Option(help="Name of thermodynamic ensemble.")],
    struct: StructPath,
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
    read_kwargs: ReadKwargsLast = None,
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
    restart_auto: Annotated[
        bool, Option(help="Whether to infer restart file if restarting dynamics.")
    ] = True,
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
        Optional[Path],
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
        Optional[Path],
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
        Optional[Path],
        Option(help="File to save trajectory. Default inferred from `file_prefix`."),
    ] = None,
    traj_append: Annotated[bool, Option(help="Whether to append trajectory.")] = False,
    traj_start: Annotated[int, Option(help="Step to start saving trajectory.")] = 0,
    traj_every: Annotated[
        int, Option(help="Frequency of steps to save trajectory.")
    ] = 100,
    temp_start: Annotated[
        Optional[float],
        Option(help="Temperature to start heating, in K."),
    ] = None,
    temp_end: Annotated[
        Optional[float],
        Option(help="Maximum temperature for heating, in K."),
    ] = None,
    temp_step: Annotated[
        Optional[float], Option(help="Size of temperature steps when heating, in K.")
    ] = None,
    temp_time: Annotated[
        Optional[float], Option(help="Time between heating steps, in fs.")
    ] = None,
    write_kwargs: WriteKwargs = None,
    post_process_kwargs: PostProcessKwargs = None,
    seed: Annotated[
        Optional[int],
        Option(help="Random seed for numpy.random and random functions."),
    ] = None,
    log: LogPath = None,
    tracker: Annotated[
        bool, Option(help="Whether to save carbon emissions of calculation")
    ] = True,
    summary: Summary = None,
) -> None:
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
        Keyword arguments to pass to ase.io.read. By default,
            read_kwargs["index"] is -1.
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
    restart_auto : bool
        Whether to infer restart file name if restarting dynamics. Default is True.
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
    seed : Optional[int]
        Random seed used by numpy.random and random functions, such as in Langevin.
        Default is None.
    log : Optional[Path]
        Path to write logs to. Default is inferred from the name of the structure file.
    tracker : bool
        Whether to save carbon emissions of calculation in log file and summary.
        Default is True.
    summary : Optional[Path]
        Path to save summary of inputs, start/end time, and carbon emissions. Default
        is inferred from the name of the structure file.
    config : Optional[Path]
        Path to yaml configuration file to define the above options. Default is None.
    """
    from janus_core.calculations.md import NPH, NPT, NVE, NVT, NVT_NH
    from janus_core.cli.utils import (
        carbon_summary,
        check_config,
        end_summary,
        parse_typer_dicts,
        save_struct_calc,
        set_read_kwargs_index,
        start_summary,
    )
    from janus_core.helpers.janus_types import Ensembles

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

    if ensemble not in get_args(Ensembles):
        raise ValueError(f"ensemble must be in {get_args(Ensembles)}")

    # Read only first structure by default and ensure only one image is read
    set_read_kwargs_index(read_kwargs)

    log_kwargs = {"filemode": "w"}
    if log:
        log_kwargs["filename"] = log

    dyn_kwargs = {
        "struct_path": struct,
        "arch": arch,
        "device": device,
        "model_path": model_path,
        "read_kwargs": read_kwargs,
        "calc_kwargs": calc_kwargs,
        "attach_logger": True,
        "log_kwargs": log_kwargs,
        "track_carbon": tracker,
        "ensemble_kwargs": ensemble_kwargs,
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
        "restart_auto": restart_auto,
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
        "seed": seed,
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

    # Set summary and log files
    summary = dyn._build_filename(
        "md-summary.yml", dyn.param_prefix, filename=summary
    ).absolute()
    log = dyn.log_kwargs["filename"]

    # Store inputs for yaml summary
    inputs = dyn_kwargs | {"ensemble": ensemble}

    # Add structure, MLIP information, and log to inputs
    save_struct_calc(
        inputs=inputs,
        struct=dyn.struct,
        struct_path=struct,
        arch=arch,
        device=device,
        model_path=model_path,
        read_kwargs=read_kwargs,
        calc_kwargs=calc_kwargs,
        log=log,
    )

    # Save summary information before simulation begins
    start_summary(command="md", summary=summary, inputs=inputs)

    # Run molecular dynamics
    dyn.run()

    # Save carbon summary
    if tracker:
        carbon_summary(summary=summary, log=log)

    # Save time after simulation has finished
    end_summary(summary=summary)
