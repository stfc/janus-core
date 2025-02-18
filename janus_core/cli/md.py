"""Set up md commandline interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, get_args

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
            help=(
                """
                Thermostat time for NPT, NPT-MTK or NVT Nosé-Hoover simulation,
                in fs. Default is 50 fs for NPT and NVT Nosé-Hoover, or 100 fs for
                NPT-MTK.
                """
            )
        ),
    ] = None,
    barostat_time: Annotated[
        float,
        Option(
            help=(
                """
                Barostat time for NPT, NPT-MTK or NPH simulation, in fs.
                Default is 75 fs for NPT and NPH, or 1000 fs for NPT-MTK.
                """
            )
        ),
    ] = None,
    bulk_modulus: Annotated[
        float, Option(help="Bulk modulus for NPT or NPH simulation, in GPa.")
    ] = 2.0,
    pressure: Annotated[
        float, Option(help="Pressure for NPT or NPH simulation, in GPa.")
    ] = 0.0,
    friction: Annotated[
        float, Option(help="Friction coefficient for NVT simulation, in fs^-1.")
    ] = 0.005,
    taut: Annotated[
        float,
        Option(
            help="Temperature coupling time constant for NVT CSVR simulation, in fs."
        ),
    ] = 100.0,
    thermostat_chain: Annotated[
        int,
        Option(help="Number of variables in thermostat chain for NPT MTK simulation."),
    ] = 3,
    barostat_chain: Annotated[
        int,
        Option(help="Number of variables in barostat chain for NPT MTK simulation."),
    ] = 3,
    thermostat_substeps: Annotated[
        int,
        Option(
            help="Number of sub-steps in thermostat integration for NPT MTK simulation."
        ),
    ] = 1,
    barostat_substeps: Annotated[
        int,
        Option(
            help="Number of sub-steps in barostat integration for NPT MTK simulation."
        ),
    ] = 1,
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
        Path | None,
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
        Path | None,
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
        Path | None,
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
        Path | None,
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
        Path | None,
        Option(help="File to save trajectory. Default inferred from `file_prefix`."),
    ] = None,
    traj_append: Annotated[bool, Option(help="Whether to append trajectory.")] = False,
    traj_start: Annotated[int, Option(help="Step to start saving trajectory.")] = 0,
    traj_every: Annotated[
        int, Option(help="Frequency of steps to save trajectory.")
    ] = 100,
    temp_start: Annotated[
        float | None,
        Option(help="Temperature to start heating, in K."),
    ] = None,
    temp_end: Annotated[
        float | None,
        Option(help="Maximum temperature for heating, in K."),
    ] = None,
    temp_step: Annotated[
        float | None, Option(help="Size of temperature steps when heating, in K.")
    ] = None,
    temp_time: Annotated[
        float | None, Option(help="Time between heating steps, in fs.")
    ] = None,
    write_kwargs: WriteKwargs = None,
    post_process_kwargs: PostProcessKwargs = None,
    seed: Annotated[
        int | None,
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
    ctx
        Typer (Click) Context. Automatically set.
    ensemble
        Name of thermodynamic ensemble.
    struct
        Path of structure to simulate.
    steps
        Number of steps in simulation. Default is 0.
    timestep
        Timestep for integrator, in fs. Default is 1.0.
    temp
        Temperature, in K. Default is 300.
    thermostat_time
        Thermostat time for NPT, NPT-MTK or NVT Nosé-Hoover simulation,
        in fs. Default is 50 fs for NPT and NVT Nosé-Hoover, or 100 fs for NPT-MTK.
    barostat_time
        Barostat time for NPT, NPT-MTK or NPH simulation, in fs.
        Default is 75 fs for NPT and NPH, or 1000 fs for NPT-MTK.
    bulk_modulus
        Bulk modulus, in GPa. Default is 2.0.
    pressure
        Pressure, in GPa. Default is 0.0.
    friction
        Friction coefficient in fs^-1. Default is 0.005.
    taut
        Time constant for CSVR thermostat coupling, in fs. Default is 100.0.
    thermostat_chain
        Number of variables in thermostat chain for NPT MTK simulation. Default is 3.
    barostat_chain
        Number of variables in barostat chain for NPT MTK simulation. Default is 3.
    thermostat_substeps
        Number of sub-steps in thermostat integration for NPT MTK simulation.
        Default is 1.
    barostat_substeps
        Number of sub-steps in barostat integration for NPT MTK simulation.
        Default is 1.
    ensemble_kwargs
        Keyword arguments to pass to ensemble initialization. Default is {}.
    arch
        MLIP architecture to use for molecular dynamics.
        Default is "mace_mp".
    device
        Device to run model on. Default is "cpu".
    model_path
        Path to MLIP model. Default is `None`.
    read_kwargs
        Keyword arguments to pass to ase.io.read. By default,
            read_kwargs["index"] is -1.
    calc_kwargs
        Keyword arguments to pass to the selected calculator. Default is {}.
    equil_steps
        Maximum number of steps at which to perform optimization and reset velocities.
        Default is 0.
    minimize
        Whether to minimize structure during equilibration. Default is False.
    minimize_every
        Frequency of minimizations. Default is -1, which disables minimization after
        beginning dynamics.
    minimize_kwargs
        Keyword arguments to pass to geometry optimizer. Default is {}.
    rescale_velocities
        Whether to rescale velocities. Default is False.
    remove_rot
        Whether to remove rotation. Default is False.
    rescale_every
        Frequency to rescale velocities. Default is 10.
    file_prefix
        Prefix for output filenames. Default is inferred from structure, ensemble,
        and temperature.
    restart
        Whether restarting dynamics. Default is False.
    restart_auto
        Whether to infer restart file name if restarting dynamics. Default is True.
    restart_stem
        Stem for restart file name. Default inferred from `file_prefix`.
    restart_every
        Frequency of steps to save restart info. Default is 1000.
    rotate_restart
        Whether to rotate restart files. Default is False.
    restarts_to_keep
        Restart files to keep if rotating. Default is 4.
    final_file
        File to save final configuration at each temperature of similation. Default
        inferred from `file_prefix`.
    stats_file
        File to save thermodynamical statistics. Default inferred from `file_prefix`.
    stats_every
        Frequency to output statistics. Default is 100.
    traj_file
        Trajectory file to save. Default inferred from `file_prefix`.
    traj_append
        Whether to append trajectory. Default is False.
    traj_start
        Step to start saving trajectory. Default is 0.
    traj_every
        Frequency of steps to save trajectory. Default is 100.
    temp_start
        Temperature to start heating, in K. Default is None, which disables
        heating.
    temp_end
        Maximum temperature for heating, in K. Default is None, which disables
        heating.
    temp_step
        Size of temperature steps when heating, in K. Default is None, which disables
        heating.
    temp_time
        Time between heating steps, in fs. Default is None, which disables
        heating.
    write_kwargs
        Keyword arguments to pass to `output_structs` when saving trajectory and final
        files. Default is {}.
    post_process_kwargs
        Kwargs to pass to post-processing.
    seed
        Random seed used by numpy.random and random functions, such as in Langevin.
        Default is None.
    log
        Path to write logs to. Default is inferred from the name of the structure file.
    tracker
        Whether to save carbon emissions of calculation in log file and summary.
        Default is True.
    summary
        Path to save summary of inputs, start/end time, and carbon emissions. Default
        is inferred from the name of the structure file.
    config
        Path to yaml configuration file to define the above options. Default is None.
    """
    from janus_core.calculations.md import NPH, NPT, NPT_MTK, NVE, NVT, NVT_CSVR, NVT_NH
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

    # Defaults
    if thermostat_time is None:
        if ensemble in ("npt", "nph", "nvt-nh"):
            thermostat_time = 50.0
        elif ensemble == "npt-mtk":
            thermostat_time = 100.0
    if barostat_time is None:
        if ensemble in ("npt", "nph"):
            barostat_time = 75.0
        elif ensemble == "npt-mtk":
            barostat_time = 1000.0

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
        "thermostat_chain": thermostat_chain,
        "barostat_chain": barostat_chain,
        "thermostat_substeps": thermostat_substeps,
        "barostat_substeps": barostat_substeps,
        "bulk_modulus": bulk_modulus,
        "pressure": pressure,
        "friction": friction,
        "taut": taut,
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
        for key in (
            "thermostat_time",
            "barostat_time",
            "bulk_modulus",
            "pressure",
            "taut",
            "thermostat_chain",
            "barostat_chain",
            "thermostat_substeps",
            "barostat_substeps",
        ):
            del dyn_kwargs[key]
        dyn = NVT(**dyn_kwargs)
    elif ensemble == "npt":
        for key in (
            "friction",
            "taut",
            "thermostat_chain",
            "barostat_chain",
            "thermostat_substeps",
            "barostat_substeps",
        ):
            del dyn_kwargs[key]
        dyn = NPT(**dyn_kwargs)
    elif ensemble == "nph":
        for key in (
            "friction",
            "thermostat_time",
            "taut",
            "thermostat_chain",
            "barostat_chain",
            "thermostat_substeps",
            "barostat_substeps",
        ):
            del dyn_kwargs[key]
        dyn = NPH(**dyn_kwargs)
    elif ensemble == "nve":
        for key in (
            "thermostat_time",
            "barostat_time",
            "bulk_modulus",
            "pressure",
            "friction",
            "taut",
            "thermostat_chain",
            "barostat_chain",
            "thermostat_substeps",
            "barostat_substeps",
        ):
            del dyn_kwargs[key]
        dyn = NVE(**dyn_kwargs)
    elif ensemble == "nvt-nh":
        for key in (
            "barostat_time",
            "bulk_modulus",
            "pressure",
            "friction",
            "taut",
            "thermostat_chain",
            "barostat_chain",
            "thermostat_substeps",
            "barostat_substeps",
        ):
            del dyn_kwargs[key]
        dyn = NVT_NH(**dyn_kwargs)
    elif ensemble == "nvt-csvr":
        for key in (
            "thermostat_time",
            "barostat_time",
            "bulk_modulus",
            "pressure",
            "friction",
            "thermostat_chain",
            "barostat_chain",
            "thermostat_substeps",
            "barostat_substeps",
        ):
            del dyn_kwargs[key]
        dyn = NVT_CSVR(**dyn_kwargs)
    elif ensemble == "npt-mtk":
        for key in ("bulk_modulus", "friction", "taut"):
            del dyn_kwargs[key]
        dyn = NPT_MTK(**dyn_kwargs)
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
