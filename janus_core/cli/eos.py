"""Set up eos commandline interface."""

from __future__ import annotations

from typing import Annotated, get_args

from click import Choice
from typer import Context, Option, Typer
from typer_config import use_config

from janus_core.cli.types import (
    Architecture,
    CalcKwargs,
    Device,
    FilePrefix,
    LogPath,
    MinimizeKwargs,
    Model,
    ModelPath,
    ReadKwargsLast,
    StructPath,
    Summary,
    Tracker,
    WriteKwargs,
)
from janus_core.cli.utils import yaml_converter_callback
from janus_core.helpers.janus_types import EoSNames

app = Typer()


@app.command()
@use_config(yaml_converter_callback, param_help="Path to configuration file.")
def eos(
    # numpydoc ignore=PR02
    ctx: Context,
    # Required
    arch: Architecture,
    struct: StructPath,
    # Calculation
    min_volume: Annotated[
        float,
        Option(help="Minimum volume scale factor.", rich_help_panel="Calculation"),
    ] = 0.95,
    max_volume: Annotated[
        float,
        Option(help="Maximum volume scale factor.", rich_help_panel="Calculation"),
    ] = 1.05,
    n_volumes: Annotated[
        int, Option(help="Number of volumes.", rich_help_panel="Calculation")
    ] = 7,
    eos_type: Annotated[
        str,
        Option(
            click_type=Choice(get_args(EoSNames)),
            help="Type of fit for equation of state.",
            rich_help_panel="Calculation",
        ),
    ] = "birchmurnaghan",
    minimize: Annotated[
        bool,
        Option(
            help="Whether to minimize initial structure before calculations.",
            rich_help_panel="Calculation",
        ),
    ] = True,
    minimize_all: Annotated[
        bool,
        Option(
            help="Whether to minimize all generated structures for calculations.",
            rich_help_panel="Calculation",
        ),
    ] = False,
    fmax: Annotated[
        float,
        Option(
            help="Maximum force for optimization convergence.",
            rich_help_panel="Calculation",
        ),
    ] = 0.1,
    minimize_kwargs: MinimizeKwargs = None,
    write_structures: Annotated[
        bool,
        Option(
            help="Whether to write out all genereated structures.",
            rich_help_panel="Calculation",
        ),
    ] = False,
    plot_to_file: Annotated[
        bool,
        Option(
            help="Whether to plot equation of state.",
            rich_help_panel="Calculation",
        ),
    ] = False,
    # MLIP Calculator
    device: Device = "cpu",
    model: Model = None,
    model_path: ModelPath = None,
    calc_kwargs: CalcKwargs = None,
    # Structure I/O
    file_prefix: FilePrefix = None,
    read_kwargs: ReadKwargsLast = None,
    write_kwargs: WriteKwargs = None,
    # Logging/summary
    log: LogPath = None,
    tracker: Tracker = True,
    summary: Summary = None,
) -> None:
    """
    Calculate equation of state and write out results.

    Parameters
    ----------
    ctx
        Typer (Click) Context. Automatically set.
    arch
        MLIP architecture to use for calculations.
    struct
        Path of structure to simulate.
    min_volume
        Minimum volume scale factor. Default is 0.95.
    max_volume
        Maximum volume scale factor. Default is 1.05.
    n_volumes
        Number of volumes to use. Default is 7.
    eos_type
        Type of fit for equation of state. Default is "birchmurnaghan".
    minimize
        Whether to minimize initial structure before calculations. Default is True.
    minimize_all
        Whether to optimize geometry for all generated structures. Default is False.
    fmax
        Set force convergence criteria for optimizer in units eV/Å.
        Default is 0.1.
    minimize_kwargs
        Other keyword arguments to pass to geometry optimizer. Default is {}.
    write_structures
        True to write out all genereated structures. Default is False.
    plot_to_file
        Whether to save plot equation of state to svg. Default is False.
    device
        Device to run model on. Default is "cpu".
    model
        Path to MLIP model or name of model. Default is `None`.
    model_path
        Deprecated. Please use `model`.
    calc_kwargs
        Keyword arguments to pass to the selected calculator. Default is {}.
    file_prefix
        Prefix for output files, including directories. Default directory is
        ./janus_results, and default filename prefix is inferred from the input
        stucture filename.
    read_kwargs
        Keyword arguments to pass to ase.io.read. By default,
            read_kwargs["index"] is -1.
    write_kwargs
        Keyword arguments to pass to ase.io.write to save generated structures.
        Default is {}.
    log
        Path to write logs to. Default is inferred from `file_prefix`.
    tracker
        Whether to save carbon emissions of calculation in log file and summary.
        Default is True.
    summary
        Path to save summary of inputs, start/end time, and carbon emissions. Default
        is inferred from `file_prefix`.
    config
        Path to yaml configuration file to define the above options. Default is None.
    """
    from janus_core.calculations.eos import EoS
    from janus_core.cli.utils import (
        carbon_summary,
        check_config,
        end_summary,
        get_config,
        get_struct_info,
        parse_typer_dicts,
        set_read_kwargs_index,
        start_summary,
    )

    # Check options from configuration file are all valid
    check_config(ctx)

    [read_kwargs, calc_kwargs, minimize_kwargs, write_kwargs] = parse_typer_dicts(
        [read_kwargs, calc_kwargs, minimize_kwargs, write_kwargs]
    )

    # Set initial config
    all_kwargs = {
        "read_kwargs": read_kwargs.copy(),
        "calc_kwargs": calc_kwargs.copy(),
        "minimize_kwargs": minimize_kwargs.copy(),
        "write_kwargs": write_kwargs.copy(),
    }
    config = get_config(params=ctx.params, all_kwargs=all_kwargs)

    # Read only first structure by default and ensure only one image is read
    set_read_kwargs_index(read_kwargs)

    # Check fmax option not duplicated
    if "fmax" in minimize_kwargs:
        raise ValueError("'fmax' must be passed through the --fmax option")
    minimize_kwargs["fmax"] = fmax

    log_kwargs = {"filemode": "w"}
    if log:
        log_kwargs["filename"] = log

    # Dictionary of inputs for EoS class
    eos_kwargs = {
        "struct": struct,
        "arch": arch,
        "device": device,
        "model": model,
        "model_path": model_path,
        "read_kwargs": read_kwargs,
        "calc_kwargs": calc_kwargs,
        "attach_logger": True,
        "log_kwargs": log_kwargs,
        "track_carbon": tracker,
        "min_volume": min_volume,
        "max_volume": max_volume,
        "n_volumes": n_volumes,
        "eos_type": eos_type,
        "minimize": minimize,
        "minimize_all": minimize_all,
        "minimize_kwargs": minimize_kwargs,
        "write_structures": write_structures,
        "write_kwargs": write_kwargs,
        "plot_to_file": plot_to_file,
        "file_prefix": file_prefix,
    }

    # Initialise EoS
    equation_of_state = EoS(**eos_kwargs)

    # Set summary and log files
    summary = equation_of_state._build_filename("eos-summary.yml", filename=summary)
    log = equation_of_state.log_kwargs["filename"]

    # Add structure, MLIP information, and log to info
    info = get_struct_info(
        struct=equation_of_state.struct,
        struct_path=struct,
    )

    output_files = equation_of_state.output_files

    # Save summary information before calculations begin
    start_summary(
        command="eos",
        summary=summary,
        info=info,
        config=config,
        output_files=output_files,
    )

    # Run equation of state calculations
    equation_of_state.run()

    # Save carbon summary
    if tracker:
        carbon_summary(summary=summary, log=log)

    # Time after calculations have finished
    end_summary(summary)
