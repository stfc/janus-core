"""Set up elasticity commandline interface."""

from __future__ import annotations

from typing import Annotated

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
    ReadKwargsLast,
    StructPath,
    Summary,
    Tracker,
    WriteKwargs,
)
from janus_core.cli.utils import yaml_converter_callback

app = Typer()


@app.command()
@use_config(yaml_converter_callback, param_help="Path to configuration file.")
def elasticity(
    # numpydoc ignore=PR02
    ctx: Context,
    # Required
    arch: Architecture,
    struct: StructPath,
    # Calculation
    shear_magnitude: Annotated[
        float,
        Option(
            help="Magnitude of shear strain for deformed structures.",
            rich_help_panel="Calculation",
        ),
    ] = 0.06,
    normal_magnitude: Annotated[
        float,
        Option(
            help="Magnitude of normal strain for deformed structures.",
            rich_help_panel="Calculation",
        ),
    ] = 0.01,
    n_strains: Annotated[
        int,
        Option(
            help="Number of normal and shear strains to use for deformed structures.",
            rich_help_panel="Calculation",
        ),
    ] = 4,
    write_voigt: Annotated[
        bool,
        Option(
            help="Write the ElasticityTensor in Voigt notation.",
            rich_help_panel="Calculation",
        ),
    ] = True,
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
    # MLIP Calculator
    device: Device = "cpu",
    model: Model = None,
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
    shear_magnitude
        The magnitude of shear strain to apply for deformed structures.
        Default is 0.06.
    normal_magnitude
        The magnitude of normal strain to apply for deformed structures.
        Default is 0.01.
    n_strains
        The number of normal and shear strains to apply for deformed structures.
        Default is 4.
    write_voigt
        Whether to write out in Voigt notation, Default is True.
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
    device
        Device to run model on. Default is "cpu".
    model
        Path to MLIP model or name of model. Default is `None`.
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
    from janus_core.calculations.elasticity import Elasticity
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

    # Dictionary of inputs for Elasticity class
    elasticity_kwargs = {
        "struct": struct,
        "arch": arch,
        "device": device,
        "model": model,
        "read_kwargs": read_kwargs,
        "calc_kwargs": calc_kwargs,
        "attach_logger": True,
        "log_kwargs": log_kwargs,
        "track_carbon": tracker,
        "minimize": minimize,
        "minimize_all": minimize_all,
        "minimize_kwargs": minimize_kwargs,
        "write_structures": write_structures,
        "write_kwargs": write_kwargs,
        "file_prefix": file_prefix,
        "write_voigt": write_voigt,
        "shear_magnitude": shear_magnitude,
        "normal_magnitude": normal_magnitude,
        "n_strains": n_strains,
    }

    # Initialise Elasticity
    elasticity = Elasticity(**elasticity_kwargs)

    # Set summary and log files
    summary = elasticity._build_filename("elasticity-summary.yml", filename=summary)
    log = elasticity.log_kwargs["filename"]

    # Add structure, MLIP information, and log to info
    info = get_struct_info(
        struct=elasticity.struct,
        struct_path=struct,
    )

    output_files = elasticity.output_files

    # Save summary information before calculations begin
    start_summary(
        command="elasticity",
        summary=summary,
        info=info,
        config=config,
        output_files=output_files,
    )

    # Run elasticity calculations
    elasticity.run()

    # Save carbon summary
    if tracker:
        carbon_summary(summary=summary, log=log)

    # Time after calculations have finished
    end_summary(summary)
