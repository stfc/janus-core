"""Set up NEB commandline interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, get_args

from click import Choice
from typer import Context, Option, Typer
from typer_config import use_config

from janus_core.cli.types import (
    Architecture,
    CalcKwargs,
    Device,
    FilePrefix,
    InterpolationKwargs,
    LogPath,
    MinimizeKwargs,
    ModelPath,
    NebKwargs,
    NebOptKwargs,
    ReadKwargsLast,
    Summary,
    Tracker,
    WriteKwargs,
)
from janus_core.cli.utils import yaml_converter_callback

app = Typer()


@app.command()
@use_config(yaml_converter_callback, param_help="Path to configuration file.")
def neb(
    # numpydoc ignore=PR02
    ctx: Context,
    # Calculation
    init_struct: Annotated[
        Path | None,
        Option(
            help="Path of initial structure in band.", rich_help_panel="Calculation"
        ),
    ] = None,
    final_struct: Annotated[
        Path | None,
        Option(help="Path of final structure in band.", rich_help_panel="Calculation"),
    ] = None,
    band_structs: Annotated[
        Path | None,
        Option(help="Path of all band images.", rich_help_panel="Calculation"),
    ] = None,
    neb_class: Annotated[
        str | None,
        Option(help="Name of ASE NEB class to use.", rich_help_panel="Calculation"),
    ] = "NEB",
    n_images: Annotated[
        int,
        Option(help="Number of images to use in NEB.", rich_help_panel="Calculation"),
    ] = 15,
    write_band: Annotated[
        bool,
        Option(
            help="Whether to write out all band images after optimization.",
            rich_help_panel="Calculation",
        ),
    ] = False,
    neb_kwargs: NebKwargs = None,
    interpolator: Annotated[
        str | None,
        Option(
            click_type=Choice(["ase", "pymatgen"]),
            help="Choice of interpolation strategy.",
            rich_help_panel="Calculation",
        ),
    ] = "ase",
    interpolator_kwargs: InterpolationKwargs = None,
    optimizer: Annotated[
        str | None,
        Option(help="Name of ASE NEB optimizer to use.", rich_help_panel="Calculation"),
    ] = "NEBOptimizer",
    fmax: Annotated[
        float,
        Option(help="Maximum force for NEB optimizer.", rich_help_panel="Calculation"),
    ] = 0.1,
    steps: Annotated[
        int,
        Option(
            help="Maximum number of steps for optimization.",
            rich_help_panel="Calculation",
        ),
    ] = 100,
    optimizer_kwargs: NebOptKwargs = None,
    plot_band: Annotated[
        bool,
        Option(
            help="Whether to plot and save NEB band.", rich_help_panel="Calculation"
        ),
    ] = False,
    minimize: Annotated[
        bool,
        Option(
            help=" Whether to minimize initial and final structures.",
            rich_help_panel="Calculation",
        ),
    ] = False,
    minimize_kwargs: MinimizeKwargs = None,
    # MLIP Calculator
    arch: Architecture = "mace_mp",
    device: Device = "cpu",
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

    init_struct
        Path of initial structure for Nudged Elastic Band method. Required if
        `band_structs` is None. Default is None.
    final_struct
        Path of final structure for Nudged Elastic Band method. Required if
        `band_structs` is None. Default is None.
    band_structs
        Path of band images to optimize, skipping interpolation between the
        initial and final structures. Sets `interpolator` to None.
    neb_class
        Nudged Elastic Band class to use. Default is ase.mep.NEB.
    n_images
        Number of images to use in NEB. Default is 15.
    write_band
        Whether to write out all band images after optimization. Default is False.
    neb_kwargs
        Keyword arguments to pass to neb_class. Default is {}.
    interpolator
        Choice of interpolation strategy. Default is "ase" if using `init_struct` and
        `final_struct`, or None if using `band_structs`.
    interpolator_kwargs
        Keyword arguments to pass to interpolator. Default is {"method": "idpp"} for
        "ase" interpolator, or {"interpolate_lattices": False, "autosort_tol", 0.5}
        for "pymatgen".
    optimizer
        Optimizer to apply to NEB object. Default is NEBOptimizer.
    fmax
        Maximum force for NEB optimizer. Default is 0.1.
    steps
        Maximum number of steps to optimize NEB. Default is 100.
    optimizer_kwargs
        Keyword arguments to pass to optimizer. Deault is {}.
    plot_band
        Whether to plot and save NEB band. Default is False.
    minimize
        Whether to perform geometry optimisation on initial and final structures.
        Default is False.
    minimize_kwargs
        Keyword arguments to pass to geometry optimizer. Default is {}.
    arch
        MLIP architecture to use for Nudged Elastic Band method. Default is "mace_mp".
    device
        Device to run MLIP model on. Default is "cpu".
    model_path
        Path to MLIP model. Default is `None`.
    calc_kwargs
        Keyword arguments to pass to the selected calculator. Default is {}.
    file_prefix
        Prefix for output files, including directories. Default directory is
        ./janus_results, and default filename prefix is inferred from the input
        stucture filename.
    read_kwargs
        Keyword arguments to pass to ase.io.read. By default, read_kwargs["index"]
        is -1 if using `init_struct` and `final_struct`, or ":" for `band_structs`.
    write_kwargs
        Keyword arguments to pass to ase.io.write when writing images.
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
    from janus_core.calculations.neb import NEB
    from janus_core.cli.utils import (
        carbon_summary,
        check_config,
        end_summary,
        get_config,
        get_struct_info,
        parse_typer_dicts,
        start_summary,
    )
    from janus_core.helpers.janus_types import Interpolators

    # Check options from configuration file are all valid
    check_config(ctx)

    [
        write_kwargs,
        neb_kwargs,
        interpolator_kwargs,
        optimizer_kwargs,
        minimize_kwargs,
        read_kwargs,
        calc_kwargs,
    ] = parse_typer_dicts(
        [
            write_kwargs,
            neb_kwargs,
            interpolator_kwargs,
            optimizer_kwargs,
            minimize_kwargs,
            read_kwargs,
            calc_kwargs,
        ]
    )

    # Set initial config
    all_kwargs = {
        "write_kwargs": write_kwargs.copy(),
        "neb_kwargs": neb_kwargs.copy(),
        "interpolator_kwargs": interpolator_kwargs.copy(),
        "optimizer_kwargs": optimizer_kwargs.copy(),
        "minimize_kwargs": minimize_kwargs.copy(),
        "read_kwargs": read_kwargs.copy(),
        "calc_kwargs": calc_kwargs.copy(),
    }
    config = get_config(params=ctx.params, all_kwargs=all_kwargs)

    if band_structs:
        if init_struct or final_struct:
            raise ValueError(
                "Initial and final structures cannot be specified in addition to the "
                "band structures"
            )
        interpolator = None

    if not band_structs and interpolator not in get_args(Interpolators):
        raise ValueError(f"Fit type must be one of: {get_args(Interpolators)}")

    log_kwargs = {"filemode": "w"}
    if log:
        log_kwargs["filename"] = log

    # Dictionary of inputs for NEB class
    neb_inputs = {
        "init_struct": init_struct,
        "final_struct": final_struct,
        "band_structs": band_structs,
        "arch": arch,
        "device": device,
        "model_path": model_path,
        "read_kwargs": read_kwargs,
        "calc_kwargs": calc_kwargs,
        "attach_logger": True,
        "log_kwargs": log_kwargs,
        "track_carbon": tracker,
        "neb_class": neb_class,
        "n_images": n_images,
        "write_band": write_band,
        "write_kwargs": write_kwargs,
        "neb_kwargs": neb_kwargs,
        "interpolator": interpolator,
        "interpolator_kwargs": interpolator_kwargs,
        "optimizer": optimizer,
        "fmax": fmax,
        "steps": steps,
        "optimizer_kwargs": optimizer_kwargs,
        "plot_band": plot_band,
        "minimize": minimize,
        "minimize_kwargs": minimize_kwargs,
        "file_prefix": file_prefix,
    }

    # Initialise NEB
    neb = NEB(**neb_inputs)

    # Set summary and log files
    summary = neb._build_filename("neb-summary.yml", filename=summary)
    log = neb.log_kwargs["filename"]

    # Add structure, MLIP information, and log to info
    info = get_struct_info(
        struct=neb.struct,
        struct_path=init_struct,
    )

    if band_structs:
        info["band_structs"] = info.pop("traj")
    else:
        info["init_struct"] = info.pop("struct")
        info["final_struct"] = {
            "n_atoms": len(neb.final_struct),
            "struct_path": final_struct,
            "formula": neb.final_struct.get_chemical_formula(),
        }

    output_files = neb.output_files

    # Save summary information before calculations begin
    start_summary(
        command="neb",
        summary=summary,
        info=info,
        config=config,
        output_files=output_files,
    )

    # Run equation of state calculations
    neb.run()

    # Save carbon summary
    if tracker:
        carbon_summary(summary=summary, log=log)

    # Time after calculations have finished
    end_summary(summary)
