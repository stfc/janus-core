"""Set up geomopt commandline interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

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
from janus_core.cli.utils import deprecated_option, yaml_converter_callback

app = Typer()


def _set_minimize_kwargs(
    minimize_kwargs: dict[str, Any],
    opt_cell_lengths: bool,
    pressure: float,
) -> None:
    """
    Set minimize_kwargs dictionary values.

    Parameters
    ----------
    minimize_kwargs
        Other keyword arguments to pass to geometry optimizer.
    opt_cell_lengths
        Whether to optimize cell vectors, as well as atomic positions, by setting
        `hydrostatic_strain` in the filter.
    pressure
        Scalar pressure when optimizing cell geometry, in GPa. Passed to the filter
        function if either `opt_cell_lengths` or `opt_cell_fully` is True.
    """
    minimize_kwargs.setdefault("opt_kwargs", {})
    minimize_kwargs.setdefault("traj_kwargs", {})

    # Check hydrostatic_strain and scalar pressure not duplicated
    if "filter_kwargs" in minimize_kwargs:
        if "hydrostatic_strain" in minimize_kwargs["filter_kwargs"]:
            raise ValueError(
                "'hydrostatic_strain' must be passed through the --opt-cell-lengths "
                "option"
            )
        if "scalar_pressure" in minimize_kwargs["filter_kwargs"]:
            raise ValueError(
                "'scalar_pressure' must be passed through the --pressure option"
            )
    else:
        minimize_kwargs["filter_kwargs"] = {}

    # Set hydrostatic_strain and scalar pressure
    minimize_kwargs["filter_kwargs"]["hydrostatic_strain"] = opt_cell_lengths
    minimize_kwargs["filter_kwargs"]["scalar_pressure"] = pressure


@app.command()
@use_config(yaml_converter_callback, param_help="Path to configuration file.")
def geomopt(
    # numpydoc ignore=PR02
    ctx: Context,
    # Required
    arch: Architecture,
    struct: StructPath,
    # Calculation
    optimizer: Annotated[
        str | None,
        Option(
            help="Name of ASE optimizer function to use.",
            rich_help_panel="Calculation",
        ),
    ] = "LBFGS",
    fmax: Annotated[
        float,
        Option(
            help="Maximum force for convergence, in eV/Å.",
            rich_help_panel="Calculation",
        ),
    ] = 0.1,
    steps: Annotated[
        int,
        Option(
            help="Maximum number of optimization steps.",
            rich_help_panel="Calculation",
        ),
    ] = 1000,
    opt_cell_lengths: Annotated[
        bool,
        Option(
            help="Optimize cell vectors, as well as atomic positions.",
            rich_help_panel="Calculation",
        ),
    ] = False,
    opt_cell_fully: Annotated[
        bool,
        Option(
            help="Fully optimize the cell vectors, angles, and atomic positions.",
            rich_help_panel="Calculation",
        ),
    ] = False,
    filter_class: Annotated[
        str | None,
        Option(
            "--filter",
            help=(
                "Name of ASE filter to wrap around atoms. If using "
                "--opt-cell-lengths or --opt-cell-fully, defaults to "
                "`FrechetCellFilter`."
            ),
            rich_help_panel="Calculation",
        ),
    ] = None,
    filter_func: Annotated[
        str | None,
        Option(
            help="Deprecated. Please use --filter",
            rich_help_panel="Calculation",
            callback=deprecated_option,
            hidden=True,
        ),
    ] = None,
    pressure: Annotated[
        float,
        Option(
            help="Scalar pressure when optimizing cell geometry, in GPa.",
            rich_help_panel="Calculation",
        ),
    ] = 0.0,
    symmetrize: Annotated[
        bool,
        Option(
            help="Whether to refine symmetry after geometry optimization.",
            rich_help_panel="Calculation",
        ),
    ] = False,
    symmetry_tolerance: Annotated[
        float,
        Option(
            help="Atom displacement tolerance for spglib symmetry determination, in Å.",
            rich_help_panel="Calculation",
        ),
    ] = 0.001,
    out: Annotated[
        Path | None,
        Option(
            help=(
                "Path to save optimized structure. Default is inferred `file_prefix`."
            ),
            rich_help_panel="Calculation",
        ),
    ] = None,
    write_traj: Annotated[
        bool,
        Option(
            help=(
                "Whether to save a trajectory file of optimization frames. "
                'If traj_kwargs["filename"] is not specified, it is inferred '
                "from `file_prefix`."
            ),
            rich_help_panel="Calculation",
        ),
    ] = False,
    minimize_kwargs: MinimizeKwargs = None,
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
    Perform geometry optimization and save optimized structure to file.

    Parameters
    ----------
    ctx
        Typer (Click) Context. Automatically set.
    arch
        MLIP architecture to use for geometry optimization.
    struct
        Path of structure to simulate.
    optimizer
        Name of optimization function from ase.optimize. Default is `LBFGS`.
    fmax
        Set force convergence criteria for optimizer, in eV/Å. Default is 0.1.
    steps
        Set maximum number of optimization steps to run. Default is 1000.
    opt_cell_lengths
        Whether to optimize cell vectors, as well as atomic positions, by setting
        `hydrostatic_strain` in the filter. Default is False.
    opt_cell_fully
        Whether to fully optimize the cell vectors, angles, and atomic positions.
        Default is False.
    filter_class
        Name of filter from ase.filters to wrap around atoms. If using
        --opt-cell-lengths or --opt-cell-fully, defaults to `FrechetCellFilter`.
    filter_func
        Deprecated. Please use `filter_class`.
    pressure
        Scalar pressure when optimizing cell geometry, in GPa. Passed to the filter
        function if either `opt_cell_lengths` or `opt_cell_fully` is True. Default is
        0.0.
    symmetrize
        Whether to refine symmetry after geometry optimization. Default is False.
    symmetry_tolerance
        Atom displacement tolerance for spglib symmetry determination, in Å.
        Default is 0.001.
    out
        Path to save optimized structure, or last structure if optimization did not
        converge. Default is inferred from `file_prefix`.
    write_traj
        Whether to save a trajectory file of optimization frames.
        If traj_kwargs["filename"] is not specified, it is inferred from `file_prefix`.
    minimize_kwargs
        Other keyword arguments to pass to geometry optimizer. Default is {}.
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
        Keyword arguments to pass to ase.io.write when saving optimized structure.
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
    from janus_core.calculations.geom_opt import GeomOpt
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

    # Check optimized structure path not duplicated
    if "filename" in write_kwargs:
        raise ValueError("'filename' must be passed through the --out option")
    if out:
        write_kwargs["filename"] = out

    _set_minimize_kwargs(minimize_kwargs, opt_cell_lengths, pressure)

    if filter_func and filter_class:
        raise ValueError("--filter-func is deprecated, please only use --filter")

    if opt_cell_fully or opt_cell_lengths:
        # Use default filter unless filter explicitly passed
        if filter_class:
            opt_cell_fully_dict = {"filter_class": filter_class}
        elif filter_func:
            opt_cell_fully_dict = {"filter_func": filter_func, "filter_class": None}
        else:
            opt_cell_fully_dict = {}
    else:
        if filter_class or filter_func:
            raise ValueError(
                "--opt-cell-lengths or --opt-cell-fully must be set to use a filter"
            )
        # Override default filter class with None
        opt_cell_fully_dict = {"filter_class": None}

    log_kwargs = {"filemode": "w"}
    if log:
        log_kwargs["filename"] = log

    # Dictionary of inputs for optimize function
    optimize_kwargs = {
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
        "optimizer": optimizer,
        "fmax": fmax,
        "steps": steps,
        "symmetrize": symmetrize,
        "symmetry_tolerance": symmetry_tolerance,
        "file_prefix": file_prefix,
        **opt_cell_fully_dict,
        **minimize_kwargs,
        "write_results": True,
        "write_kwargs": write_kwargs,
        "write_traj": write_traj,
    }

    # Set up geometry optimization
    optimizer = GeomOpt(**optimize_kwargs)

    # Set summary and log files
    summary = optimizer._build_filename("geomopt-summary.yml", filename=summary)
    log = optimizer.log_kwargs["filename"]

    # Add structure, MLIP information, and log to info
    info = get_struct_info(
        struct=optimizer.struct,
        struct_path=struct,
    )

    output_files = optimizer.output_files

    # Save summary information before optimization begins
    start_summary(
        command="geomopt",
        summary=summary,
        info=info,
        config=config,
        output_files=output_files,
    )

    # Run geometry optimization and save output structure
    optimizer.run()

    # Save carbon summary
    if tracker:
        carbon_summary(summary=summary, log=log)

    # Time after optimization has finished
    end_summary(summary)
