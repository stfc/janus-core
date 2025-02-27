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
    LogPath,
    MinimizeKwargs,
    ModelPath,
    ReadKwargsLast,
    StructPath,
    Summary,
    TrajKwargs,
    WriteKwargs,
)
from janus_core.cli.utils import yaml_converter_callback

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
        `hydrostatic_strain` in the filter function.
    pressure
        Scalar pressure when optimizing cell geometry, in GPa. Passed to the filter
        function if either `opt_cell_lengths` or `opt_cell_fully` is True.
    """
    if "opt_kwargs" not in minimize_kwargs:
        minimize_kwargs["opt_kwargs"] = {}

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
@use_config(yaml_converter_callback)
def geomopt(
    # numpydoc ignore=PR02
    ctx: Context,
    struct: StructPath,
    optimizer: Annotated[
        str | None,
        Option(help="Name of ASE optimizer function to use."),
    ] = "LBFGS",
    fmax: Annotated[
        float, Option(help="Maximum force for convergence, in eV/Å.")
    ] = 0.1,
    steps: Annotated[int, Option(help="Maximum number of optimization steps.")] = 1000,
    arch: Architecture = "mace_mp",
    device: Device = "cpu",
    model_path: ModelPath = None,
    opt_cell_lengths: Annotated[
        bool,
        Option(help="Optimize cell vectors, as well as atomic positions."),
    ] = False,
    opt_cell_fully: Annotated[
        bool,
        Option(
            help="Fully optimize the cell vectors, angles, and atomic positions.",
        ),
    ] = False,
    filter_func: Annotated[
        str | None,
        Option(
            help=(
                "Name of ASE filter/constraint function to use. If using "
                "--opt-cell-lengths or --opt-cell-fully, defaults to "
                "`FrechetCellFilter` if available, otherwise `ExpCellFilter`."
            )
        ),
    ] = None,
    pressure: Annotated[
        float, Option(help="Scalar pressure when optimizing cell geometry, in GPa.")
    ] = 0.0,
    symmetrize: Annotated[
        bool, Option(help="Whether to refine symmetry after geometry optimization.")
    ] = False,
    symmetry_tolerance: Annotated[
        float,
        Option(
            help="Atom displacement tolerance for spglib symmetry determination, in Å."
        ),
    ] = 0.001,
    file_prefix: Annotated[
        Path | None,
        Option(help="Prefix for output filenames. Default is inferred from structure."),
    ] = None,
    out: Annotated[
        Path | None,
        Option(
            help=(
                "Path to save optimized structure. Default is inferred from name "
                "of structure file."
            ),
        ),
    ] = None,
    write_trajectory: Annotated[
        bool, Option(help="Whether to save a trajectory file of optimization frames.")
    ] = False,
    traj_kwargs: TrajKwargs = None,
    read_kwargs: ReadKwargsLast = None,
    calc_kwargs: CalcKwargs = None,
    minimize_kwargs: MinimizeKwargs = None,
    write_kwargs: WriteKwargs = None,
    log: LogPath = None,
    tracker: Annotated[
        bool, Option(help="Whether to save carbon emissions of calculation")
    ] = True,
    summary: Summary = None,
) -> None:
    """
    Perform geometry optimization and save optimized structure to file.

    Parameters
    ----------
    ctx
        Typer (Click) Context. Automatically set.
    struct
        Path of structure to simulate.
    optimizer
        Name of optimization function from ase.optimize. Default is `LBFGS`.
    fmax
        Set force convergence criteria for optimizer, in eV/Å. Default is 0.1.
    steps
        Set maximum number of optimization steps to run. Default is 1000.
    arch
        MLIP architecture to use for geometry optimization.
        Default is "mace_mp".
    device
        Device to run model on. Default is "cpu".
    model_path
        Path to MLIP model. Default is `None`.
    opt_cell_lengths
        Whether to optimize cell vectors, as well as atomic positions, by setting
        `hydrostatic_strain` in the filter function. Default is False.
    opt_cell_fully
        Whether to fully optimize the cell vectors, angles, and atomic positions.
        Default is False.
    filter_func
        Name of filter function from ase.filters or ase.constraints, to apply
        constraints to atoms. If using --opt-cell-lengths or --opt-cell-fully, defaults
        to `FrechetCellFilter` if available, otherwise `ExpCellFilter`.
    pressure
        Scalar pressure when optimizing cell geometry, in GPa. Passed to the filter
        function if either `opt_cell_lengths` or `opt_cell_fully` is True. Default is
        0.0.
    symmetrize
        Whether to refine symmetry after geometry optimization. Default is False.
    symmetry_tolerance
        Atom displacement tolerance for spglib symmetry determination, in Å.
        Default is 0.001.
    file_prefix
        Prefix for output filenames. Default is inferred from structure.
    out
        Path to save optimized structure, or last structure if optimization did not
        converge. Default is inferred from name of structure file.
    write_trajectory
        Whether to save a trajectory file of optimization frames.
    traj_kwargs
        Keyword arguments to pass to ase.io.write when saving trajectory.
        If "filename" is not included, it is inferred from --file-prefix.
    read_kwargs
        Keyword arguments to pass to ase.io.read. By default,
            read_kwargs["index"] is -1.
    calc_kwargs
        Keyword arguments to pass to the selected calculator. Default is {}.
    minimize_kwargs
        Other keyword arguments to pass to geometry optimizer. Default is {}.
    write_kwargs
        Keyword arguments to pass to ase.io.write when saving optimized structure.
        Default is {}.
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
    from janus_core.calculations.geom_opt import GeomOpt
    from janus_core.cli.utils import (
        carbon_summary,
        check_config,
        end_summary,
        parse_typer_dicts,
        save_struct_calc,
        set_read_kwargs_index,
        start_summary,
    )

    # Check options from configuration file are all valid
    check_config(ctx)

    [read_kwargs, calc_kwargs, minimize_kwargs, write_kwargs] = parse_typer_dicts(
        [read_kwargs, calc_kwargs, minimize_kwargs, write_kwargs]
    )

    # Read only first structure by default and ensure only one image is read
    set_read_kwargs_index(read_kwargs)

    # Check optimized structure path not duplicated
    if "filename" in write_kwargs:
        raise ValueError("'filename' must be passed through the --out option")
    if out:
        write_kwargs["filename"] = out

    _set_minimize_kwargs(minimize_kwargs, opt_cell_lengths, pressure)

    if opt_cell_fully or opt_cell_lengths:
        # Use default filter unless filter function explicitly passed
        opt_cell_fully_dict = {"filter_func": filter_func} if filter_func else {}
    else:
        if filter_func:
            raise ValueError(
                "--opt-cell-lengths or --opt-cell-fully must be set to use a filter "
                "function"
            )
        # Override default filter function with None
        opt_cell_fully_dict = {"filter_func": None}

    log_kwargs = {"filemode": "w"}
    if log:
        log_kwargs["filename"] = log

    # Dictionary of inputs for optimize function
    optimize_kwargs = {
        "struct": struct,
        "arch": arch,
        "device": device,
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
        "write_trajectory": write_trajectory,
        "traj_kwargs": traj_kwargs,
    }

    # Set up geometry optimization
    optimizer = GeomOpt(**optimize_kwargs)

    # Set summary and log files
    summary = optimizer._build_filename(
        "geomopt-summary.yml", filename=summary
    ).absolute()
    log = optimizer.log_kwargs["filename"]

    # Store inputs for yaml summary
    inputs = optimize_kwargs.copy()

    # Add structure, MLIP information, and log to inputs
    save_struct_calc(
        inputs=inputs,
        struct=optimizer.struct,
        struct_path=struct,
        arch=arch,
        device=device,
        model_path=model_path,
        read_kwargs=read_kwargs,
        calc_kwargs=calc_kwargs,
        log=log,
    )

    # Save summary information before optimization begins
    start_summary(command="geomopt", summary=summary, inputs=inputs)

    # Run geometry optimization and save output structure
    optimizer.run()

    # Save carbon summary
    if tracker:
        carbon_summary(summary=summary, log=log)

    # Time after optimization has finished
    end_summary(summary)
