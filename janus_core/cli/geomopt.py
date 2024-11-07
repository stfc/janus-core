# ruff: noqa: I002, FA100
"""Set up geomopt commandline interface."""

# Issues with future annotations and typer
# c.f. https://github.com/maxb2/typer-config/issues/295
# from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Optional

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
    WriteKwargs,
)
from janus_core.cli.utils import yaml_converter_callback

app = Typer()


def _set_minimize_kwargs(
    minimize_kwargs: dict[str, Any],
    traj: Optional[str],
    opt_cell_lengths: bool,
    pressure: float,
) -> None:
    """
    Set minimize_kwargs dictionary values.

    Parameters
    ----------
    minimize_kwargs : dict[str, Any]
        Other keyword arguments to pass to geometry optimizer.
    traj : Optional[str]
        Path if saving optimization frames.
    opt_cell_lengths : bool
        Whether to optimize cell vectors, as well as atomic positions, by setting
        `hydrostatic_strain` in the filter function.
    pressure : float
        Scalar pressure when optimizing cell geometry, in GPa. Passed to the filter
        function if either `opt_cell_lengths` or `opt_cell_fully` is True.
    """
    if "opt_kwargs" in minimize_kwargs:
        # Check trajectory path not duplicated
        if "trajectory" in minimize_kwargs["opt_kwargs"]:
            raise ValueError("'trajectory' must be passed through the --traj option")
    else:
        minimize_kwargs["opt_kwargs"] = {}

    if "traj_kwargs" not in minimize_kwargs:
        minimize_kwargs["traj_kwargs"] = {}

    # Set same trajectory filenames to overwrite saved binary with xyz
    if traj:
        minimize_kwargs["opt_kwargs"]["trajectory"] = traj
        minimize_kwargs["traj_kwargs"]["filename"] = traj

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
        Optional[str],
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
        Optional[str],
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
    out: Annotated[
        Optional[Path],
        Option(
            help=(
                "Path to save optimized structure. Default is inferred from name "
                "of structure file."
            ),
        ),
    ] = None,
    traj: Annotated[
        str,
        Option(help="Path if saving optimization frames."),
    ] = None,
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
    ctx : Context
        Typer (Click) Context. Automatically set.
    struct : Path
        Path of structure to simulate.
    optimizer : Optional[str]
        Name of optimization function from ase.optimize. Default is `LBFGS`.
    fmax : float
        Set force convergence criteria for optimizer, in eV/Å. Default is 0.1.
    steps : int
        Set maximum number of optimization steps to run. Default is 1000.
    arch : Optional[str]
        MLIP architecture to use for geometry optimization.
        Default is "mace_mp".
    device : Optional[str]
        Device to run model on. Default is "cpu".
    model_path : Optional[str]
        Path to MLIP model. Default is `None`.
    opt_cell_lengths : bool
        Whether to optimize cell vectors, as well as atomic positions, by setting
        `hydrostatic_strain` in the filter function. Default is False.
    opt_cell_fully : bool
        Whether to fully optimize the cell vectors, angles, and atomic positions.
        Default is False.
    filter_func : Optional[str]
        Name of filter function from ase.filters or ase.constraints, to apply
        constraints to atoms. If using --opt-cell-lengths or --opt-cell-fully, defaults
        to `FrechetCellFilter` if available, otherwise `ExpCellFilter`.
    pressure : float
        Scalar pressure when optimizing cell geometry, in GPa. Passed to the filter
        function if either `opt_cell_lengths` or `opt_cell_fully` is True. Default is
        0.0.
    symmetrize : bool
        Whether to refine symmetry after geometry optimization. Default is False.
    symmetry_tolerance : float
        Atom displacement tolerance for spglib symmetry determination, in Å.
        Default is 0.001.
    out : Optional[Path]
        Path to save optimized structure, or last structure if optimization did not
        converge. Default is inferred from name of structure file.
    traj : Optional[str]
        Path if saving optimization frames. Default is None.
    read_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ase.io.read. By default,
            read_kwargs["index"] is -1.
    calc_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to the selected calculator. Default is {}.
    minimize_kwargs : Optional[dict[str, Any]]
        Other keyword arguments to pass to geometry optimizer. Default is {}.
    write_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ase.io.write when saving optimized structure.
        Default is {}.
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

    _set_minimize_kwargs(minimize_kwargs, traj, opt_cell_lengths, pressure)

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
        "struct_path": struct,
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
        **opt_cell_fully_dict,
        **minimize_kwargs,
        "write_results": True,
        "write_kwargs": write_kwargs,
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
