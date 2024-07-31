"""Set up geomopt commandline interface."""

from pathlib import Path
from typing import Annotated, Any, Optional

from typer import Context, Option, Typer
from typer_config import use_config

from janus_core.calculations.geom_opt import GeomOpt
from janus_core.calculations.single_point import SinglePoint
from janus_core.cli.types import (
    Architecture,
    CalcKwargs,
    Device,
    LogPath,
    MinimizeKwargs,
    ModelPath,
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
from janus_core.helpers.utils import dict_paths_to_strs

app = Typer()


def _set_minimize_kwargs(
    minimize_kwargs: dict[str, Any],
    traj: Optional[str],
    vectors_only: bool,
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
    vectors_only : bool
        Whether to optimize cell vectors, as well as atomic positions, by setting
        `hydrostatic_strain` in the filter function.
    pressure : float
        Scalar pressure when optimizing cell geometry, in GPa. Passed to the filter
        function if either `vectors_only` or `fully_opt` is True.
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
                "'hydrostatic_strain' must be passed through the --vectors-only option"
            )
        if "scalar_pressure" in minimize_kwargs["filter_kwargs"]:
            raise ValueError(
                "'scalar_pressure' must be passed through the --pressure option"
            )
    else:
        minimize_kwargs["filter_kwargs"] = {}

    # Set hydrostatic_strain and scalar pressure
    minimize_kwargs["filter_kwargs"]["hydrostatic_strain"] = vectors_only
    minimize_kwargs["filter_kwargs"]["scalar_pressure"] = pressure


@app.command()
@use_config(yaml_converter_callback)
def geomopt(
    # pylint: disable=too-many-arguments,too-many-locals,duplicate-code
    # numpydoc ignore=PR02
    ctx: Context,
    struct: StructPath,
    optimizer: Annotated[
        str,
        Option(help="Name of ASE optimizer function to use."),
    ] = "LBFGS",
    fmax: Annotated[
        float, Option(help="Maximum force for convergence, in eV/Å.")
    ] = 0.1,
    steps: Annotated[int, Option(help="Maximum number of optimization steps.")] = 1000,
    arch: Architecture = "mace_mp",
    device: Device = "cpu",
    model_path: ModelPath = None,
    vectors_only: Annotated[
        bool,
        Option(help="Optimize cell vectors, as well as atomic positions."),
    ] = False,
    fully_opt: Annotated[
        bool,
        Option(
            help="Fully optimize the cell vectors, angles, and atomic positions.",
        ),
    ] = False,
    filter_func: Annotated[
        str,
        Option(
            help=(
                "Name of ASE filter/constraint function to use. If using "
                "--vectors-only or --fully-opt, defaults to `FrechetCellFilter` if "
                "available, otherwise `ExpCellFilter`."
            )
        ),
    ] = None,
    pressure: Annotated[
        float, Option(help="Scalar pressure when optimizing cell geometry, in GPa.")
    ] = 0.0,
    out: Annotated[
        Path,
        Option(
            help=(
                "Path to save optimized structure. Default is inferred from name "
                "of structure file."
            ),
        ),
    ] = None,
    traj: Annotated[
        str,
        Option(help="Path if saving optimization frames.  [default: None]"),
    ] = None,
    read_kwargs: ReadKwargs = None,
    calc_kwargs: CalcKwargs = None,
    minimize_kwargs: MinimizeKwargs = None,
    write_kwargs: WriteKwargs = None,
    log: LogPath = "geomopt.log",
    summary: Summary = "geomopt_summary.yml",
):
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
    vectors_only : bool
        Whether to optimize cell vectors, as well as atomic positions, by setting
        `hydrostatic_strain` in the filter function. Default is False.
    fully_opt : bool
        Whether to fully optimize the cell vectors, angles, and atomic positions.
        Default is False.
    filter_func : Optional[str]
        Name of filter function from ase.filters or ase.constraints, to apply
        constraints to atoms. If using --vectors only or --fully-opt, defaults to
        `FrechetCellFilter` if available, otherwise `ExpCellFilter`.
    pressure : float
        Scalar pressure when optimizing cell geometry, in GPa. Passed to the filter
        function if either `vectors_only` or `fully_opt` is True. Default is 0.0.
    out : Optional[Path]
        Path to save optimized structure, or last structure if optimization did not
        converge. Default is inferred from name of structure file.
    traj : Optional[str]
        Path if saving optimization frames. Default is None.
    read_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ase.io.read. Default is {}.
    calc_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to the selected calculator. Default is {}.
    minimize_kwargs : Optional[dict[str, Any]]
        Other keyword arguments to pass to geometry optimizer. Default is {}.
    write_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ase.io.write when saving optimized structure.
        Default is {}.
    log : Optional[Path]
        Path to write logs to. Default is "geomopt.log".
    summary : Path
        Path to save summary of inputs and start/end time. Default is
        geomopt_summary.yml.
    config : Path
        Path to yaml configuration file to define the above options. Default is None.
    """
    # Check options from configuration file are all valid
    check_config(ctx)

    [read_kwargs, calc_kwargs, minimize_kwargs, write_kwargs] = parse_typer_dicts(
        [read_kwargs, calc_kwargs, minimize_kwargs, write_kwargs]
    )

    # Set up single point calculator
    s_point = SinglePoint(
        struct_path=struct,
        architecture=arch,
        device=device,
        model_path=model_path,
        read_kwargs=read_kwargs,
        calc_kwargs=calc_kwargs,
        log_kwargs={"filename": log, "filemode": "w"},
    )

    # Check optimized structure path not duplicated
    if "filename" in write_kwargs:
        raise ValueError("'filename' must be passed through the --out option")

    # Set default filname for writing optimized structure if not specified
    if out:
        write_kwargs["filename"] = out
    else:
        write_kwargs["filename"] = f"{s_point.struct_name}-opt.extxyz"

    _set_minimize_kwargs(minimize_kwargs, traj, vectors_only, pressure)

    if fully_opt or vectors_only:
        # Use default filter unless filter function explicitly passed
        fully_opt_dict = {"filter_func": filter_func} if filter_func else {}
    else:
        if filter_func:
            raise ValueError(
                "--vectors-only or --fully-opt must be set to use a filter function"
            )
        # Override default filter function with None
        fully_opt_dict = {"filter_func": None}

    # Dictionary of inputs for optimize function
    optimize_kwargs = {
        "struct": s_point.struct,
        "optimizer": optimizer,
        "fmax": fmax,
        "steps": steps,
        **fully_opt_dict,
        **minimize_kwargs,
        "write_results": True,
        "write_kwargs": write_kwargs,
        "log_kwargs": {"filename": log, "filemode": "a"},
    }

    # Store inputs for yaml summary
    inputs = optimize_kwargs.copy()

    # Store only filename as filemode is not set by user
    del inputs["log_kwargs"]
    inputs["log"] = log

    save_struct_calc(
        inputs, s_point, arch, device, model_path, read_kwargs, calc_kwargs
    )

    # Convert all paths to strings in inputs nested dictionary
    dict_paths_to_strs(inputs)

    # Save summary information before optimization begins
    start_summary(command="geomopt", summary=summary, inputs=inputs)

    # Run geometry optimization and save output structure
    optimizer = GeomOpt(**optimize_kwargs)
    optimizer.run()

    # Time after optimization has finished
    end_summary(summary)
