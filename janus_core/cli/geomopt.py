"""Set up geomopt commandline interface."""

from pathlib import Path
from typing import Annotated

from typer import Option, Typer
from typer_config import use_config

from janus_core.calculations.geom_opt import optimize
from janus_core.calculations.single_point import SinglePoint
from janus_core.cli.types import (
    Architecture,
    CalcKwargs,
    Device,
    LogPath,
    OptKwargs,
    ReadKwargs,
    StructPath,
    Summary,
    WriteKwargs,
)
from janus_core.cli.utils import (
    end_summary,
    parse_typer_dicts,
    start_summary,
    yaml_converter_callback,
)
from janus_core.helpers.utils import dict_paths_to_strs

app = Typer()


@app.command(
    help="Perform geometry optimization and save optimized structure to file.",
)
@use_config(yaml_converter_callback)
def geomopt(
    # pylint: disable=too-many-arguments,too-many-locals,duplicate-code
    # numpydoc ignore=PR02
    struct: StructPath,
    fmax: Annotated[float, Option(help="Maximum force for convergence.")] = 0.1,
    steps: Annotated[int, Option(help="Maximum number of optimization steps.")] = 1000,
    arch: Architecture = "mace_mp",
    device: Device = "cpu",
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
    opt_kwargs: OptKwargs = None,
    write_kwargs: WriteKwargs = None,
    log: LogPath = "geomopt.log",
    summary: Summary = "geomopt_summary.yml",
):
    """
    Perform geometry optimization and save optimized structure to file.

    Parameters
    ----------
    struct : Path
        Path of structure to simulate.
    fmax : float
        Set force convergence criteria for optimizer in units eV/Å.
        Default is 0.1.
    steps : int
        Set maximum number of optimization steps to run. Default is 1000.
    arch : Optional[str]
        MLIP architecture to use for geometry optimization.
        Default is "mace_mp".
    device : Optional[str]
        Device to run model on. Default is "cpu".
    vectors_only : bool
        Whether to optimize cell vectors, as well as atomic positions, by setting
        `hydrostatic_strain` in the filter function. Default is False.
    fully_opt : bool
        Whether to fully optimize the cell vectors, angles, and atomic positions.
        Default is False.
    out : Optional[Path]
        Path to save optimized structure, or last structure if optimization did not
        converge. Default is inferred from name of structure file.
    traj : Optional[str]
        Path if saving optimization frames. Default is None.
    read_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ase.io.read. Default is {}.
    calc_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to the selected calculator. Default is {}.
    opt_kwargs : Optional[ASEOptArgs]
        Keyword arguments to pass to optimizer. Default is {}.
    write_kwargs : Optional[ASEWriteArgs]
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
    [read_kwargs, calc_kwargs, opt_kwargs, write_kwargs] = parse_typer_dicts(
        [read_kwargs, calc_kwargs, opt_kwargs, write_kwargs]
    )

    # Set up single point calculator
    s_point = SinglePoint(
        struct_path=struct,
        architecture=arch,
        device=device,
        read_kwargs=read_kwargs,
        calc_kwargs=calc_kwargs,
        log_kwargs={"filename": log, "filemode": "w"},
    )

    # Check optimized structure path not duplicated
    if "filename" in write_kwargs:
        raise ValueError("'filename' must be passed through the --out option")

    # Check trajectory path not duplicated
    if "trajectory" in opt_kwargs:
        raise ValueError("'trajectory' must be passed through the --traj option")

    # Set default filname for writing optimized structure if not specified
    if out:
        write_kwargs["filename"] = out
    else:
        write_kwargs["filename"] = f"{s_point.struct_name}-opt.xyz"

    # Set same trajectory filenames to overwrite saved binary with xyz
    opt_kwargs["trajectory"] = traj if traj else None
    traj_kwargs = {"filename": traj} if traj else None

    # Set hydrostatic_strain
    # If not passed --fully-opt or --vectors-only, will be unused
    filter_kwargs = {"hydrostatic_strain": vectors_only}

    # Use default filter if passed --fully-opt or --vectors-only
    # Otherwise override with None
    fully_opt_dict = {} if (fully_opt or vectors_only) else {"filter_func": None}

    # Dictionary of inputs for optimize function
    optimize_kwargs = {
        "struct": s_point.struct,
        "fmax": fmax,
        "steps": steps,
        "filter_kwargs": filter_kwargs,
        **fully_opt_dict,
        "opt_kwargs": opt_kwargs,
        "write_results": True,
        "write_kwargs": write_kwargs,
        "traj_kwargs": traj_kwargs,
        "log_kwargs": {"filename": log, "filemode": "a"},
    }

    # Store inputs for yaml summary
    inputs = optimize_kwargs.copy()

    # Store only filename as filemode is not set by user
    del inputs["log_kwargs"]
    inputs["log"] = log

    inputs["struct"] = {
        "n_atoms": len(s_point.struct),
        "struct_path": struct,
        "struct_name": s_point.struct_name,
        "formula": s_point.struct.get_chemical_formula(),
    }

    inputs["calc"] = {
        "arch": arch,
        "device": device,
        "read_kwargs": read_kwargs,
        "calc_kwargs": calc_kwargs,
    }

    # Convert all paths to strings in inputs nested dictionary
    dict_paths_to_strs(inputs)

    # Save summary information before optimization begins
    start_summary(command="geomopt", summary=summary, inputs=inputs)

    # Run geometry optimization and save output structure
    optimize(**optimize_kwargs)

    # Time after optimization has finished
    end_summary(summary)
