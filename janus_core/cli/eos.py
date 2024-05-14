"""Set up eos commandline interface."""

from pathlib import Path
from typing import Annotated, Optional

from typer import Context, Option, Typer
from typer_config import use_config

from janus_core.calculations.eos import calc_eos
from janus_core.calculations.single_point import SinglePoint
from janus_core.cli.types import (
    Architecture,
    CalcKwargs,
    Device,
    LogPath,
    MinimizeKwargs,
    ReadKwargs,
    StructPath,
    Summary,
)
from janus_core.cli.utils import (
    check_config,
    end_summary,
    parse_typer_dicts,
    start_summary,
    yaml_converter_callback,
)
from janus_core.helpers.utils import dict_paths_to_strs

app = Typer()


@app.command(help="Calculate equation of state.")
@use_config(yaml_converter_callback)
def eos(
    # pylint: disable=too-many-arguments,too-many-locals
    # numpydoc ignore=PR02
    ctx: Context,
    struct: StructPath,
    struct_name: Annotated[
        Optional[str],
        Option(help="Name of structure name."),
    ] = None,
    min_lattice: Annotated[
        float, Option(help="Minimum lattice constant scale factor.")
    ] = 0.95,
    max_lattice: Annotated[
        float, Option(help="Maximum lattice constant scale factor.")
    ] = 1.05,
    n_lattice: Annotated[int, Option(help="Number of lattice constants.")] = 5,
    minimize: Annotated[
        bool, Option(help="Whether to minimize structure before calculations.")
    ] = False,
    fmax: Annotated[
        float, Option(help="Maximum force for optimization convergence.")
    ] = 0.1,
    minimize_kwargs: MinimizeKwargs = None,
    arch: Architecture = "mace_mp",
    device: Device = "cpu",
    read_kwargs: ReadKwargs = None,
    calc_kwargs: CalcKwargs = None,
    file_prefix: Annotated[
        Optional[Path],
        Option(
            help=(
                """
                Prefix for output filenames. Default is inferred from structure name,
                or chemical formula.
                """
            ),
        ),
    ] = None,
    log: LogPath = "eos.log",
    summary: Summary = "eos_summary.yml",
):
    """
    Calculate equation of state and write out results.

    Parameters
    ----------
    ctx : Context
        Typer (Click) Context. Automatically set.
    struct : Path
        Path of structure to simulate.
    struct_name : Optional[str]
        Name of structure to simulate. Default is inferred from filepath or chemical
        formula.
    min_lattice : float
        Minimum lattice constant scale factor. Default is 0.95.
    max_lattice : float
        Maximum lattice constant scale factor. Default is 1.05.
    n_lattice : int
        Number of lattice constants to use. Default is 5.
    minimize : bool
        Whether to minimize structure before calculations. Default is False.
    fmax : float
        Set force convergence criteria for optimizer in units eV/Å.
        Default is 0.1.
    minimize_kwargs : Optional[dict[str, Any]]
        Other keyword arguments to pass to geometry optimizer. Default is {}.
    arch : Optional[str]
        MLIP architecture to use for geometry optimization.
        Default is "mace_mp".
    device : Optional[str]
        Device to run model on. Default is "cpu".
    read_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ase.io.read. Default is {}.
    calc_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to the selected calculator. Default is {}.
    file_prefix : Optional[PathLike]
        Prefix for output filenames. Default is inferred from structure name, or
        chemical formula.
    log : Optional[Path]
        Path to write logs to. Default is "eos.log".
    summary : Path
        Path to save summary of inputs and start/end time. Default is eos.yml.
    config : Path
        Path to yaml configuration file to define the above options. Default is None.
    """
    # Check options from configuration file are all valid
    check_config(ctx)

    [read_kwargs, calc_kwargs, minimize_kwargs] = parse_typer_dicts(
        [read_kwargs, calc_kwargs, minimize_kwargs]
    )

    # Set up single point calculator
    s_point = SinglePoint(
        struct_path=struct,
        struct_name=struct_name,
        architecture=arch,
        device=device,
        read_kwargs=read_kwargs,
        calc_kwargs=calc_kwargs,
        log_kwargs={"filename": log, "filemode": "w"},
    )

    log_kwargs = {"filename": log, "filemode": "a"}

    # Check fmax option not duplicated
    if "fmax" in minimize_kwargs:
        raise ValueError("'fmax' must be passed through the --fmax option")
    minimize_kwargs["fmax"] = fmax

    # Dictionary of inputs for eos
    eos_kwargs = {
        "struct": s_point.struct,
        "struct_name": s_point.struct_name,
        "min_lattice": min_lattice,
        "max_lattice": max_lattice,
        "n_lattice": n_lattice,
        "minimize": minimize,
        "minimize_kwargs": minimize_kwargs,
        "file_prefix": file_prefix,
        "log_kwargs": log_kwargs,
    }

    # Store inputs for yaml summary
    inputs = eos_kwargs.copy()

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

    # Save summary information before calculations begin
    start_summary(command="eos", summary=summary, inputs=inputs)

    # Calculate equation of state
    calc_eos(**eos_kwargs)

    # Time after calculations have finished
    end_summary(summary)
