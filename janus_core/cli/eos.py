"""Set up eos commandline interface."""

from pathlib import Path
from typing import Annotated, Optional, get_args

from typer import Context, Option, Typer
from typer_config import use_config

from janus_core.calculations.eos import EoS
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
from janus_core.helpers.janus_types import EoSNames
from janus_core.helpers.utils import dict_paths_to_strs

app = Typer()


@app.command()
@use_config(yaml_converter_callback)
def eos(
    # pylint: disable=too-many-arguments,too-many-locals,duplicate-code
    # numpydoc ignore=PR02
    ctx: Context,
    struct: StructPath,
    struct_name: Annotated[
        Optional[str],
        Option(help="Name of structure name."),
    ] = None,
    min_volume: Annotated[float, Option(help="Minimum volume scale factor.")] = 0.95,
    max_volume: Annotated[float, Option(help="Maximum volume scale factor.")] = 1.05,
    n_volumes: Annotated[int, Option(help="Number of volumes.")] = 7,
    eos_type: Annotated[
        str, Option(help="Type of fit for equation of state.")
    ] = "birchmurnaghan",
    minimize: Annotated[
        bool, Option(help="Whether to minimize initial structure before calculations.")
    ] = True,
    minimize_all: Annotated[
        bool,
        Option(help="Whether to minimize all generated structures for calculations."),
    ] = False,
    fmax: Annotated[
        float, Option(help="Maximum force for optimization convergence.")
    ] = 0.1,
    minimize_kwargs: MinimizeKwargs = None,
    write_structures: Annotated[
        bool,
        Option(help="Whether to write out all genereated structures."),
    ] = False,
    write_kwargs: WriteKwargs = None,
    arch: Architecture = "mace_mp",
    device: Device = "cpu",
    model_path: ModelPath = None,
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
    min_volume : float
        Minimum volume scale factor. Default is 0.95.
    max_volume : float
        Maximum volume scale factor. Default is 1.05.
    n_volumes : int
        Number of volumes to use. Default is 7.
    eos_type : Optional[str]
        Type of fit for equation of state. Default is "birchmurnaghan".
    minimize : bool
        Whether to minimize initial structure before calculations. Default is True.
    minimize_all : bool
        Whether to optimize geometry for all generated structures. Default is False.
    fmax : float
        Set force convergence criteria for optimizer in units eV/Å.
        Default is 0.1.
    minimize_kwargs : Optional[dict[str, Any]]
        Other keyword arguments to pass to geometry optimizer. Default is {}.
    write_structures : bool
        True to write out all genereated structures. Default is False.
    write_kwargs : Optional[dict[str, Any]],
        Keyword arguments to pass to ase.io.write to save generated structures.
        Default is {}.
    arch : Optional[str]
        MLIP architecture to use for geometry optimization.
        Default is "mace_mp".
    device : Optional[str]
        Device to run model on. Default is "cpu".
    model_path : Optional[str]
        Path to MLIP model. Default is `None`.
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

    [read_kwargs, calc_kwargs, minimize_kwargs, write_kwargs] = parse_typer_dicts(
        [read_kwargs, calc_kwargs, minimize_kwargs, write_kwargs]
    )

    if not eos_type in get_args(EoSNames):
        raise ValueError(f"Fit type must be one of: {get_args(EoSNames)}")

    # Set up single point calculator
    s_point = SinglePoint(
        struct_path=struct,
        struct_name=struct_name,
        architecture=arch,
        device=device,
        model_path=model_path,
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
        "min_volume": min_volume,
        "max_volume": max_volume,
        "n_volumes": n_volumes,
        "eos_type": eos_type,
        "minimize": minimize,
        "minimize_all": minimize_all,
        "minimize_kwargs": minimize_kwargs,
        "write_structures": write_structures,
        "write_kwargs": write_kwargs,
        "file_prefix": file_prefix,
        "log_kwargs": log_kwargs,
    }

    # Store inputs for yaml summary
    inputs = eos_kwargs.copy()

    # Store only filename as filemode is not set by user
    del inputs["log_kwargs"]
    inputs["log"] = log

    save_struct_calc(
        inputs, s_point, arch, device, model_path, read_kwargs, calc_kwargs
    )

    # Convert all paths to strings in inputs nested dictionary
    dict_paths_to_strs(inputs)

    # Save summary information before calculations begin
    start_summary(command="eos", summary=summary, inputs=inputs)

    # Calculate equation of state
    equation_of_state = EoS(**eos_kwargs)
    equation_of_state.run()

    # Time after calculations have finished
    end_summary(summary)
