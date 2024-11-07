# ruff: noqa: I002, FA100
"""Set up eos commandline interface."""

# Issues with future annotations and typer
# c.f. https://github.com/maxb2/typer-config/issues/295
# from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional, get_args

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


@app.command()
@use_config(yaml_converter_callback)
def eos(
    # numpydoc ignore=PR02
    ctx: Context,
    struct: StructPath,
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
    plot_to_file: Annotated[
        bool,
        Option(help="Whether to plot equation of state."),
    ] = False,
    arch: Architecture = "mace_mp",
    device: Device = "cpu",
    model_path: ModelPath = None,
    read_kwargs: ReadKwargsLast = None,
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
    log: LogPath = None,
    tracker: Annotated[
        bool, Option(help="Whether to save carbon emissions of calculation")
    ] = True,
    summary: Summary = None,
) -> None:
    """
    Calculate equation of state and write out results.

    Parameters
    ----------
    ctx : Context
        Typer (Click) Context. Automatically set.
    struct : Path
        Path of structure to simulate.
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
    plot_to_file : bool
        Whether to save plot equation of state to svg. Default is False.
    arch : Optional[str]
        MLIP architecture to use for geometry optimization.
        Default is "mace_mp".
    device : Optional[str]
        Device to run model on. Default is "cpu".
    model_path : Optional[str]
        Path to MLIP model. Default is `None`.
    read_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ase.io.read. By default,
            read_kwargs["index"] is -1.
    calc_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to the selected calculator. Default is {}.
    file_prefix : Optional[PathLike]
        Prefix for output filenames. Default is inferred from structure name, or
        chemical formula.
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
    from janus_core.calculations.eos import EoS
    from janus_core.cli.utils import (
        carbon_summary,
        check_config,
        end_summary,
        parse_typer_dicts,
        save_struct_calc,
        set_read_kwargs_index,
        start_summary,
    )
    from janus_core.helpers.janus_types import EoSNames

    # Check options from configuration file are all valid
    check_config(ctx)

    [read_kwargs, calc_kwargs, minimize_kwargs, write_kwargs] = parse_typer_dicts(
        [read_kwargs, calc_kwargs, minimize_kwargs, write_kwargs]
    )

    if eos_type not in get_args(EoSNames):
        raise ValueError(f"Fit type must be one of: {get_args(EoSNames)}")

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
        "struct_path": struct,
        "arch": arch,
        "device": device,
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
    summary = equation_of_state._build_filename(
        "eos-summary.yml", filename=summary
    ).absolute()
    log = equation_of_state.log_kwargs["filename"]

    # Store inputs for yaml summary
    inputs = eos_kwargs.copy()

    # Add structure, MLIP information, and log to inputs
    save_struct_calc(
        inputs=inputs,
        struct=equation_of_state.struct,
        struct_path=struct,
        arch=arch,
        device=device,
        model_path=model_path,
        read_kwargs=read_kwargs,
        calc_kwargs=calc_kwargs,
        log=log,
    )

    # Save summary information before calculations begin
    start_summary(command="eos", summary=summary, inputs=inputs)

    # Run equation of state calculations
    equation_of_state.run()

    # Save carbon summary
    if tracker:
        carbon_summary(summary=summary, log=log)

    # Time after calculations have finished
    end_summary(summary)
