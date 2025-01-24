"""Set up MLIP descriptors commandline interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from typer import Context, Option, Typer
from typer_config import use_config

from janus_core.cli.types import (
    Architecture,
    CalcKwargs,
    Device,
    LogPath,
    ModelPath,
    ReadKwargsAll,
    StructPath,
    Summary,
    WriteKwargs,
)
from janus_core.cli.utils import yaml_converter_callback

app = Typer()


@app.command()
@use_config(yaml_converter_callback)
def descriptors(
    # numpydoc ignore=PR02
    ctx: Context,
    struct: StructPath,
    invariants_only: Annotated[
        bool,
        Option(help="Only calculate invariant descriptors."),
    ] = True,
    calc_per_element: Annotated[
        bool,
        Option(help="Calculate mean descriptors for each element."),
    ] = False,
    calc_per_atom: Annotated[
        bool,
        Option(help="Calculate descriptors for each atom."),
    ] = False,
    arch: Architecture = "mace_mp",
    device: Device = "cpu",
    model_path: ModelPath = None,
    out: Annotated[
        Path | None,
        Option(
            help=(
                "Path to save structure with calculated descriptors. Default is "
                "inferred from name of structure file."
            ),
        ),
    ] = None,
    read_kwargs: ReadKwargsAll = None,
    calc_kwargs: CalcKwargs = None,
    write_kwargs: WriteKwargs = None,
    log: LogPath = None,
    tracker: Annotated[
        bool, Option(help="Whether to save carbon emissions of calculation")
    ] = True,
    summary: Summary = None,
) -> None:
    """
    Calculate MLIP descriptors for the given structure(s).

    Parameters
    ----------
    ctx
        Typer (Click) Context. Automatically set.
    struct
        Path of structure to simulate.
    invariants_only
        Whether only the invariant descriptors should be returned. Default is True.
    calc_per_element
        Whether to calculate mean descriptors for each element. Default is False.
    calc_per_atom
        Whether to calculate descriptors for each atom. Default is False.
    arch
        MLIP architecture to use for single point calculations.
        Default is "mace_mp".
    device
        Device to run model on. Default is "cpu".
    model_path
        Path to MLIP model. Default is `None`.
    out
        Path to save structure with calculated results. Default is inferred from name
        of the structure file.
    read_kwargs
        Keyword arguments to pass to ase.io.read. By default,
            read_kwargs["index"] is ":".
    calc_kwargs
        Keyword arguments to pass to the selected calculator. Default is {}.
    write_kwargs
        Keyword arguments to pass to ase.io.write when saving results. Default is {}.
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
    from janus_core.calculations.descriptors import Descriptors
    from janus_core.cli.utils import (
        carbon_summary,
        check_config,
        end_summary,
        parse_typer_dicts,
        save_struct_calc,
        start_summary,
    )

    # Check options from configuration file are all valid
    check_config(ctx)

    [read_kwargs, calc_kwargs, write_kwargs] = parse_typer_dicts(
        [read_kwargs, calc_kwargs, write_kwargs]
    )

    # Check optimized structure path not duplicated
    if "filename" in write_kwargs:
        raise ValueError("'filename' must be passed through the --out option")

    # Set default filname for writing structure with descriptors if not specified
    if out:
        write_kwargs["filename"] = out

    log_kwargs = {"filemode": "w"}
    if log:
        log_kwargs["filename"] = log

    # Dictionary of inputs for Descriptors class
    descriptors_kwargs = {
        "struct_path": struct,
        "arch": arch,
        "device": device,
        "model_path": model_path,
        "read_kwargs": read_kwargs,
        "calc_kwargs": calc_kwargs,
        "attach_logger": True,
        "log_kwargs": log_kwargs,
        "track_carbon": tracker,
        "invariants_only": invariants_only,
        "calc_per_element": calc_per_element,
        "calc_per_atom": calc_per_atom,
        "write_results": True,
        "write_kwargs": write_kwargs,
    }

    # Initialise descriptors
    descript = Descriptors(**descriptors_kwargs)

    # Set summary and log files
    summary = descript._build_filename(
        "descriptors-summary.yml", filename=summary
    ).absolute()
    log = descript.log_kwargs["filename"]

    # Store inputs for yaml summary
    inputs = descriptors_kwargs.copy()

    # Add structure, MLIP information, and log to inputs
    save_struct_calc(
        inputs=inputs,
        struct=descript.struct,
        struct_path=struct,
        arch=arch,
        device=device,
        model_path=model_path,
        read_kwargs=read_kwargs,
        calc_kwargs=calc_kwargs,
        log=log,
    )

    # Save summary information before calculation begins
    start_summary(command="descriptors", summary=summary, inputs=inputs)

    # Calculate descriptors
    descript.run()

    # Save carbon summary
    if tracker:
        carbon_summary(summary=summary, log=log)

    # Time after calculation has finished
    end_summary(summary)
