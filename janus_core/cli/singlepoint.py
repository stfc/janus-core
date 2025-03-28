"""Set up singlepoint commandline interface."""

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
    LogPath,
    ModelPath,
    ReadKwargsAll,
    StructPath,
    Summary,
    WriteKwargs,
)
from janus_core.cli.utils import yaml_converter_callback
from janus_core.helpers.janus_types import Properties

app = Typer()


@app.command()
@use_config(yaml_converter_callback)
def singlepoint(
    # numpydoc ignore=PR02
    ctx: Context,
    struct: StructPath,
    arch: Architecture = "mace_mp",
    device: Device = "cpu",
    model_path: ModelPath = None,
    properties: Annotated[
        list[str] | None,
        Option(
            click_type=Choice(get_args(Properties)),
            help=(
                "Properties to calculate. If not specified, 'energy', 'forces' "
                "and 'stress' will be returned."
            ),
        ),
    ] = None,
    file_prefix: FilePrefix = None,
    out: Annotated[
        Path | None,
        Option(
            help=(
                "Path to save structure with calculated results. Default is inferred "
                "from `file_prefix`."
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
    Perform single point calculations and save to file.

    Parameters
    ----------
    ctx
        Typer (Click) Context. Automatically set.
    struct
        Path of structure to simulate.
    arch
        MLIP architecture to use for single point calculations.
        Default is "mace_mp".
    device
        Device to run model on. Default is "cpu".
    model_path
        Path to MLIP model. Default is `None`.
    properties
        Physical properties to calculate. Default is ("energy", "forces", "stress").
    file_prefix
        Prefix for output files, including directories. Default directory is
        ./janus_results, and default filename prefix is inferred from the input
        stucture filename.
    out
        Path to save structure with calculated results. Default is inferred from
        `file_prefix`.
    read_kwargs
        Keyword arguments to pass to ase.io.read. By default,
            read_kwargs["index"] is ":".
    calc_kwargs
        Keyword arguments to pass to the selected calculator. Default is {}.
    write_kwargs
        Keyword arguments to pass to ase.io.write when saving results. Default is {}.
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
    from janus_core.calculations.single_point import SinglePoint
    from janus_core.cli.utils import (
        carbon_summary,
        check_config,
        end_summary,
        get_config,
        get_struct_info,
        parse_typer_dicts,
        start_summary,
    )

    # Check options from configuration file are all valid
    check_config(ctx)

    [read_kwargs, calc_kwargs, write_kwargs] = parse_typer_dicts(
        [read_kwargs, calc_kwargs, write_kwargs]
    )

    # Check filename for results not duplicated
    if "filename" in write_kwargs:
        raise ValueError("'filename' must be passed through the --out option")

    # Set initial config
    all_kwargs = {
        "read_kwargs": read_kwargs.copy(),
        "calc_kwargs": calc_kwargs.copy(),
        "write_kwargs": write_kwargs.copy(),
    }
    config = get_config(params=ctx.params, all_kwargs=all_kwargs)

    if out:
        write_kwargs["filename"] = out

    log_kwargs = {"filemode": "w"}
    if log:
        log_kwargs["filename"] = log

    singlepoint_kwargs = {
        "struct": struct,
        "properties": properties,
        "write_kwargs": write_kwargs,
        "write_results": True,
        "arch": arch,
        "device": device,
        "file_prefix": file_prefix,
        "model_path": model_path,
        "read_kwargs": read_kwargs,
        "calc_kwargs": calc_kwargs,
        "attach_logger": True,
        "log_kwargs": log_kwargs,
        "track_carbon": tracker,
        "enable_progress_bar": True,
    }

    # Initialise singlepoint structure and calculator
    s_point = SinglePoint(**singlepoint_kwargs)

    # Store inputs for yaml summary
    summary = s_point._build_filename("singlepoint-summary.yml", filename=summary)
    log = s_point.log_kwargs["filename"]

    # Add structure, MLIP information, and log to info
    info = get_struct_info(
        struct=s_point.struct,
        struct_path=struct,
    )

    output_files = s_point.output_files

    # Save summary information before singlepoint calculation begins
    start_summary(
        command="singlepoint",
        summary=summary,
        info=info,
        config=config,
        output_files=output_files,
    )

    # Run singlepoint calculation
    s_point.run()

    # Save carbon summary
    if tracker:
        carbon_summary(summary=summary, log=log)

    # Save time after simulation has finished
    end_summary(summary)
