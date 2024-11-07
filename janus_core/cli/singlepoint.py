# ruff: noqa: I002, FA100
"""Set up singlepoint commandline interface."""

# Issues with future annotations and typer
# c.f. https://github.com/maxb2/typer-config/issues/295
# from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

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
def singlepoint(
    # numpydoc ignore=PR02
    ctx: Context,
    struct: StructPath,
    arch: Architecture = "mace_mp",
    device: Device = "cpu",
    model_path: ModelPath = None,
    properties: Annotated[
        Optional[list[str]],
        Option(
            help=(
                "Properties to calculate. If not specified, 'energy', 'forces' "
                "and 'stress' will be returned."
            ),
        ),
    ] = None,
    out: Annotated[
        Optional[Path],
        Option(
            help=(
                "Path to save structure with calculated results. Default is inferred "
                "from name of structure file."
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
    ctx : Context
        Typer (Click) Context. Automatically set.
    struct : Path
        Path of structure to simulate.
    arch : Optional[str]
        MLIP architecture to use for single point calculations.
        Default is "mace_mp".
    device : Optional[str]
        Device to run model on. Default is "cpu".
    model_path : Optional[str]
        Path to MLIP model. Default is `None`.
    properties : Optional[list[str]]
        Physical properties to calculate. Default is ("energy", "forces", "stress").
    out : Optional[Path]
        Path to save structure with calculated results. Default is inferred from name
        of the structure file.
    read_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ase.io.read. By default,
            read_kwargs["index"] is ":".
    calc_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to the selected calculator. Default is {}.
    write_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ase.io.write when saving results. Default is {}.
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
    from janus_core.calculations.single_point import SinglePoint
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

    # Check filename for results not duplicated
    if "filename" in write_kwargs:
        raise ValueError("'filename' must be passed through the --out option")
    if out:
        write_kwargs["filename"] = out

    log_kwargs = {"filemode": "w"}
    if log:
        log_kwargs["filename"] = log

    singlepoint_kwargs = {
        "struct_path": struct,
        "properties": properties,
        "write_kwargs": write_kwargs,
        "write_results": True,
        "arch": arch,
        "device": device,
        "model_path": model_path,
        "read_kwargs": read_kwargs,
        "calc_kwargs": calc_kwargs,
        "attach_logger": True,
        "log_kwargs": log_kwargs,
        "track_carbon": tracker,
    }

    # Initialise singlepoint structure and calculator
    s_point = SinglePoint(**singlepoint_kwargs)

    # Store inputs for yaml summary
    summary = s_point._build_filename(
        "singlepoint-summary.yml", filename=summary
    ).absolute()
    log = s_point.log_kwargs["filename"]

    # Store inputs for yaml summary
    inputs = singlepoint_kwargs.copy()

    # Add structure, MLIP information, and log to inputs
    save_struct_calc(
        inputs=inputs,
        struct=s_point.struct,
        struct_path=struct,
        arch=arch,
        device=device,
        model_path=model_path,
        read_kwargs=read_kwargs,
        calc_kwargs=calc_kwargs,
        log=log,
    )

    # Save summary information before singlepoint calculation begins
    start_summary(command="singlepoint", summary=summary, inputs=inputs)

    # Run singlepoint calculation
    s_point.run()

    # Save carbon summary
    if tracker:
        carbon_summary(summary=summary, log=log)

    # Save time after simulation has finished
    end_summary(summary)
