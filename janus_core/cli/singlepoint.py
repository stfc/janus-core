"""Set up singlepoint commandline interface."""

from pathlib import Path
from typing import Annotated

from typer import Context, Option, Typer
from typer_config import use_config

from janus_core.calculations.single_point import SinglePoint
from janus_core.cli.types import (
    Architecture,
    CalcKwargs,
    Device,
    LogPath,
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


@app.command()
@use_config(yaml_converter_callback)
def singlepoint(
    # pylint: disable=too-many-arguments,too-many-locals,duplicate-code
    # numpydoc ignore=PR02
    ctx: Context,
    struct: StructPath,
    arch: Architecture = "mace_mp",
    device: Device = "cpu",
    model_path: ModelPath = None,
    properties: Annotated[
        list[str],
        Option(
            help=(
                "Properties to calculate. If not specified, 'energy', 'forces' "
                "and 'stress' will be returned."
            ),
        ),
    ] = None,
    out: Annotated[
        Path,
        Option(
            help=(
                "Path to save structure with calculated results. Default is inferred "
                "from name of structure file."
            ),
        ),
    ] = None,
    read_kwargs: ReadKwargs = None,
    calc_kwargs: CalcKwargs = None,
    write_kwargs: WriteKwargs = None,
    log: LogPath = "singlepoint.log",
    summary: Summary = "singlepoint_summary.yml",
):
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
    properties : Optional[str]
        Physical properties to calculate. Default is "energy".
    out : Optional[Path]
        Path to save structure with calculated results. Default is inferred from name
        of the structure file.
    read_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ase.io.read. Default is {}.
    calc_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to the selected calculator. Default is {}.
    write_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ase.io.write when saving results. Default is {}.
    log : Optional[Path]
        Path to write logs to. Default is "singlepoint.log".
    summary : Path
        Path to save summary of inputs and start/end time. Default is
        singlepoint_summary.yml.
    config : Path
        Path to yaml configuration file to define the above options. Default is None.
    """
    # Check options from configuration file are all valid
    check_config(ctx)

    [read_kwargs, calc_kwargs, write_kwargs] = parse_typer_dicts(
        [read_kwargs, calc_kwargs, write_kwargs]
    )

    # Check filename for results not duplicated
    if "filename" in write_kwargs:
        raise ValueError("'filename' must be passed through the --out option")

    # Default filname for saving results determined in SinglePoint if not specified
    if out:
        write_kwargs["filename"] = out

    singlepoint_kwargs = {
        "struct_path": struct,
        "architecture": arch,
        "device": device,
        "model_path": model_path,
        "read_kwargs": read_kwargs,
        "calc_kwargs": calc_kwargs,
        "log_kwargs": {"filename": log, "filemode": "w"},
    }

    # Initialise singlepoint structure and calculator
    s_point = SinglePoint(**singlepoint_kwargs)

    # Store inputs for yaml summary

    # Store only filename as filemode is not set by user
    inputs = {"log": log}

    save_struct_calc(
        inputs, s_point, arch, device, model_path, read_kwargs, calc_kwargs
    )

    inputs["run"] = {
        "properties": properties,
        "write_kwargs": write_kwargs,
    }

    # Convert all paths to strings in inputs nested dictionary
    dict_paths_to_strs(inputs)

    # Save summary information before singlepoint calculation begins
    start_summary(command="singlepoint", summary=summary, inputs=inputs)

    # Run singlepoint calculation
    s_point.run(properties=properties, write_results=True, write_kwargs=write_kwargs)

    # Save time after simulation has finished
    end_summary(summary)
