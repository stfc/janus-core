"""Set up MLIP descriptors commandline interface."""

from pathlib import Path
from typing import Annotated

from typer import Context, Option, Typer
from typer_config import use_config

from janus_core.calculations.descriptors import Descriptors
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
def descriptors(
    # pylint: disable=too-many-arguments,too-many-locals,duplicate-code
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
    arch: Architecture = "mace_mp",
    device: Device = "cpu",
    model_path: ModelPath = None,
    out: Annotated[
        Path,
        Option(
            help=(
                "Path to save structure with calculated descriptors. Default is "
                "inferred from name of structure file."
            ),
        ),
    ] = None,
    read_kwargs: ReadKwargs = None,
    calc_kwargs: CalcKwargs = None,
    write_kwargs: WriteKwargs = None,
    log: LogPath = "descriptors.log",
    summary: Summary = "descriptors_summary.yml",
):
    """
    Calculate MLIP descriptors for the given structure(s).

    Parameters
    ----------
    ctx : Context
        Typer (Click) Context. Automatically set.
    struct : Path
        Path of structure to simulate.
    invariants_only : bool
        Whether only the invariant descriptors should be returned. Default is True.
    calc_per_element : bool
        Whether to calculate mean descriptors for each element. Default is False.
    arch : Optional[str]
        MLIP architecture to use for single point calculations.
        Default is "mace_mp".
    device : Optional[str]
        Device to run model on. Default is "cpu".
    model_path : Optional[str]
        Path to MLIP model. Default is `None`.
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
        Path to write logs to. Default is "descriptors.log".
    summary : Path
        Path to save summary of inputs and start/end time. Default is
        descriptors_summary.yml.
    config : Path
        Path to yaml configuration file to define the above options. Default is None.
    """
    # Check options from configuration file are all valid
    check_config(ctx)

    [read_kwargs, calc_kwargs, write_kwargs] = parse_typer_dicts(
        [read_kwargs, calc_kwargs, write_kwargs]
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

    # Set default filname for writing structure with descriptors if not specified
    if out:
        write_kwargs["filename"] = out
    else:
        write_kwargs["filename"] = f"{s_point.struct_name}-descriptors.extxyz"

    # Dictionary of inputs for optimize function
    descriptors_kwargs = {
        "struct": s_point.struct,
        "invariants_only": invariants_only,
        "calc_per_element": calc_per_element,
        "write_results": True,
        "write_kwargs": write_kwargs,
        "log_kwargs": {"filename": log, "filemode": "a"},
    }

    # Store inputs for yaml summary
    inputs = descriptors_kwargs.copy()

    # Store only filename as filemode is not set by user
    del inputs["log_kwargs"]
    inputs["log"] = log

    save_struct_calc(
        inputs, s_point, arch, device, model_path, read_kwargs, calc_kwargs
    )

    # Convert all paths to strings in inputs nested dictionary
    dict_paths_to_strs(inputs)

    # Save summary information before optimization begins
    start_summary(command="descriptors", summary=summary, inputs=inputs)

    # Run geometry optimization and save output structure
    descript = Descriptors(**descriptors_kwargs)
    descript.run()

    # Time after optimization has finished
    end_summary(summary)
