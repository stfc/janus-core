"""Set up NEB commandline interface."""

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
    InterpolationKwargs,
    LogPath,
    MinimizeKwargs,
    ModelPath,
    NebKwargs,
    NebOptKwargs,
    ReadKwargsLast,
    StructPath,
    Summary,
    WriteKwargs,
)
from janus_core.cli.utils import dict_paths_to_strs, yaml_converter_callback

app = Typer()


@app.command()
@use_config(yaml_converter_callback)
def neb(
    # numpydoc ignore=PR02
    ctx: Context,
    init_struct: StructPath | None = None,
    final_struct: StructPath | None = None,
    band_structs: StructPath | None = None,
    neb_class: Annotated[
        str | None,
        Option(help="Name of ASE NEB class to use."),
    ] = "NEB",
    n_images: Annotated[int, Option(help="Number of images to use in NEB.")] = 15,
    write_band: Annotated[
        bool,
        Option(help="Whether to write out all band images after optimization."),
    ] = False,
    write_kwargs: WriteKwargs = None,
    neb_kwargs: NebKwargs = None,
    interpolator: Annotated[
        str | None,
        Option(
            click_type=Choice(["ase", "pymatgen"]),
            help="Choice of interpolation strategy.",
        ),
    ] = "ase",
    interpolator_kwargs: InterpolationKwargs = None,
    optimizer: Annotated[
        str | None,
        Option(help="Name of ASE NEB optimizer to use."),
    ] = "NEBOptimizer",
    fmax: Annotated[float, Option(help="Maximum force for NEB optimizer.")] = 0.1,
    steps: Annotated[
        int, Option(help="Maximum number of steps for optimization.")
    ] = 100,
    optimizer_kwargs: NebOptKwargs = None,
    plot_band: Annotated[
        bool,
        Option(help="Whether to plot and save NEB band."),
    ] = False,
    minimize: Annotated[
        bool, Option(help=" Whether to minimize initial and final structures.")
    ] = False,
    minimize_kwargs: MinimizeKwargs = None,
    arch: Architecture = "mace_mp",
    device: Device = "cpu",
    model_path: ModelPath = None,
    read_kwargs: ReadKwargsLast = None,
    calc_kwargs: CalcKwargs = None,
    file_prefix: Annotated[
        Path | None,
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
    ctx
        Typer (Click) Context. Automatically set.

    init_struct
        Path of initial structure for Nudged Elastic Band method. Required if
        `band_structs` is None. Default is None.
    final_struct
        Path of final structure for Nudged Elastic Band method. Required if
        `band_structs` is None. Default is None.
    band_structs
        Path of band images to optimize, skipping interpolation between the
        initial and final structures. Sets `interpolator` to None.
    neb_class
        Nudged Elastic Band class to use. Default is ase.mep.NEB.
    n_images
        Number of images to use in NEB. Default is 15.
    write_band
        Whether to write out all band images after optimization. Default is False.
    write_kwargs
        Keyword arguments to pass to ase.io.write when writing images.
    neb_kwargs
        Keyword arguments to pass to neb_class. Default is {}.
    interpolator
        Choice of interpolation strategy. Default is "ase" if using `init_struct` and
        `final_struct`, or None if using `band_structs`.
    interpolator_kwargs
        Keyword arguments to pass to interpolator. Default is {"method": "idpp"} for
        "ase" interpolator, or {"interpolate_lattices": False, "autosort_tol", 0.5}
        for "pymatgen".
    optimizer
        Optimizer to apply to NEB object. Default is NEBOptimizer.
    fmax
        Maximum force for NEB optimizer. Default is 0.1.
    steps
        Maximum number of steps to optimize NEB. Default is 100.
    optimizer_kwargs
        Keyword arguments to pass to optimizer. Deault is {}.
    plot_band
        Whether to plot and save NEB band. Default is False.
    minimize
        Whether to perform geometry optimisation on initial and final structures.
        Default is False.
    minimize_kwargs
        Keyword arguments to pass to geometry optimizer. Default is {}.
    arch
        MLIP architecture to use for Nudged Elastic Band method. Default is
        "mace_mp".
    device
        Device to run MLIP model on. Default is "cpu".
    model_path
        Path to MLIP model. Default is `None`.
    read_kwargs
        Keyword arguments to pass to ase.io.read. By default, read_kwargs["index"]
        is -1 if using `init_struct` and `final_struct`, or ":" for `band_structs`.
    calc_kwargs
        Keyword arguments to pass to the selected calculator. Default is {}.
    file_prefix
        Prefix for output filenames. Default is inferred from the intial structure
        name, or chemical formula of the intial structure.
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
    from janus_core.calculations.neb import NEB
    from janus_core.cli.utils import (
        carbon_summary,
        check_config,
        end_summary,
        parse_typer_dicts,
        save_struct_calc,
        start_summary,
    )
    from janus_core.helpers.janus_types import Interpolators

    # Check options from configuration file are all valid
    check_config(ctx)

    [
        write_kwargs,
        neb_kwargs,
        interpolator_kwargs,
        optimizer_kwargs,
        minimize_kwargs,
        read_kwargs,
        calc_kwargs,
    ] = parse_typer_dicts(
        [
            write_kwargs,
            neb_kwargs,
            interpolator_kwargs,
            optimizer_kwargs,
            minimize_kwargs,
            read_kwargs,
            calc_kwargs,
        ]
    )

    if band_structs:
        if init_struct or final_struct:
            raise ValueError(
                "Initial and final structures cannot be specified in addition to the "
                "band structures"
            )
        interpolator = None

    if not band_structs and interpolator not in get_args(Interpolators):
        raise ValueError(f"Fit type must be one of: {get_args(Interpolators)}")

    log_kwargs = {"filemode": "w"}
    if log:
        log_kwargs["filename"] = log

    # Dictionary of inputs for NEB class
    neb_inputs = {
        "init_struct_path": init_struct,
        "final_struct_path": final_struct,
        "band_path": band_structs,
        "arch": arch,
        "device": device,
        "model_path": model_path,
        "read_kwargs": read_kwargs,
        "calc_kwargs": calc_kwargs,
        "attach_logger": True,
        "log_kwargs": log_kwargs,
        "track_carbon": tracker,
        "neb_class": neb_class,
        "n_images": n_images,
        "write_band": write_band,
        "write_kwargs": write_kwargs,
        "neb_kwargs": neb_kwargs,
        "interpolator": interpolator,
        "interpolator_kwargs": interpolator_kwargs,
        "optimizer": optimizer,
        "fmax": fmax,
        "steps": steps,
        "optimizer_kwargs": optimizer_kwargs,
        "plot_band": plot_band,
        "minimize": minimize,
        "minimize_kwargs": minimize_kwargs,
        "file_prefix": file_prefix,
    }

    # Initialise NEB
    neb = NEB(**neb_inputs)

    # Set summary and log files
    summary = neb._build_filename("neb-summary.yml", filename=summary).absolute()
    log = neb.log_kwargs["filename"]

    # Store inputs for yaml summary
    inputs = neb_inputs.copy()

    del inputs["init_struct_path"]
    del inputs["final_struct_path"]
    del inputs["band_path"]

    # Add structure, MLIP information, and log to inputs
    save_struct_calc(
        inputs=inputs,
        struct=neb.struct,
        struct_path=init_struct,
        arch=arch,
        device=device,
        model_path=model_path,
        read_kwargs=read_kwargs,
        calc_kwargs=calc_kwargs,
        log=log,
    )
    if band_structs:
        inputs["band_structs"] = inputs.pop("traj")
    else:
        inputs["init_struct"] = inputs.pop("struct")
        inputs["final_struct"] = {
            "n_atoms": len(neb.final_struct),
            "struct_path": final_struct,
            "formula": neb.final_struct.get_chemical_formula(),
        }

    # Convert all paths to strings in inputs nested dictionary
    dict_paths_to_strs(inputs)

    # Save summary information before calculations begin
    start_summary(command="neb", summary=summary, inputs=inputs)

    # Run equation of state calculations
    neb.run()

    # Save carbon summary
    if tracker:
        carbon_summary(summary=summary, log=log)

    # Time after calculations have finished
    end_summary(summary)
