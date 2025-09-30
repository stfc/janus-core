"""Set up phonons commandline interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from typer import Context, Option, Typer
from typer_config import use_config

from janus_core.cli.types import (
    Architecture,
    CalcKwargs,
    Device,
    DisplacementKwargs,
    DoSKwargs,
    FilePrefix,
    LogPath,
    MinimizeKwargs,
    Model,
    ModelPath,
    PDoSKwargs,
    ProgressBar,
    ReadKwargsLast,
    StructPath,
    Summary,
    Tracker,
)
from janus_core.cli.utils import deprecated_option, yaml_converter_callback

app = Typer()


@app.command()
@use_config(yaml_converter_callback, param_help="Path to configuration file.")
def phonons(
    # numpydoc ignore=PR02
    ctx: Context,
    # Required
    arch: Architecture,
    struct: StructPath,
    # Calculation
    supercell: Annotated[
        str,
        Option(
            help="Supercell matrix, in the Phonopy style. Must be passed as a string "
            "in one of three forms: single integer ('2'), which specifies all "
            "diagonal elements; three integers ('1 2 3'), which specifies each "
            "individual diagonal element; or nine values ('1 2 3 4 5 6 7 8 9'), "
            "which specifies all elements, filling the matrix row-wise.",
            rich_help_panel="Calculation",
        ),
    ] = "2 2 2",
    displacement: Annotated[
        float,
        Option(
            help="Displacement for force constants calculation, in A.",
            rich_help_panel="Calculation",
        ),
    ] = 0.01,
    displacement_kwargs: DisplacementKwargs = None,
    mesh: Annotated[
        tuple[int, int, int],
        Option(help="Mesh numbers along a, b, c axes.", rich_help_panel="Calculation"),
    ] = (10, 10, 10),
    bands: Annotated[
        bool,
        Option(
            help="Whether to compute band structure.", rich_help_panel="Calculation"
        ),
    ] = False,
    n_qpoints: Annotated[
        int,
        Option(
            help=(
                "Number of q-points to sample along generated path, including end "
                "points. Unused if `qpoint_file` is specified"
            ),
            rich_help_panel="Calculation",
        ),
    ] = 51,
    qpoint_file: Annotated[
        Path | None,
        Option(
            help=(
                "Path to yaml file with info to generate a path of q-points for band "
                "structure."
            ),
            rich_help_panel="Calculation",
        ),
    ] = None,
    symmetrize: Annotated[
        bool,
        Option(
            help="Whether to symmetrize force constants.", rich_help_panel="Calculation"
        ),
    ] = False,
    minimize: Annotated[
        bool,
        Option(
            help="Whether to minimize structure before calculations.",
            rich_help_panel="Calculation",
        ),
    ] = False,
    fmax: Annotated[
        float,
        Option(
            help="Maximum force for optimization convergence.",
            rich_help_panel="Calculation",
        ),
    ] = 0.1,
    minimize_kwargs: MinimizeKwargs = None,
    force_consts_to_hdf5: Annotated[
        bool,
        Option(
            help="Whether to save force constants in hdf5.",
            rich_help_panel="Calculation",
            callback=deprecated_option,
            hidden=True,
        ),
    ] = None,
    hdf5: Annotated[
        bool,
        Option(
            help="Whether to save force constants and bands in hdf5.",
            rich_help_panel="Calculation",
        ),
    ] = True,
    plot_to_file: Annotated[
        bool,
        Option(
            help="Whether to plot band structure and/or dos/pdos when calculated.",
            rich_help_panel="Calculation",
        ),
    ] = False,
    write_full: Annotated[
        bool,
        Option(
            help=(
                "Whether to write eigenvectors, group velocities, etc. to bands file."
            ),
            rich_help_panel="Calculation",
        ),
    ] = True,
    # DOS
    dos: Annotated[
        bool,
        Option(help="Whether to calculate the DOS.", rich_help_panel="DOS"),
    ] = False,
    dos_kwargs: DoSKwargs = None,
    # PDOS
    pdos: Annotated[
        bool,
        Option(help="Whether to calculate the PDOS.", rich_help_panel="PDOS"),
    ] = False,
    pdos_kwargs: PDoSKwargs = None,
    # Thermal properties
    thermal: Annotated[
        bool,
        Option(
            help="Whether to calculate thermal properties.",
            rich_help_panel="Thermal properties",
        ),
    ] = False,
    temp_min: Annotated[
        float,
        Option(
            help="Start temperature for thermal properties calculations, in K.",
            rich_help_panel="Thermal properties",
        ),
    ] = 0.0,
    temp_max: Annotated[
        float,
        Option(
            help="End temperature for thermal properties calculations, in K.",
            rich_help_panel="Thermal properties",
        ),
    ] = 1000.0,
    temp_step: Annotated[
        float,
        Option(
            help="Temperature step for thermal properties calculations, in K.",
            rich_help_panel="Thermal properties",
        ),
    ] = 50,
    # MLIP Calculator
    device: Device = "cpu",
    model: Model = None,
    model_path: ModelPath = None,
    calc_kwargs: CalcKwargs = None,
    # Strucuture I/O
    file_prefix: FilePrefix = None,
    read_kwargs: ReadKwargsLast = None,
    # Logging/summary
    log: LogPath = None,
    tracker: Tracker = True,
    summary: Summary = None,
    progress_bar: ProgressBar = True,
) -> None:
    """
    Perform phonon calculations and write out results.

    Parameters
    ----------
    ctx
        Typer (Click) Context. Automatically set.
    arch
        MLIP architecture to use for phonon calculations.
    struct
        Path of structure to simulate.
    supercell
        Supercell matrix, in the Phonopy style. Must be passed as a string in one of
        three forms: single integer ('2'), which specifies all diagonal elements;
        three integers ('1 2 3'), which specifies each individual diagonal element;
        or nine values ('1 2 3 4 5 6 7 8 9'), which specifies all elements, filling the
        matrix row-wise.
    displacement
        Displacement for force constants calculation, in A. Default is 0.01.
    displacement_kwargs
        Keyword arguments to pass to generate_displacements. Default is {}.
    mesh
        Mesh for sampling. Default is (10, 10, 10).
    bands
        Whether to calculate and save the band structure. Default is False.
    n_qpoints
        Number of q-points to sample along generated path, including end points.
        Unused if `qpoint_file` is specified. Default is 51.
    qpoint_file
        Path to yaml file with info to generate a path of q-points for band structure.
        Default is None.
    symmetrize
        Whether to symmetrize force constants. Default is False.
    minimize
        Whether to minimize structure before calculations. Default is False.
    fmax
        Set force convergence criteria for optimizer in units eV/Å.
        Default is 0.1.
    minimize_kwargs
        Other keyword arguments to pass to geometry optimizer. Default is {}.
    force_consts_to_hdf5
        Deprecated. Please use `hdf5`.
    hdf5
        Whether to save force constants and bands in hdf5 format.
        Default is True.
    plot_to_file
        Whether to plot. Default is False.
    write_full
        Whether to maximize information written in various output files.
        Default is True.
    dos
        Whether to calculate and save the DOS. Default is False.
    dos_kwargs
        Other keyword arguments to pass to run_total_dos. Default is {}.
    pdos
        Whether to calculate and save the PDOS. Default is False.
    pdos_kwargs
        Other keyword arguments to pass to run_projected_dos. Default is {}.
    thermal
        Whether to calculate thermal properties. Default is False.
    temp_min
        Start temperature for thermal calculations, in K. Unused if `thermal` is False.
        Default is 0.0.
    temp_max
        End temperature for thermal calculations, in K. Unused if `thermal` is False.
        Default is 1000.0.
    temp_step
        Temperature step for thermal calculations, in K. Unused if `thermal` is False.
        Default is 50.0.
    device
        Device to run model on. Default is "cpu".
    model
        Path to MLIP model or name of model. Default is `None`.
    model_path
        Deprecated. Please use `model`.
    calc_kwargs
        Keyword arguments to pass to the selected calculator. Default is {}.
    file_prefix
        Prefix for output files, including directories. Default directory is
        ./janus_results, and default filename prefix is inferred from the input
        stucture filename.
    read_kwargs
        Keyword arguments to pass to ase.io.read. By default,
            read_kwargs["index"] is 0.
    log
        Path to write logs to. Default is inferred from `file_prefix`.
    tracker
        Whether to save carbon emissions of calculation in log file and summary.
        Default is True.
    summary
        Path to save summary of inputs, start/end time, and carbon emissions. Default
        is inferred from `file_prefix`.
    progress_bar
        Whether to show progress bar.
    config
        Path to yaml configuration file to define the above options. Default is None.
    """
    from janus_core.calculations.phonons import Phonons
    from janus_core.cli.utils import (
        carbon_summary,
        check_config,
        end_summary,
        get_config,
        get_struct_info,
        parse_typer_dicts,
        set_read_kwargs_index,
        start_summary,
    )

    # Check options from configuration file are all valid
    check_config(ctx)

    (
        displacement_kwargs,
        read_kwargs,
        calc_kwargs,
        minimize_kwargs,
        dos_kwargs,
        pdos_kwargs,
    ) = parse_typer_dicts(
        [
            displacement_kwargs,
            read_kwargs,
            calc_kwargs,
            minimize_kwargs,
            dos_kwargs,
            pdos_kwargs,
        ]
    )

    # Set initial config
    all_kwargs = {
        "displacement_kwargs": displacement_kwargs.copy(),
        "read_kwargs": read_kwargs.copy(),
        "calc_kwargs": calc_kwargs.copy(),
        "minimize_kwargs": minimize_kwargs.copy(),
        "dos_kwargs": dos_kwargs.copy(),
        "pdos_kwargs": pdos_kwargs.copy(),
    }
    config = get_config(params=ctx.params, all_kwargs=all_kwargs)

    # Read only first structure by default and ensure only one image is read
    set_read_kwargs_index(read_kwargs)

    # Check fmax option not duplicated
    if "fmax" in minimize_kwargs:
        raise ValueError("'fmax' must be passed through the --fmax option")
    minimize_kwargs["fmax"] = fmax

    try:
        supercell = [int(x) for x in supercell.split()]
    except ValueError as exc:
        raise ValueError(
            "Please pass lattice vectors as integers in the form '1 2 3'"
        ) from exc

    supercell_length = len(supercell)
    if supercell_length == 1:
        supercell = supercell[0]
    elif supercell_length not in [3, 9]:
        raise ValueError(
            "Please pass lattice vectors as space-separated integers in quotes. "
            "For example, '1 2 3'."
        )

    calcs = []
    if bands:
        calcs.append("bands")
    if thermal:
        calcs.append("thermal")
    if dos:
        calcs.append("dos")
    if pdos:
        calcs.append("pdos")

    log_kwargs = {"filemode": "w"}
    if log:
        log_kwargs["filename"] = log

    # Dictionary of inputs for Phonons class
    phonons_kwargs = {
        "struct": struct,
        "arch": arch,
        "device": device,
        "model": model,
        "model_path": model_path,
        "read_kwargs": read_kwargs,
        "calc_kwargs": calc_kwargs,
        "attach_logger": True,
        "log_kwargs": log_kwargs,
        "track_carbon": tracker,
        "calcs": calcs,
        "supercell": supercell,
        "displacement": displacement,
        "displacement_kwargs": displacement_kwargs,
        "mesh": mesh,
        "symmetrize": symmetrize,
        "minimize": minimize,
        "minimize_kwargs": minimize_kwargs,
        "n_qpoints": n_qpoints,
        "qpoint_file": qpoint_file,
        "dos_kwargs": dos_kwargs,
        "pdos_kwargs": pdos_kwargs,
        "temp_min": temp_min,
        "temp_max": temp_max,
        "temp_step": temp_step,
        "hdf5": hdf5,
        "plot_to_file": plot_to_file,
        "write_results": True,
        "write_full": write_full,
        "file_prefix": file_prefix,
        "enable_progress_bar": progress_bar,
    }

    # Initialise phonons
    phonon = Phonons(**phonons_kwargs)

    # Set summary and log files
    summary = phonon._build_filename("phonons-summary.yml", filename=summary)
    log = phonon.log_kwargs["filename"]

    # Add structure, MLIP information, and log to info
    info = get_struct_info(
        struct=phonon.struct,
        struct_path=struct,
    )

    output_files = phonon.output_files

    # Save summary information before calculations begin
    start_summary(
        command="phonons",
        summary=summary,
        info=info,
        config=config,
        output_files=output_files,
    )

    # Run phonon calculations
    phonon.run()

    # Save carbon summary
    if tracker:
        carbon_summary(summary=summary, log=log)

    # Time after calculations have finished
    end_summary(summary)
