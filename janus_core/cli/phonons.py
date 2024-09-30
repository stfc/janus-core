"""Set up phonons commandline interface."""

from pathlib import Path
from typing import Annotated, Optional

from typer import Context, Option, Typer
from typer_config import use_config

from janus_core.calculations.phonons import Phonons
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
)
from janus_core.cli.utils import (
    carbon_summary,
    check_config,
    dict_tuples_to_lists,
    end_summary,
    parse_typer_dicts,
    save_struct_calc,
    set_read_kwargs_index,
    start_summary,
    yaml_converter_callback,
)

app = Typer()


@app.command()
@use_config(yaml_converter_callback)
def phonons(
    # numpydoc ignore=PR02
    ctx: Context,
    struct: StructPath,
    supercell: Annotated[
        tuple[int, int, int], Option(help="Supercell lattice vectors.")
    ] = (2, 2, 2),
    displacement: Annotated[
        float, Option(help="Displacement for force constants calculation, in A.")
    ] = 0.01,
    mesh: Annotated[
        tuple[int, int, int], Option(help="Mesh numbers along a, b, c axes.")
    ] = (10, 10, 10),
    bands: Annotated[
        bool,
        Option(help="Whether to compute band structure."),
    ] = False,
    dos: Annotated[bool, Option(help="Whether to calculate the DOS.")] = False,
    pdos: Annotated[bool, Option(help="Whether to calculate the PDOS.")] = False,
    thermal: Annotated[
        bool, Option(help="Whether to calculate thermal properties.")
    ] = False,
    temp_min: Annotated[
        float,
        Option(help="Start temperature for thermal properties calculations, in K."),
    ] = 0.0,
    temp_max: Annotated[
        float,
        Option(help="End temperature for thermal properties calculations, in K."),
    ] = 1000.0,
    temp_step: Annotated[
        float,
        Option(help="Temperature step for thermal properties calculations, in K."),
    ] = 50,
    symmetrize: Annotated[
        bool, Option(help="Whether to symmetrize force constants.")
    ] = False,
    minimize: Annotated[
        bool, Option(help="Whether to minimize structure before calculations.")
    ] = False,
    fmax: Annotated[
        float, Option(help="Maximum force for optimization convergence.")
    ] = 0.1,
    minimize_kwargs: MinimizeKwargs = None,
    hdf5: Annotated[
        bool, Option(help="Whether to save force constants in hdf5.")
    ] = True,
    plot_to_file: Annotated[
        bool,
        Option(help="Whether to plot band structure and/or dos/pdos when calculated."),
    ] = False,
    write_full: Annotated[
        bool,
        Option(
            help=(
                "Whether to write eigenvectors, group velocities, etc. to bands file."
            ),
        ),
    ] = True,
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
    summary: Summary = None,
):
    """
    Perform phonon calculations and write out results.

    Parameters
    ----------
    ctx : Context
        Typer (Click) Context. Automatically set.
    struct : Path
        Path of structure to simulate.
    supercell : tuple[int, int, int]
        Supercell lattice vectors. Default is (2, 2, 2).
    displacement : float
        Displacement for force constants calculation, in A. Default is 0.01.
    mesh : tuple[int, int, int]
        Mesh for sampling. Default is (10, 10, 10).
    bands : bool
        Whether to calculate and save the band structure. Default is False.
    dos : bool
        Whether to calculate and save the DOS. Default is False.
    pdos : bool
        Whether to calculate and save the PDOS. Default is False.
    thermal : bool
        Whether to calculate thermal properties. Default is False.
    temp_min : float
        Start temperature for thermal calculations, in K. Unused if `thermal` is False.
        Default is 0.0.
    temp_max : float
        End temperature for thermal calculations, in K. Unused if `thermal` is False.
        Default is 1000.0.
    temp_step : float
        Temperature step for thermal calculations, in K. Unused if `thermal` is False.
        Default is 50.0.
    symmetrize : bool
        Whether to symmetrize force constants. Default is False.
    minimize : bool
        Whether to minimize structure before calculations. Default is False.
    fmax : float
        Set force convergence criteria for optimizer in units eV/Å.
        Default is 0.1.
    minimize_kwargs : Optional[dict[str, Any]]
        Other keyword arguments to pass to geometry optimizer. Default is {}.
    hdf5 : bool
        Whether to save force constants in hdf5 format. Default is True.
    plot_to_file : bool
        Whether to plot. Default is False.
    write_full : bool
        Whether to maximize information written in various output files.
        Default is True.
    arch : Optional[str]
        MLIP architecture to use for geometry optimization.
        Default is "mace_mp".
    device : Optional[str]
        Device to run model on. Default is "cpu".
    model_path : Optional[str]
        Path to MLIP model. Default is `None`.
    read_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ase.io.read. By default,
            read_kwargs["index"] is 0.
    calc_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to the selected calculator. Default is {}.
    file_prefix : Optional[PathLike]
        Prefix for output filenames. Default is inferred from structure name, or
        chemical formula.
    log : Optional[Path]
        Path to write logs to. Default is inferred from the name of the structure file.
    summary : Optional[Path]
        Path to save summary of inputs, start/end time, and carbon emissions. Default
        is inferred from the name of the structure file.
    config : Optional[Path]
        Path to yaml configuration file to define the above options. Default is None.
    """
    # Check options from configuration file are all valid
    check_config(ctx)

    read_kwargs, calc_kwargs, minimize_kwargs = parse_typer_dicts(
        [read_kwargs, calc_kwargs, minimize_kwargs]
    )

    # Read only first structure by default and ensure only one image is read
    set_read_kwargs_index(read_kwargs)

    # Check fmax option not duplicated
    if "fmax" in minimize_kwargs:
        raise ValueError("'fmax' must be passed through the --fmax option")
    minimize_kwargs["fmax"] = fmax

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
        "struct_path": struct,
        "arch": arch,
        "device": device,
        "model_path": model_path,
        "read_kwargs": read_kwargs,
        "calc_kwargs": calc_kwargs,
        "attach_logger": True,
        "log_kwargs": log_kwargs,
        "calcs": calcs,
        "supercell": supercell,
        "displacement": displacement,
        "mesh": mesh,
        "symmetrize": symmetrize,
        "minimize": minimize,
        "minimize_kwargs": minimize_kwargs,
        "temp_min": temp_min,
        "temp_max": temp_max,
        "temp_step": temp_step,
        "force_consts_to_hdf5": hdf5,
        "plot_to_file": plot_to_file,
        "write_results": True,
        "write_full": write_full,
        "file_prefix": file_prefix,
    }

    # Initialise phonons
    phonon = Phonons(**phonons_kwargs)

    # Set summary and log files
    summary = phonon._build_filename("phonons-summary.yml", filename=summary).absolute()
    log = phonon.log_kwargs["filename"]

    # Store inputs for yaml summary
    inputs = phonons_kwargs.copy()

    # Add structure, MLIP information, and log to inputs
    save_struct_calc(
        inputs=inputs,
        struct=phonon.struct,
        struct_path=struct,
        arch=arch,
        device=device,
        model_path=model_path,
        read_kwargs=read_kwargs,
        calc_kwargs=calc_kwargs,
        log=log,
    )

    # Convert all tuples to list in inputs nested dictionary
    dict_tuples_to_lists(inputs)

    # Save summary information before calculations begin
    start_summary(command="phonons", summary=summary, inputs=inputs)

    # Run phonon calculations
    phonon.run()

    # Save carbon summary
    carbon_summary(summary=summary, log=log)

    # Time after calculations have finished
    end_summary(summary)
