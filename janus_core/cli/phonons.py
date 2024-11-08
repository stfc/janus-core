# ruff: noqa: I002, FA100
"""Set up phonons commandline interface."""

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
    DisplacementKwargs,
    DoSKwargs,
    LogPath,
    MinimizeKwargs,
    ModelPath,
    PDoSKwargs,
    ReadKwargsLast,
    StructPath,
    Summary,
)
from janus_core.cli.utils import yaml_converter_callback

app = Typer()


@app.command()
@use_config(yaml_converter_callback)
def phonons(
    # numpydoc ignore=PR02
    ctx: Context,
    struct: StructPath,
    supercell: Annotated[
        str,
        Option(
            help="Supercell matrix, in the Phonopy style. Must be passed as a string "
            "in one of three forms: single integer ('2'), which specifies all "
            "diagonal elements; three integers ('1 2 3'), which specifies each "
            "individual diagonal element; or nine values ('1 2 3 4 5 6 7 8 9'), "
            "which specifies all elements, filling the matrix row-wise."
        ),
    ] = "2 2 2",
    displacement: Annotated[
        float, Option(help="Displacement for force constants calculation, in A.")
    ] = 0.01,
    displacement_kwargs: DisplacementKwargs = None,
    mesh: Annotated[
        tuple[int, int, int], Option(help="Mesh numbers along a, b, c axes.")
    ] = (10, 10, 10),
    bands: Annotated[
        bool,
        Option(help="Whether to compute band structure."),
    ] = False,
    n_qpoints: Annotated[
        int,
        Option(
            help=(
                "Number of q-points to sample along generated path, including end "
                "points. Unused if `qpoint_file` is specified"
            )
        ),
    ] = 51,
    qpoint_file: Annotated[
        Optional[Path],
        Option(
            help=(
                "Path to yaml file with info to generate a path of q-points for band "
                "structure."
            )
        ),
    ] = None,
    dos: Annotated[bool, Option(help="Whether to calculate the DOS.")] = False,
    dos_kwargs: DoSKwargs = None,
    pdos: Annotated[bool, Option(help="Whether to calculate the PDOS.")] = False,
    pdos_kwargs: PDoSKwargs = None,
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
    tracker: Annotated[
        bool, Option(help="Whether to save carbon emissions of calculation")
    ] = True,
    summary: Summary = None,
) -> None:
    """
    Perform phonon calculations and write out results.

    Parameters
    ----------
    ctx : Context
        Typer (Click) Context. Automatically set.
    struct : Path
        Path of structure to simulate.
    supercell : str
        Supercell matrix, in the Phonopy style. Must be passed as a string in one of
        three forms: single integer ('2'), which specifies all diagonal elements;
        three integers ('1 2 3'), which specifies each individual diagonal element;
        or nine values ('1 2 3 4 5 6 7 8 9'), which specifies all elements, filling the
        matrix row-wise.
    displacement : float
        Displacement for force constants calculation, in A. Default is 0.01.
    displacement_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to generate_displacements. Default is {}.
    mesh : tuple[int, int, int]
        Mesh for sampling. Default is (10, 10, 10).
    bands : bool
        Whether to calculate and save the band structure. Default is False.
    n_qpoints : int
        Number of q-points to sample along generated path, including end points.
        Unused if `qpoint_file` is specified. Default is 51.
    qpoint_file : Optional[PathLike]
        Path to yaml file with info to generate a path of q-points for band structure.
        Default is None.
    dos : bool
        Whether to calculate and save the DOS. Default is False.
    dos_kwargs : Optional[dict[str, Any]]
        Other keyword arguments to pass to run_total_dos. Default is {}.
    pdos : bool
        Whether to calculate and save the PDOS. Default is False.
    pdos_kwargs : Optional[dict[str, Any]]
        Other keyword arguments to pass to run_projected_dos. Default is {}.
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
    tracker : bool
        Whether to save carbon emissions of calculation in log file and summary.
        Default is True.
    summary : Optional[Path]
        Path to save summary of inputs, start/end time, and carbon emissions. Default
        is inferred from the name of the structure file.
    config : Optional[Path]
        Path to yaml configuration file to define the above options. Default is None.
    """
    from janus_core.calculations.phonons import Phonons
    from janus_core.cli.utils import (
        carbon_summary,
        check_config,
        dict_tuples_to_lists,
        end_summary,
        parse_typer_dicts,
        save_struct_calc,
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
        "struct_path": struct,
        "arch": arch,
        "device": device,
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
        "force_consts_to_hdf5": hdf5,
        "plot_to_file": plot_to_file,
        "write_results": True,
        "write_full": write_full,
        "file_prefix": file_prefix,
        "enable_progress_bar": True,
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
    if tracker:
        carbon_summary(summary=summary, log=log)

    # Time after calculations have finished
    end_summary(summary)
