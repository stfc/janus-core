"""Set up phonons commandline interface."""

from pathlib import Path
from typing import Annotated, Optional

from typer import Context, Option, Typer
from typer_config import use_config

from janus_core.calculations.phonons import Phonons
from janus_core.calculations.single_point import SinglePoint
from janus_core.cli.types import (
    Architecture,
    CalcKwargs,
    Device,
    LogPath,
    MinimizeKwargs,
    ReadKwargs,
    StructPath,
    Summary,
)
from janus_core.cli.utils import (
    check_config,
    end_summary,
    parse_typer_dicts,
    start_summary,
    yaml_converter_callback,
)
from janus_core.helpers.utils import dict_paths_to_strs

app = Typer()


@app.command(help="Calculate phonons and save results.")
@use_config(yaml_converter_callback)
def phonons(
    # pylint: disable=too-many-arguments,too-many-locals
    # numpydoc ignore=PR02
    ctx: Context,
    struct: StructPath,
    struct_name: Annotated[
        Optional[str],
        Option(help="Name of structure name."),
    ] = None,
    supercell: Annotated[
        str,
        Option(help="Supercell lattice vectors in the form '1x2x3'."),
    ] = "2x2x2",
    displacement: Annotated[
        float,
        Option(help="Displacement for force constants calculation, in A."),
    ] = 0.01,
    thermal: Annotated[
        bool,
        Option(help="Whether to calculate thermal properties."),
    ] = False,
    temp_start: Annotated[
        float,
        Option(help="Start temperature for thermal properties calculations, in K."),
    ] = 0.0,
    temp_end: Annotated[
        float,
        Option(help="End temperature for thermal properties calculations, in K."),
    ] = 1000.0,
    temp_step: Annotated[
        float,
        Option(help="Temperature step for thermal properties calculations, in K."),
    ] = 50,
    dos: Annotated[
        bool,
        Option(help="Whether to calculate the DOS."),
    ] = False,
    pdos: Annotated[
        bool,
        Option(
            help="Whether to calculate the PDOS.",
        ),
    ] = False,
    minimize: Annotated[
        bool, Option(help="Whether to minimize structure before calculations.")
    ] = False,
    fmax: Annotated[
        float, Option(help="Maximum force for optimization convergence.")
    ] = 0.1,
    minimize_kwargs: MinimizeKwargs = None,
    arch: Architecture = "mace_mp",
    device: Device = "cpu",
    read_kwargs: ReadKwargs = None,
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
    log: LogPath = "phonons.log",
    summary: Summary = "phonons_summary.yml",
):
    """
    Perform phonon calculations and write out results.

    Parameters
    ----------
    ctx : Context
        Typer (Click) Context. Automatically set.
    struct : Path
        Path of structure to simulate.
    struct_name : Optional[PathLike]
        Name of structure to simulate. Default is inferred from filepath or chemical
        formula.
    supercell : str
        Supercell lattice vectors. Must be passed in the form '1x2x3'. Default is
        2x2x2.
    displacement : float
        Displacement for force constants calculation, in A. Default is 0.01.
    thermal : bool
        Whether to calculate thermal properties. Default is False.
    temp_start : float
        Start temperature for CV calculations, in K. Unused if `thermal` is False.
        Default is 0.0.
    temp_end : float
        End temperature for CV calculations, in K. Unused if `thermal` is False.
        Default is 1000.0.
    temp_step : float
        Temperature step for CV calculations, in K. Unused if `thermal` is False.
        Default is 50.0.
    dos : bool
        Whether to calculate and save the DOS. Default is False.
    pdos : bool
        Whether to calculate and save the PDOS. Default is False.
    minimize : bool
        Whether to minimize structure before calculations. Default is False.
    fmax : float
        Set force convergence criteria for optimizer in units eV/Å.
        Default is 0.1.
    minimize_kwargs : Optional[dict[str, Any]]
        Other keyword arguments to pass to geometry optimizer. Default is {}.
    arch : Optional[str]
        MLIP architecture to use for geometry optimization.
        Default is "mace_mp".
    device : Optional[str]
        Device to run model on. Default is "cpu".
    read_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ase.io.read. Default is {}.
    calc_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to the selected calculator. Default is {}.
    file_prefix : Optional[PathLike]
        Prefix for output filenames. Default is inferred from structure name, or
        chemical formula.
    log : Optional[Path]
        Path to write logs to. Default is "phonons.log".
    summary : Path
        Path to save summary of inputs and start/end time. Default is
        phonons.yml.
    config : Path
        Path to yaml configuration file to define the above options. Default is None.
    """
    # Check options from configuration file are all valid
    check_config(ctx)

    [read_kwargs, calc_kwargs, minimize_kwargs] = parse_typer_dicts(
        [read_kwargs, calc_kwargs, minimize_kwargs]
    )

    # Set up single point calculator
    s_point = SinglePoint(
        struct_path=struct,
        struct_name=struct_name,
        architecture=arch,
        device=device,
        read_kwargs=read_kwargs,
        calc_kwargs=calc_kwargs,
        log_kwargs={"filename": log, "filemode": "w"},
    )

    log_kwargs = {"filename": log, "filemode": "a"}

    # Check fmax option not duplicated
    if "fmax" in minimize_kwargs:
        raise ValueError("'fmax' must be passed through the --fmax option")
    minimize_kwargs["fmax"] = fmax

    try:
        supercell = [int(x) for x in supercell.split("x")]
    except ValueError as exc:
        raise ValueError(
            "Please pass lattice vectors as integers in the form 1x2x3"
        ) from exc

    # Validate supercell list
    if len(supercell) != 3:
        raise ValueError("Please pass three lattice vectors in the form 1x2x3")

    # Dictionary of inputs for phonons
    phonons_kwargs = {
        "struct": s_point.struct,
        "struct_name": s_point.struct_name,
        "supercell": supercell,
        "displacement": displacement,
        "t_min": temp_start,
        "t_max": temp_end,
        "t_step": temp_step,
        "minimize": minimize,
        "minimize_kwargs": minimize_kwargs,
        "file_prefix": file_prefix,
        "log_kwargs": log_kwargs,
    }

    # Store inputs for yaml summary
    inputs = phonons_kwargs.copy()

    # Store only filename as filemode is not set by user
    del inputs["log_kwargs"]
    inputs["log"] = log

    inputs["dos"] = dos
    inputs["pdos"] = pdos
    inputs["thermal"] = thermal

    inputs["struct"] = {
        "n_atoms": len(s_point.struct),
        "struct_path": struct,
        "struct_name": s_point.struct_name,
        "formula": s_point.struct.get_chemical_formula(),
    }

    inputs["calc"] = {
        "arch": arch,
        "device": device,
        "read_kwargs": read_kwargs,
        "calc_kwargs": calc_kwargs,
    }

    # Convert all paths to strings in inputs nested dictionary
    dict_paths_to_strs(inputs)

    # Save summary information before calculations begin
    start_summary(command="phonons", summary=summary, inputs=inputs)

    # Initialise phonons class
    phonon = Phonons(**phonons_kwargs)

    # Calculate phonons
    phonon.calc_phonons()

    # Calculate DOS and PDOS is specified
    if thermal:
        phonon.calc_thermal_props()

    # Calculate DOS and PDOS is specified
    if dos:
        phonon.calc_dos()
    if pdos:
        phonon.calc_pdos()

    # Time after calculations have finished
    end_summary(summary)