"""Set up phonons commandline interface."""

from typing import Annotated

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
    supercell: Annotated[
        list[int],
        Option(help="Supercell lattice vectors."),
    ],
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
    supercell : List[int]
        Supercell lattice vectors. Can be passed as a single value, or list of three.
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

    if len(supercell) == 1:
        supercell *= 3
    if len(supercell) != 3:
        raise ValueError(
            """Please pass the lattice vectors via either --supercell x --supercell y \
--supercell z, or --supercell x."""
        )

    # Dictionary of inputs for phonons
    phonons_kwargs = {
        "struct": s_point.struct,
        "supercell": supercell,
        "minimize": minimize,
        "minimize_kwargs": minimize_kwargs,
        "log_kwargs": log_kwargs,
    }

    # Store inputs for yaml summary
    inputs = phonons_kwargs.copy()

    # Store only filename as filemode is not set by user
    del inputs["log_kwargs"]
    inputs["log"] = log

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
    if dos:
        phonon.calc_dos()
    if pdos:
        phonon.calc_pdos()

    # Time after calculations have finished
    end_summary(summary)
