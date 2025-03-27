"""Set up commandline interface."""

from __future__ import annotations

from typing import Annotated

from typer import Exit, Option, Typer

from janus_core import __version__
from janus_core.cli.descriptors import descriptors
from janus_core.cli.eos import eos
from janus_core.cli.geomopt import geomopt
from janus_core.cli.md import md
from janus_core.cli.neb import neb
from janus_core.cli.phonons import phonons
from janus_core.cli.preprocess import preprocess
from janus_core.cli.singlepoint import singlepoint
from janus_core.cli.train import train

app = Typer(
    name="janus",
    no_args_is_help=True,
    epilog="Try 'janus COMMAND --help' for subcommand options",
)
app.command(
    help="Perform single point calculations and save to file.",
    rich_help_panel="Calculations",
)(singlepoint)
app.command(
    help="Perform geometry optimization and save optimized structure to file.",
    rich_help_panel="Calculations",
)(geomopt)
app.command(
    help="Run molecular dynamics simulation, and save trajectory and statistics.",
    rich_help_panel="Calculations",
)(md)
app.command(
    help="Calculate phonons and save results.",
    rich_help_panel="Calculations",
)(phonons)
app.command(
    help="Calculate equation of state.",
    rich_help_panel="Calculations",
)(eos)
app.command(
    help="Run Nudged Elastic Band method.",
    rich_help_panel="Calculations",
)(neb)
app.command(
    help="Calculate MLIP descriptors.",
    rich_help_panel="Calculations",
)(descriptors)
app.command(
    help="Train or fine-tune an MLIP.",
    rich_help_panel="Training",
)(train)
app.command(
    help="Preprocess data before training.",
    rich_help_panel="Training",
)(preprocess)


@app.callback(invoke_without_command=True, help="")
def print_version(
    version: Annotated[
        bool, Option("--version", help="Print janus version and exit.")
    ] = None,
) -> None:
    """
    Print current janus-core version and exit.

    Parameters
    ----------
    version
        Whether to print the current janus-core version.
    """
    if version:
        print(f"janus-core version: {__version__}")
        raise Exit()
