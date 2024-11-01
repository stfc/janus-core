# ruff: noqa: I002, FA100
"""Set up commandline interface."""

# Issues with future annotations and typer
# c.f. https://github.com/maxb2/typer-config/issues/295
# from __future__ import annotations

from typing import Annotated

from typer import Exit, Option, Typer

from janus_core import __version__
from janus_core.cli.descriptors import descriptors
from janus_core.cli.eos import eos
from janus_core.cli.geomopt import geomopt
from janus_core.cli.md import md
from janus_core.cli.phonons import phonons
from janus_core.cli.preprocess import preprocess
from janus_core.cli.singlepoint import singlepoint
from janus_core.cli.train import train

app = Typer(name="janus", no_args_is_help=True)
app.command(help="Perform single point calculations and save to file.")(singlepoint)
app.command(help="Perform geometry optimization and save optimized structure to file.")(
    geomopt
)
app.command(
    help="Run molecular dynamics simulation, and save trajectory and statistics."
)(md)
app.command(help="Calculate phonons and save results.")(phonons)
app.command(help="Calculate equation of state.")(eos)
app.command(help="Calculate MLIP descriptors.")(descriptors)
app.command(help="Running training for an MLIP.")(train)
app.command(help="Running preprocessing for an MLIP.")(preprocess)


@app.callback(invoke_without_command=True, help="")
def print_version(
    version: Annotated[
        bool, Option("--version", help="Print janus version and exit.")
    ] = None,
) -> False:
    """
    Print current janus-core version and exit.

    Parameters
    ----------
    version : bool
        Whether to print the current janus-core version.
    """
    if version:
        print(f"janus-core version: {__version__}")
        raise Exit()
