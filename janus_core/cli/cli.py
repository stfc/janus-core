"""Set up commandline interface."""

from typing import Annotated

from typer import Exit, Option, Typer

from janus_core import __version__
from janus_core.cli.geomopt import geomopt
from janus_core.cli.md import md
from janus_core.cli.singlepoint import singlepoint

app = Typer(name="janus", no_args_is_help=True)
app.command()(singlepoint)
app.command()(geomopt)
app.command()(md)


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
