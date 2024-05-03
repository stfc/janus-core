"""Set up MLIP training commandline interface."""

from pathlib import Path
from typing import Annotated

from typer import Option, Typer

from janus_core.helpers.train import train as run_train

app = Typer()


@app.command(help="Perform single point calculations and save to file.")
def train(
    mlip_config: Annotated[Path, Option(help="Configuration file to pass to MLIP CLI.")]
):
    """
    Run training for MLIP by passing a configuration file to the MLIP's CLI.

    Parameters
    ----------
    mlip_config : Path
        Configuration file to pass to MLIP CLI.
    """
    run_train(mlip_config)
