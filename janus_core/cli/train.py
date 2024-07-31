"""Set up MLIP training commandline interface."""

from pathlib import Path
from typing import Annotated

from typer import Option, Typer
import yaml

from janus_core.helpers.train import train as run_train

app = Typer()


@app.command()
def train(
    mlip_config: Annotated[
        Path, Option(help="Configuration file to pass to MLIP CLI.")
    ],
    fine_tune: Annotated[
        bool, Option(help="Whether to fine-tune a foundational model.")
    ] = False,
):
    """
    Run training for MLIP by passing a configuration file to the MLIP's CLI.

    Parameters
    ----------
    mlip_config : Path
        Configuration file to pass to MLIP CLI.
    fine_tune : bool
        Whether to fine-tune a foundational model. Default is False.
    """
    with open(mlip_config, encoding="utf8") as config_file:
        config = yaml.safe_load(config_file)

    if fine_tune:
        if "foundation_model" not in config:
            raise ValueError(
                "Please include `foundation_model` in your configuration file"
            )
        if (
            config["foundation_model"]
            not in ("small", "medium", "large", "small_off", "medium_off", "large_off")
            and not Path(config["foundation_model"]).exists()
        ):
            raise ValueError(
                """
                Invalid foundational model. Valid options are: 'small', 'medium',
                'large', 'small_off', 'medium_off', 'large_off', or a path to the model
                """
            )
    elif "foundation_model" in config:
        raise ValueError("Please include the `--fine-tune` option for fine-tuning")

    run_train(mlip_config)
