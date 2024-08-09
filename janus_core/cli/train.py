"""Set up MLIP training commandline interface."""

from pathlib import Path
from typing import Annotated

from typer import Option, Typer
import yaml

from janus_core.cli.types import LogPath, Summary
from janus_core.cli.utils import carbon_summary, end_summary, start_summary
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
    log: LogPath = "train.log",
    summary: Summary = "train_summary.yml",
):
    """
    Run training for MLIP by passing a configuration file to the MLIP's CLI.

    Parameters
    ----------
    mlip_config : Path
        Configuration file to pass to MLIP CLI.
    fine_tune : bool
        Whether to fine-tune a foundational model. Default is False.
    log : Optional[Path]
        Path to write logs to. Default is "train.log".
    summary : Path
        Path to save summary of inputs and start/end time. Default is
        train_summary.yml.
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

    inputs = {"mlip_config": str(mlip_config), "fine_tune": fine_tune}

    # Save summary information before training begins
    start_summary(command="train", summary=summary, inputs=inputs)

    # Run training
    run_train(mlip_config, log_kwargs={"filename": log, "filemode": "w"})

    carbon_summary(summary=summary, log=log)

    # Save time after training has finished
    end_summary(summary)
