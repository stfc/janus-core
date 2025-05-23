"""Set up MLIP training commandline interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from typer import Option, Typer
import yaml

from janus_core.cli.types import LogPath, Summary, Tracker

app = Typer()


@app.command()
def train(
    mlip_config: Annotated[
        Path, Option(help="Configuration file to pass to MLIP CLI.", show_default=False)
    ],
    fine_tune: Annotated[
        bool, Option(help="Whether to fine-tune a foundational model.")
    ] = False,
    log: LogPath = Path("./janus_results/train-log.yml"),
    tracker: Tracker = True,
    summary: Summary = Path("./janus_results/train-summary.yml"),
) -> None:
    """
    Run training for MLIP by passing a configuration file to the MLIP's CLI.

    Parameters
    ----------
    mlip_config
        Configuration file to pass to MLIP CLI.
    fine_tune
        Whether to fine-tune a foundational model. Default is False.
    log
        Path to write logs to. Default is Path("train-log.yml").
    tracker
        Whether to save carbon emissions of calculation in log file and summary.
        Default is True.
    summary
        Path to save summary of inputs, start/end time, and carbon emissions. Default
        is Path("train-summary.yml").
    """
    from janus_core.cli.utils import carbon_summary, end_summary, start_summary
    from janus_core.training.train import train as run_train

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

    config = {
        "mlip_config": mlip_config,
        "fine_tune": fine_tune,
        "log": log,
        "tracker": tracker,
        "summary": summary,
    }

    log_kwargs = {"filemode": "w"}
    if log:
        log_kwargs["filename"] = log

    output_files = {"log": log.absolute()}

    # Save summary information before training begins
    start_summary(
        command="train",
        summary=summary,
        info={},
        config=config,
        output_files=output_files,
    )

    # Run training
    run_train(
        mlip_config, attach_logger=True, log_kwargs=log_kwargs, track_carbon=tracker
    )

    # Save carbon summary
    if tracker:
        carbon_summary(summary=summary, log=log)

    # Save time after training has finished
    end_summary(summary)
