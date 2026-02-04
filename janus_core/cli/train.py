"""Set up MLIP training commandline interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from typer import Option, Typer
import yaml

from janus_core.cli.types import Architectures, LogPath, Summary

app = Typer()


@app.command()
def train(
    arch: Architectures,
    mlip_config: Annotated[
        Path, Option(help="Configuration file to pass to MLIP CLI.", show_default=False)
    ],
    fine_tune: Annotated[
        bool, Option(help="Whether to fine-tune a foundational model.")
    ] = False,
    file_prefix: Annotated[
        Path | None,
        Option(
            help=(
                "Prefix for output files, including directories."
                "Default directory is ./janus_results."
            )
        ),
    ] = None,
    log: LogPath | None = None,
    tracker: bool = True,
    summary: Summary | None = None,
) -> None:
    """
    Run training for MLIP by passing a configuration file to the MLIP's CLI.

    Parameters
    ----------
    arch
        The achitecture to train with.
    mlip_config
        Configuration file to pass to MLIP CLI.
    fine_tune
        Whether to fine-tune a foundational model. Default is False.
    file_prefix
        Prefix for output files, including directories.
        Default directory is ./janus_results.
    log
        Path to write logs to. Default is inferred from `file_prefix`.
    tracker
        Whether to save carbon emissions of calculation in log file and summary.
        Default is True.
    summary
        Path to save summary of inputs, start/end time, and carbon emissions.
        Default is inferred from `file_prefix`.
    """
    from janus_core.cli.utils import carbon_summary, end_summary, start_summary
    from janus_core.training.train import train as run_train

    with open(mlip_config, encoding="utf8") as config_file:
        config = yaml.safe_load(config_file)

    match arch:
        case "mace" | "mace_mp" | "mace_off" | "mace_omol":
            if fine_tune:
                if "foundation_model" not in config:
                    raise ValueError(
                        "Please include `foundation_model` in your configuration file"
                    )
                if (
                    config["foundation_model"]
                    not in (
                        "small",
                        "medium",
                        "large",
                        "small_off",
                        "medium_off",
                        "large_off",
                    )
                    and not Path(config["foundation_model"]).exists()
                ):
                    raise ValueError(
                        """
                        Invalid foundational model. Valid options are: 'small',
                        'medium','large', 'small_off', 'medium_off', 'large_off',
                        or a path to the model
                        """
                    )
            elif "foundation_model" in config:
                raise ValueError(
                    "Please include the `--fine-tune` option for fine-tuning"
                )

        case "nequip":
            if "training_module" not in config:
                raise ValueError(
                    """There is no top-level training_module section in your
                    configuration file."""
                )
            if "model" not in config["training_module"]:
                raise ValueError(
                    """There is no model section in the training_module section
                    of your configuration file."""
                )
            if "_target_" not in config["training_module"]["model"]:
                raise ValueError(
                    """There is _target_ section in the model section
                    of your configuration file."""
                )
            model = config["training_module"]["model"]["_target_"]

            # See nequip.model.__all__
            if fine_tune and model not in (
                "nequip.model.ModelFromCheckpoint",
                "nequip.model.ModelFromPackage",
            ):
                raise ValueError(
                    """Fine-tuning requested but there is no checkpoint or
                    package specified in your config."""
                )
        case "sevennet":
            continue_section = config["train"].get("continue")
            if continue_section is None and fine_tune:
                raise ValueError(
                    """Fine-tuning requested but there is no continue
                    section in yor config."""
                )
                model = continue_section.get("checkpoint")
                if model is None:
                    raise ValueError(
                        """No model specified as a checkpoint for
                        fine-tuning.
                        """
                    )
            if not fine_tune and continue_section is not None:
                raise ValueError(
                    """Fine-tuning not requested but a continue
                    section is in your config. Please use
                    --fine-tune"""
                )

        case _:
            raise ValueError(f"Unsupported Architecture ({arch})")

    if file_prefix is None:
        file_prefix = Path.cwd() / "janus_results"

    if log is None:
        log = file_prefix / "train-log.yml"

    if summary is None:
        summary = file_prefix / "train-summary.yml"

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
        arch,
        mlip_config,
        file_prefix,
        attach_logger=True,
        log_kwargs=log_kwargs,
        track_carbon=tracker,
    )

    # Save carbon summary
    if tracker:
        carbon_summary(summary=summary, log=log)

    # Save time after training has finished
    end_summary(summary)
