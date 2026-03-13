"""Train MLIP."""

from __future__ import annotations

from argparse import ArgumentParser
from typing import Any

import yaml

from janus_core.helpers.janus_types import Architectures, PathLike
from janus_core.helpers.log import config_logger, config_tracker
from janus_core.helpers.utils import none_to_dict, set_log_tracker


def train(
    arch: Architectures,
    mlip_config: PathLike,
    file_prefix: PathLike,
    attach_logger: bool = False,
    log_kwargs: dict[str, Any] | None = None,
    track_carbon: bool | None = None,
    tracker_kwargs: dict[str, Any] | None = None,
) -> None:
    """
    Run training for MLIP by passing a configuration file to the MLIP's CLI.

    Currently only supports MACE models, but this can be extended by replacing the
    argument parsing.

    Parameters
    ----------
    arch
        The architecture to train.
    mlip_config
        Configuration file to pass to MLIP.
    file_prefix
        Prefix for output files, including directories.
    attach_logger
        Whether to attach a logger. Default is True if "filename" is passed in
        log_kwargs, else False.
    log_kwargs
        Keyword arguments to pass to `config_logger`. Default is {}.
    track_carbon
        Whether to track carbon emissions of calculation. Requires attach_logger.
        Default is True if attach_logger is True, else False.
    tracker_kwargs
        Keyword arguments to pass to `config_tracker`. Default is {}.
    """
    with open(mlip_config, encoding="utf8") as file:
        options = yaml.safe_load(file)

    foundation_model = None

    match arch:
        case "mace" | "mace_mp" | "mace_off" | "mace_omol":
            from mace.cli.run_train import run
            from mace.tools import build_default_arg_parser

            # Path must be passed as a string
            mlip_args = build_default_arg_parser().parse_args(
                ["--config", str(mlip_config), "--work_dir", str(file_prefix)]
            )

            foundation_model = options.get("foundation_model")

        case "nequip":
            from hydra import compose
            from hydra import initialize_config_dir as initialize
            from hydra.core.hydra_config import HydraConfig
            from nequip.scripts.train import main as run

            if mlip_config.suffix != ".yaml":
                raise ValueError(
                    "Hydra (nequip) only supports .yaml config files, "
                    f"{mlip_config} will not be found."
                )

            # Setup the HydraConfig global singleton (Compose API).
            # Paths must be strings.
            initialize(version_base=None, config_dir=str(mlip_config.parent.absolute()))
            # Obtain the HydraConfig from the path.
            mlip_args = compose(config_name=mlip_config.stem, return_hydra_config=True)
            # This is normally set when using the Hydra CLI directly. The Compose
            # API does not set it.
            mlip_args.hydra.runtime.output_dir = file_prefix
            HydraConfig().set_config(mlip_args)

            model = options["training_module"]["model"]
            foundation_model = model.get("package_path")
            if "checkpoint_path" in model:
                if foundation_model:
                    raise ValueError(
                        f"Both package_path and checkpoint_path in {mlip_config}."
                    )
                foundation_model = model["checkpoint_path"]

        case "sevennet":
            from sevenn.main.sevenn import cmd_parser_train, run

            parser = ArgumentParser()
            cmd_parser_train(parser)
            mlip_args = parser.parse_args(
                [str(mlip_config), "--working_dir", str(file_prefix), "-s"]
            )

        case _:
            raise ValueError(f"{arch} is currently unsupported in train.")

    log_kwargs, tracker_kwargs = none_to_dict(log_kwargs, tracker_kwargs)

    attach_logger, track_carbon = set_log_tracker(
        attach_logger, log_kwargs, track_carbon
    )

    # Configure logging
    if attach_logger:
        log_kwargs.setdefault("filename", "train-log.yml")
    log_kwargs.setdefault("name", __name__)
    logger = config_logger(**log_kwargs)

    if logger and foundation_model is not None:
        logger.info("Fine tuning model: %s", foundation_model)

    tracker = config_tracker(logger, track_carbon, **tracker_kwargs)

    if logger:
        logger.info("Starting training")
    if tracker:
        tracker.start_task("Training")

    run(mlip_args)

    if logger:
        logger.info("Training complete")
    if tracker:
        tracker.stop_task()
        tracker.stop()
