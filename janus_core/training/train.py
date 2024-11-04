"""Train MLIP."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mace.cli.run_train import run
from mace.tools import build_default_arg_parser as mace_parser
import yaml

from janus_core.helpers.janus_types import PathLike
from janus_core.helpers.log import config_logger, config_tracker
from janus_core.helpers.utils import check_files_exist, none_to_dict


def train(
    mlip_config: PathLike,
    req_file_keys: Sequence[PathLike] = (
        "train_file",
        "test_file",
        "valid_file",
        "statistics_file",
    ),
    attach_logger: bool = False,
    log_kwargs: dict[str, Any] | None = None,
    track_carbon: bool = True,
    tracker_kwargs: dict[str, Any] | None = None,
) -> None:
    """
    Run training for MLIP by passing a configuration file to the MLIP's CLI.

    Currently only supports MACE models, but this can be extended by replacing the
    argument parsing.

    Parameters
    ----------
    mlip_config : PathLike
        Configuration file to pass to MLIP.
    req_file_keys : Sequence[PathLike]
        List of files that must exist if defined in the configuration file.
        Default is ("train_file", "test_file", "valid_file", "statistics_file").
    attach_logger : bool
        Whether to attach a logger. Default is False.
    log_kwargs : dict[str, Any] | None
        Keyword arguments to pass to `config_logger`. Default is {}.
    track_carbon : bool
        Whether to track carbon emissions of calculation. Default is True.
    tracker_kwargs : dict[str, Any] | None
        Keyword arguments to pass to `config_tracker`. Default is {}.
    """
    log_kwargs, tracker_kwargs = none_to_dict(log_kwargs, tracker_kwargs)

    # Validate inputs
    with open(mlip_config, encoding="utf8") as file:
        options = yaml.safe_load(file)
    check_files_exist(options, req_file_keys)

    # Configure logging
    if attach_logger:
        log_kwargs.setdefault("filename", "train-log.yml")
    log_kwargs.setdefault("name", __name__)
    logger = config_logger(**log_kwargs)
    tracker = config_tracker(logger, track_carbon, **tracker_kwargs)

    if logger and "foundation_model" in options:
        logger.info("Fine tuning model: %s", options["foundation_model"])

    # Path must be passed as a string
    mlip_args = mace_parser().parse_args(["--config", str(mlip_config)])

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
