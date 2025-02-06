"""Preprocess MLIP training data."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mace.cli.preprocess_data import run
from mace.tools import build_preprocess_arg_parser as mace_parser
import yaml

from janus_core.helpers.janus_types import PathLike
from janus_core.helpers.log import config_logger, config_tracker
from janus_core.helpers.utils import check_files_exist, none_to_dict, set_log_tracker


def preprocess(
    mlip_config: PathLike,
    req_file_keys: Sequence[PathLike] = ("train_file", "test_file", "valid_file"),
    attach_logger: bool = False,
    log_kwargs: dict[str, Any] | None = None,
    track_carbon: bool | None = None,
    tracker_kwargs: dict[str, Any] | None = None,
) -> None:
    """
    Convert training data to hdf5 by passing a configuration file to the MLIP's CLI.

    Currently only supports MACE models, but this can be extended by replacing the
    argument parsing.

    Parameters
    ----------
    mlip_config
        Configuration file to pass to MLIP.
    req_file_keys
        List of files that must exist if defined in the configuration file.
        Default is ("train_file", "test_file", "valid_file").
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
    log_kwargs, tracker_kwargs = none_to_dict(log_kwargs, tracker_kwargs)

    # Validate inputs
    with open(mlip_config, encoding="utf8") as file:
        options = yaml.safe_load(file)
    check_files_exist(options, req_file_keys)

    attach_logger, track_carbon = set_log_tracker(
        attach_logger, log_kwargs, track_carbon
    )

    # Configure logging
    if attach_logger:
        log_kwargs.setdefault("filename", "preprocess-log.yml")
    log_kwargs.setdefault("name", __name__)
    logger = config_logger(**log_kwargs)
    tracker = config_tracker(logger, track_carbon, **tracker_kwargs)

    if logger and "foundation_model" in options:
        logger.info("Fine tuning model: %s", options["foundation_model"])

    mlip_args = mace_parser().parse_args(["--config", str(mlip_config)])

    if logger:
        logger.info("Starting preprocessing")
    if tracker:
        tracker.start_task("Preprocessing")

    run(mlip_args)

    if logger:
        logger.info("Preprocessing complete")
    if tracker:
        tracker.stop_task()
        tracker.stop()
