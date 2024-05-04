"""Train MLIP."""

from pathlib import Path
from typing import Optional

try:
    from mace.cli.run_train import run as run_train
except ImportError as e:
    raise NotImplementedError("Please update MACE to use this module.") from e
from mace.tools import build_default_arg_parser as mace_parser
import yaml

from janus_core.helpers.janus_types import PathLike


def check_files_exist(config: dict, req_file_keys: list[PathLike]) -> None:
    """
    Check files specified in the MLIP configuration file exist.

    Parameters
    ----------
    config : dict
        MLIP configuration file options.
    req_file_keys : list[Pathlike]
        List of files that must exist if defined in the configuration file.

    Raises
    ------
    FileNotFoundError
        If a key from `req_file_keys` is in the configuration file, but the
        file corresponding to the configuration value do not exist.
    """
    for file_key in req_file_keys:
        # Only check if file key is in the configuration file
        if file_key in config and not Path(config[file_key]).exists():
            raise FileNotFoundError(f"{config[file_key]} does not exist")


def train(
    mlip_config: PathLike, req_file_keys: Optional[list[PathLike]] = None
) -> None:
    """
    Run training for MLIP by passing a configuration file to the MLIP's CLI.

    Currently only supports MACE models, but this can be extended by replacing the
    argument parsing.

    Parameters
    ----------
    mlip_config : PathLike
        Configuration file to pass to MLIP.
    req_file_keys : Optional[list[PathLike]]
        List of files that must exist if defined in the configuration file.
        Default is ["train_file", "test_file", "valid_file", "statistics_file"].
    """
    if req_file_keys is None:
        req_file_keys = ["train_file", "test_file", "valid_file", "statistics_file"]

    # Validate inputs
    with open(mlip_config, encoding="utf8") as file:
        options = yaml.safe_load(file)
    check_files_exist(options, req_file_keys)

    if "foundation_model" in options:
        print(f"Fine tuning model: {options['foundation_model']}")

    # Path must be passed as a string
    mlip_args = mace_parser().parse_args(["--config", str(mlip_config)])
    run_train(mlip_args)
