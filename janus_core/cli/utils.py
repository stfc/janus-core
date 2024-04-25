"""Utility functions for CLI."""

import datetime
import logging
from pathlib import Path
from typing import Any

from typer import Context
from typer_config import conf_callback_factory, yaml_loader
import yaml

from janus_core.cli.types import TyperDict
from janus_core.helpers.utils import dict_remove_hyphens


def parse_typer_dicts(typer_dicts: list[TyperDict]) -> list[dict]:
    """
    Convert list of TyperDict objects to list of dictionaries.

    Parameters
    ----------
    typer_dicts : list[TyperDict]
        List of TyperDict objects to convert.

    Returns
    -------
    list[dict]
        List of converted dictionaries.

    Raises
    ------
    ValueError
        If items in list are not converted to dicts.
    """
    for i, typer_dict in enumerate(typer_dicts):
        typer_dicts[i] = typer_dict.value if typer_dict else {}
        if not isinstance(typer_dicts[i], dict):
            raise ValueError(
                f"""{typer_dicts[i]} must be passed as a dictionary wrapped in quotes.\
 For example, "{{'key' : value}}" """
            )
    return typer_dicts


def yaml_converter_loader(config_file: str) -> dict[str, Any]:
    """
    Load yaml configuration and replace hyphens with underscores.

    Parameters
    ----------
    config_file : str
        Yaml configuration file to read.

    Returns
    -------
    dict[str, Any]
        Dictionary with loaded configuration.
    """
    if not config_file:
        return {}

    config = yaml_loader(config_file)
    # Replace all "-"" with "_" in conf
    return dict_remove_hyphens(config)


yaml_converter_callback = conf_callback_factory(yaml_converter_loader)


def start_summary(*, command: str, summary: Path, inputs: dict) -> None:
    """
    Write initial summary contents.

    Parameters
    ----------
    command : str
        Name of CLI command being used.
    summary : Path
        Path to summary file being saved.
    inputs : dict
        Inputs to CLI command to save.
    """
    save_info = {
        "command": f"janus {command}",
        "start_time": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
        "inputs": inputs,
    }
    with open(summary, "w", encoding="utf8") as outfile:
        yaml.dump(save_info, outfile, default_flow_style=False)


def end_summary(summary: Path) -> None:
    """
    Write final time to summary and close.

    Parameters
    ----------
    summary : Path
        Path to summary file being saved.
    """
    with open(summary, "a", encoding="utf8") as outfile:
        yaml.dump(
            {"end_time": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")},
            outfile,
            default_flow_style=False,
        )
    logging.shutdown()


def check_config(ctx: Context) -> None:
    """
    Check options in configuration file are valid options for CLI command.

    Parameters
    ----------
    ctx : Context
        Typer (Click) Context within command.
    """
    # Compare options from config file (default_map) to function definition (params)
    for option in ctx.default_map:
        # Check options individually so can inform user of specific issue
        if option not in ctx.params:
            raise ValueError(f"'{option}' in configuration file is not a valid option")
