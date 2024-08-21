"""Utility functions for CLI."""

from __future__ import annotations

from collections.abc import Sequence
import datetime
import logging
from typing import Any, TYPE_CHECKING

from typer_config import conf_callback_factory


if TYPE_CHECKING:
    from pathlib import Path
    from typer import Context

    from janus_core.cli.types import TyperDict
    from janus_core.helpers.janus_types import (
        Architectures,
        ASEReadArgs,
        Devices,
    )
    from janus_core.calculations.single_point import SinglePoint


def set_read_kwargs_index(read_kwargs: dict[str, Any]) -> None:
    """
    Set default read_kwargs["index"] and check its value is an integer.

    To ensure only a single Atoms object is read, slices such as ":" are forbidden.

    Parameters
    ----------
    read_kwargs : dict[str, Any]
        Keyword arguments to be passed to ase.io.read. If specified,
        read_kwargs["index"] must be an integer, and if not, a default value
        of 0 is set.
    """
    read_kwargs.setdefault("index", 0)
    try:
        int(read_kwargs["index"])
    except ValueError as e:
        raise ValueError("`read_kwargs['index']` must be an integer") from e


def parse_typer_dicts(typer_dicts: list["TyperDict"]) -> list[dict]:
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
    from typer_config import yaml_loader

    from janus_core.helpers.utils import dict_remove_hyphens


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
    import yaml

    save_info = {
        "command": f"janus {command}",
        "start_time": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
        "inputs": inputs,
    }
    with open(summary, "w", encoding="utf8") as outfile:
        yaml.dump(save_info, outfile, default_flow_style=False)


def carbon_summary(*, summary: Path, log: Path) -> None:
    """
    Calculate and write carbon tracking summary.

    Parameters
    ----------
    summary : Path
        Path to summary file being saved.
    log : Path
        Path to log file with carbon emissions saved.
    """
    import yaml

    with open(log, encoding="utf8") as file:
        logs = yaml.safe_load(file)

    emissions = sum(
        lg["message"]["emissions"]
        for lg in logs
        if isinstance(lg["message"], dict) and "emissions" in lg["message"]
    )

    with open(summary, "a", encoding="utf8") as outfile:
        yaml.dump({"emissions": emissions}, outfile, default_flow_style=False)


def end_summary(summary: Path) -> None:
    """
    Write final time to summary and close.

    Parameters
    ----------
    summary : Path
        Path to summary file being saved.
    """
    import yaml

    with open(summary, "a", encoding="utf8") as outfile:
        yaml.dump(
            {"end_time": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")},
            outfile,
            default_flow_style=False,
        )
    logging.shutdown()


def save_struct_calc(
    inputs: dict,
    s_point: SinglePoint,
    arch: Architectures,
    device: Devices,
    model_path: str,
    read_kwargs: ASEReadArgs,
    calc_kwargs: dict[str, Any],
) -> None:
    """
    Add structure and calculator input information to a dictionary.

    Parameters
    ----------
    inputs : dict
        Inputs dictionary to add information to.
    s_point : SinglePoint
        SinglePoint object storing structure with attached calculator.
    arch : Architectures
        MLIP architecture.
    device : Devices
        Device to run calculations on.
    model_path : str
        Path to MLIP model.
    read_kwargs : ASEReadArgs
        Keyword arguments to pass to ase.io.read.
    calc_kwargs : dict[str, Any]]
        Keyword arguments to pass to the calculator.
    """
    from ase import Atoms

    # Remove duplicate struct if already in inputs:
    inputs.pop("struct", None)

    if isinstance(s_point.struct, Atoms):
        inputs["struct"] = {
            "n_atoms": len(s_point.struct),
            "struct_path": s_point.struct_path,
            "formula": s_point.struct.get_chemical_formula(),
        }
    elif isinstance(s_point.struct, Sequence):
        inputs["traj"] = {
            "length": len(s_point.struct),
            "struct_path": s_point.struct_path,
            "struct": {
                "n_atoms": len(s_point.struct[0]),
                "formula": s_point.struct[0].get_chemical_formula(),
            },
        }

    inputs["calc"] = {
        "arch": arch,
        "device": device,
        "model_path": model_path,
        "read_kwargs": read_kwargs,
        "calc_kwargs": calc_kwargs,
    }


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
