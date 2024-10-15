"""Utility functions for CLI."""

from __future__ import annotations

from collections.abc import Sequence
import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from typer_config import conf_callback_factory, yaml_loader
import yaml

if TYPE_CHECKING:
    from ase import Atoms
    from typer import Context

    from janus_core.cli.types import TyperDict
    from janus_core.helpers.janus_types import (
        Architectures,
        ASEReadArgs,
        Devices,
        MaybeSequence,
    )


def dict_paths_to_strs(dictionary: dict) -> None:
    """
    Recursively iterate over dictionary, converting Path values to strings.

    Parameters
    ----------
    dictionary : dict
        Dictionary to be converted.
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dict_paths_to_strs(value)
        elif isinstance(value, Path):
            dictionary[key] = str(value)


def dict_tuples_to_lists(dictionary: dict) -> None:
    """
    Recursively iterate over dictionary, converting tuple values to lists.

    Parameters
    ----------
    dictionary : dict
        Dictionary to be converted.
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dict_paths_to_strs(value)
        elif isinstance(value, tuple):
            dictionary[key] = list(value)


def dict_remove_hyphens(dictionary: dict) -> dict:
    """
    Recursively iterate over dictionary, replacing hyphens with underscores in keys.

    Parameters
    ----------
    dictionary : dict
        Dictionary to be converted.

    Returns
    -------
    dict
        Dictionary with hyphens in keys replaced with underscores.
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = dict_remove_hyphens(value)
    return {k.replace("-", "_"): v for k, v in dictionary.items()}


def set_read_kwargs_index(read_kwargs: dict[str, Any]) -> None:
    """
    Set default read_kwargs["index"] to final image and check its value is an integer.

    To ensure only a single Atoms object is read, slices such as ":" are forbidden.

    Parameters
    ----------
    read_kwargs : dict[str, Any]
        Keyword arguments to be passed to ase.io.read. If specified,
        read_kwargs["index"] must be an integer, and if not, a default value
        of -1 is set.
    """
    read_kwargs.setdefault("index", -1)
    try:
        int(read_kwargs["index"])
    except ValueError as e:
        raise ValueError("`read_kwargs['index']` must be an integer") from e


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
    with open(summary, "a", encoding="utf8") as outfile:
        yaml.dump(
            {"end_time": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")},
            outfile,
            default_flow_style=False,
        )
    logging.shutdown()


def save_struct_calc(
    *,
    inputs: dict,
    struct: MaybeSequence[Atoms],
    struct_path: Path,
    arch: Architectures,
    device: Devices,
    model_path: str,
    read_kwargs: ASEReadArgs,
    calc_kwargs: dict[str, Any],
    log: Path,
) -> None:
    """
    Add structure and calculator input information to a dictionary.

    Parameters
    ----------
    inputs : dict
        Inputs dictionary to add information to.
    struct : MaybeSequence[Atoms]
        Structure to be simulated.
    struct_path : Path
        Path of structure file.
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
    log : Path
        Path to log file.
    """
    from ase import Atoms

    # Clean up duplicate parameters
    for key in (
        "struct",
        "struct_path",
        "arch",
        "device",
        "model_path",
        "read_kwargs",
        "calc_kwargs",
        "log_kwargs",
    ):
        inputs.pop(key, None)

    if isinstance(struct, Atoms):
        inputs["struct"] = {
            "n_atoms": len(struct),
            "struct_path": struct_path,
            "formula": struct.get_chemical_formula(),
        }
    elif isinstance(struct, Sequence):
        inputs["traj"] = {
            "length": len(struct),
            "struct_path": struct_path,
            "struct": {
                "n_atoms": len(struct[0]),
                "formula": struct[0].get_chemical_formula(),
            },
        }

    inputs["calc"] = {
        "arch": arch,
        "device": device,
        "model_path": model_path,
        "read_kwargs": read_kwargs,
        "calc_kwargs": calc_kwargs,
    }

    inputs["log"] = log

    # Convert all paths to strings in inputs nested dictionary
    dict_paths_to_strs(inputs)


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
