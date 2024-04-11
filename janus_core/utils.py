"""Utility functions for janus_core."""

from pathlib import Path
from typing import Optional


def none_to_dict(dictionaries: list[Optional[dict]]) -> list[dict]:
    """
    Ensure dictionaries that may be None are dictionaires.

    Parameters
    ----------
    dictionaries : list[dict]
        List of dictionaries that be be None.

    Returns
    -------
    list[dict]
        Dictionaries set to {} if previously None.
    """
    for i, dictionary in enumerate(dictionaries):
        dictionaries[i] = dictionary if dictionary else {}
    return dictionaries


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
