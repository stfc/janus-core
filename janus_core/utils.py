"""Utility functions for janus_core."""

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
