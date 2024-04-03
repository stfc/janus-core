"""
Module that reads the md stats output timeseries.
"""

import re

from numpy import genfromtxt

from janus_core.janus_types import PathLike


class Stats:
    """
    Configure shared molecular dynamics simulation options.

    Parameters
    ----------
    source : PathLike
        File that contains the stats of a molecular dynamics simulation.
    """

    def __init__(self, source: PathLike = None) -> None:
        """
        Initialise MD stats reader.

        Parameters
        ----------
        source : PathLike
            File that contains the stats of a molecular dynamics simulation.
        """

        self.rows = 0
        self.columns = 0
        self.data = None
        self.labels = None
        self.units = None
        if source is not None:
            self.source = source
            self.read(source)

    def read(self, filename: PathLike = None) -> None:
        """
        Read MD stats and store them.

        Parameters
        ----------
        filename : PathLike
            File that contains the stats of a molecular dynamics simulation.
        """
        self.data = genfromtxt(filename, skip_header=1)
        with open(filename, "r+", encoding="utf-8") as file:
            head = file.readline().split("|")
            self.units = [
                re.search(r"\[.+?\]", x).group(0) if re.search(r"\[.+?\]", x) else ""
                for x in head
            ]
            self.labels = [
                re.sub(r"[\[].*?[\]]", "", x).rstrip().lstrip() for x in head
            ]
        self.rows, self.columns = self.data.shape
