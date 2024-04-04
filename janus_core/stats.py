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

    # pylint: disable=too-many-instance-attributes

    def __init__(self, source: PathLike) -> None:
        """
        Initialise MD stats reader.

        Parameters
        ----------
        source : PathLike
            File that contains the stats of a molecular dynamics simulation.
        """

        self._rows = 0
        self._columns = 0
        self.data = None
        self.labels = None
        self.units = None
        self.source = source
        self.read(source)

    @property
    def rows(self) -> int:
        """
        Return number of rows.

        Returns
        -------
        int
            Number of rows in `data`.
        """
        return self._rows

    @rows.setter
    def rows(self, val_rows: int) -> None:
        """
        Set number of rows.

        Parameters
        ----------
        val_rows : int
            Number of rows in `data`.
        """
        self._rows = val_rows

    @property
    def columns(self) -> int:
        """
        Return number of columns.

        Returns
        -------
        int
            Number of columns in `data`.
        """
        return self._columns

    @columns.setter
    def columns(self, val_cols: int) -> None:
        """
        Set number of columns.

        Parameters
        ----------
        val_cols : int
            Number of columns in `data`.
        """
        self._columns = val_cols

    def read(self, filename: PathLike) -> None:
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
                match[0] if (match := re.search(r"\[.+?\]", x)) else ""
                for x in head
            ]
            self.labels = [
                re.sub(r"[\[].*?[\]]", "", x).rstrip().lstrip() for x in head
            ]
        self.rows, self.columns = self.data.shape

    def summary(self) -> None:
        """
        Summary of timeseries contained, units, headers.
        """

        print(f"contains {self.columns} timeseries, each with {self.rows} elements")
        print("index label units")
        for index, (label, unit) in enumerate(zip(self.labels, self.units)):
            print(f"{index} {label} {unit}")
