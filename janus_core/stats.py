"""
Module that reads the md stats output timeseries.
"""

import re

from numpy import float64, genfromtxt
from numpy.typing import NDArray

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
        self._data = None
        self._labels = None
        self._units = None
        self._source = source
        self.read()

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

    @property
    def source(self) -> PathLike:
        """
        Return filename which is the source of data.

        Returns
        -------
        PathLike
            Filename for the source of `data`.
        """
        return self._source

    @source.setter
    def source(self, val_source: PathLike) -> None:
        """
        Set the filename for the data source.

        Parameters
        ----------
        val_source : PathLike
            Filename for the `data` source.
        """
        self._source = val_source

    @property
    def labels(self) -> list[str]:
        """
        Return a list of labels for the columns in `data`.

        Returns
        -------
        list[str]
            List of labels for the columns in `data`.
        """
        return self._labels

    @labels.setter
    def labels(self, val_labels: list[str]) -> None:
        """
        Set labels for columns.

        Parameters
        ----------
        val_labels : list[str]
            List of labels for columns in `data`.
        """
        self._labels = val_labels

    @property
    def units(self) -> list[str]:
        """
        Return a list of units for the columns in `data`.

        Returns
        -------
        list[str]
            List of units for the columns in `data`.
        """
        return self._units

    @property
    def data(self) -> NDArray[float64]:
        """
        Return the timeseries `data`.

        Returns
        -------
        NDArray[float64]
            Data for timeseries in `data`.
        """
        return self._data

    @data.setter
    def data(self, val_data: NDArray[float64]) -> None:
        """
        Set data for timeseries.

        Parameters
        ----------
        val_data : NDArray[float64]
            Data for timeseries in `data`.
        """
        self._data = val_data

    def read(self) -> None:
        """
        Read MD stats and store them in `data`.
        """
        self.data = genfromtxt(self.source, skip_header=1)
        with open(self.source, "r+", encoding="utf-8") as file:
            head = file.readline().split("|")
            self.units = [
                match[0] if (match := re.search(r"\[.+?\]", x)) else "" for x in head
            ]
            self.labels = [re.sub(r"[\[].*?\]", "", x).strip() for x in head]
        self.rows, self.columns = self.data.shape

    def summary(self) -> None:
        """
        Summary of timeseries contained, units, headers.
        """

        print(f"contains {self.columns} timeseries, each with {self.rows} elements")
        print("index label units")
        for index, (label, unit) in enumerate(zip(self.labels, self.units)):
            print(f"{index} {label} {unit}")
