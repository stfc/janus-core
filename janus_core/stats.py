"""
Module that reads the md stats output timeseries.
"""

import re

from numpy import float64, genfromtxt, zeros
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

        self._data = zeros((0, 0))
        self._labels = ()
        self._units = ()
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
        return self.data.shape[0]

    @property
    def columns(self) -> int:
        """
        Return number of columns.

        Returns
        -------
        int
            Number of columns in `data`.
        """
        return self.data.shape[1]

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

    @property
    def labels(self) -> tuple[str, ...]:
        """
        Return a list of labels for the columns in `data`.

        Returns
        -------
        tuple[str, ...]
            List of labels for the columns in `data`.
        """
        return self._labels

    @property
    def units(self) -> tuple[str, ...]:
        """
        Return a list of units for the columns in `data`.

        Returns
        -------
        tuple[str, ...]
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

    def read(self) -> None:
        """
        Read MD stats and store them in `data`.
        """
        self._data = genfromtxt(self.source, skip_header=1)
        with open(self.source, "r+", encoding="utf-8") as file:
            head = file.readline().split("|")
            self._units = tuple(
                match[1] if (match := re.search(r"\[(.+?)\]", x)) else "" for x in head
            )
            self._labels = tuple(re.sub(r"\[.*?\]", "", x).strip() for x in head)

    def __repr__(self) -> str:
        """
        Summary of timeseries contained, units, headers.

        Returns
        -------
        str
           Summary of the `data`.
        """

        header = f"contains {self.columns} timeseries, each with {self.rows} elements"
        header += "\nindex label units"
        for index, (label, unit) in enumerate(zip(self.labels, self.units)):
            header += f"\n{index} {label} {unit}"
        return header
