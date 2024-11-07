"""Module that reads the md stats output timeseries."""

from __future__ import annotations

from collections.abc import Iterator
from functools import singledispatchmethod
import re
from typing import TypeVar

from numpy import float64, genfromtxt, zeros
from numpy.typing import NDArray

from janus_core.helpers.janus_types import PathLike

try:  # Python >=3.10
    from types import EllipsisType
except ImportError:
    EllipsisType = type(...)

T = TypeVar("T")


class Stats:
    """
    Configure shared molecular dynamics simulation options.

    Parameters
    ----------
    source : PathLike
        File that contains the stats of a molecular dynamics simulation.
    """

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

    @singledispatchmethod
    def _getind(self, lab: T) -> T:
        """
        Convert an index label from str to int if present in labels.

        Otherwise return the input.

        Parameters
        ----------
        lab : str
            Label to find.

        Returns
        -------
        int
            Index of label in self or input if not string.

        Raises
        ------
        IndexError
            Label not found in labels.
        """
        return lab

    @_getind.register
    def _(self, lab: str) -> int:  # numpydoc ignore=GL08
        # Case-insensitive fuzzy match, only has to be `in` the labels
        index = next(
            (
                index
                for index, label in enumerate(self.labels)
                if lab.lower() in label.lower()
            ),
            None,
        )
        if index is None:
            raise IndexError(f"{lab} not found in labels")
        return index

    @singledispatchmethod
    def __getitem__(self, ind) -> NDArray[float64]:
        """
        Get member of stats data by label or index.

        Parameters
        ----------
        ind : Any
            Index or label to find.

        Returns
        -------
        NDArray[float64]
            Columns of data by label.

        Raises
        ------
        IndexError
            Invalid index type or label not found in labels.
        """
        raise IndexError(f"Unknown index {ind}")

    @__getitem__.register(int)
    @__getitem__.register(slice)
    @__getitem__.register(EllipsisType)
    def _(self, ind) -> NDArray[float64]:  # numpydoc ignore=GL08
        return self.data[:, ind]

    @__getitem__.register(list)
    @__getitem__.register(tuple)
    def _(self, ind) -> NDArray[float64]:  # numpydoc ignore=GL08
        ind = list(map(self._getind, ind))
        return self.data[:, ind]

    @__getitem__.register(str)
    def _(self, ind) -> NDArray[float64]:  # numpydoc ignore=GL08
        ind = self._getind(ind)
        return self[ind]

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

    @property
    def data_tags(self) -> Iterator[tuple[str, str]]:
        """
        Return the labels and their units together.

        Returns
        -------
        Iterator[tuple[str, str]]
            Zipped labels and units.
        """
        return zip(self.labels, self.units)

    def read(self) -> None:
        """Read MD stats and store them in `data`."""
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
        for index, (label, unit) in enumerate(self.data_tags):
            header += f"\n{index} {label} {unit}"
        return header
