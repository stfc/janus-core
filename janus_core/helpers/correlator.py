"""
Module to correlate scalar data on-the-fly.
"""

from abc import abstractmethod
from collections.abc import Iterable
from typing import Union

from ase import units
from ase.md.langevin import Langevin
from ase.md.npt import NPT as ASE_NPT
from ase.md.verlet import VelocityVerlet
import numpy as np


class Correlator:
    """
    Correlate scalar real values.

    Parameters
    ----------
    blocks : int
        Number of correlation blocks.
    points : int
        Number of points per block.
    window : int
        Averaging window per block level.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, *, blocks: int, points: int, window: int):
        """
        Initialise an empty Correlator.

        Parameters
        ----------
        blocks : int
            Number of correlation blocks.
        points : int
            Number of points per block.
        window : int
            Averaging window per block level.
        """
        self._blocks = blocks
        self._points = points
        self._window = window
        self._max_block_used = 0
        self._min_dist = self._points / self._window

        self._accumulator = np.zeros((self._blocks, 2))
        self._count_accumulated = np.zeros(self._blocks).astype(int)
        self._shift_index = np.zeros(self._blocks).astype(int)
        self._shift = np.zeros((self._blocks, self._points, 2))
        self._shift_not_null = np.zeros((self._blocks, self._points)).astype(bool)
        self._correlation = np.zeros((self._blocks, self._points))
        self._count_correlated = np.zeros((self._blocks, self._points)).astype(int)

    def update(self, observable_a: float, observable_b: float):
        """
        Update the correlation with new values a and b.

        Parameters
        ----------
        observable_a : float
            Newly observed value of left correland.
        observable_b : float
            Newly observed value of right correland.
        """
        self._add(observable_a, observable_b, 0)

    def _add(self, observable_a: float, observable_b: float, block: int):
        """
        Propagate update down block hierarchy.

        Parameters
        ----------
        observable_a : float
            Newly observed value of left correland/average.
        observable_b : float
            Newly observed value of right correland/average.
        block : int
            Block in the hierachy being updated.
        """
        if block == self._blocks:
            return

        shift = self._shift_index[block]
        self._max_block_used = max(self._max_block_used, block)
        self._shift[block, shift, :] = observable_a, observable_b
        self._accumulator[block, :] += observable_a, observable_b
        self._shift_not_null[block, shift] = True
        self._count_accumulated[block] += 1

        if self._count_accumulated[block] == self._window:
            self._add(
                self._accumulator[block, 0] / self._window,
                self._accumulator[block, 1] / self._window,
                block + 1,
            )
            self._accumulator[block, :] = 0.0
            self._count_accumulated[block] = 0

        i = self._shift_index[block]
        if block == 0:
            j = i
            for point in range(self._points):
                if self._shifts_valid(block, i, j):
                    self._correlation[block, point] += (
                        self._shift[block, i, 0] * self._shift[block, j, 1]
                    )
                    self._count_correlated[block, point] += 1
                j -= 1
                if j < 0:
                    j += self._points
        else:
            for point in range(self._min_dist, self._points):
                if j < 0:
                    j = j + self._points
                if self._shifts_valid(block, i, j):
                    self._correlation[block, point] += (
                        self._shift[block, i, 0] * self._shift[block, j, 1]
                    )
                    self._count_correlated[block, point] += 1
                j = j - 1
        self._shift_index[block] = (self._shift_index[block] + 1) % self._points

    def _shifts_valid(self, block: int, p_i: int, p_j: int) -> bool:
        """
        True if the shift registers have data.

        Parameters
        ----------
        block : int
            Block to check the shift register of.
        p_i : int
            Index i in the shift (left correland).
        p_j : int
            Index j in the shift (right correland).

        Returns
        -------
        bool
            Whether the shift indices have data.
        """
        return self._shift_not_null[block, p_i] and self._shift_not_null[block, p_j]

    def get(self) -> tuple[Iterable[float], Iterable[float]]:
        """
        Obtain the correlation and lag times.

        Returns
        -------
        tuple[Iterable[float], Iterable[float]]
            The correlation and lags.
        """
        correlation = np.zeros(self._points * self._blocks)
        lags = np.zeros(self._points * self._blocks)

        lag = 0
        for i in range(self._points):
            if self._count_correlated[0, i] > 0:
                correlation[lag] = (
                    self._correlation[0, i] / self._count_correlated[0, i]
                )
                lags[lag] = i
                lag += 1
        for k in range(1, self._max_block_used):
            for i in range(self._min_dist, self._points):
                if self._count_correlated[k, i] > 0:
                    correlation[lag] = (
                        self._correlation[k, i] / self._count_correlated[k, i]
                    )
                    lags[lag] = float(i) * float(self._window) ** k
                    lag += 1
        return (correlation[0:lag], lags[0:lag])


# pylint: disable=R0903
class Observable:
    """An abstract observable quantity."""

    @staticmethod
    def _component_to_index(component: str, *, voigt: bool = False) -> int:
        """
        Convert a component code to an index.

        Parameters
        ----------
        component : str
            The component code, x, y, z, ....
        voigt : bool
            Use voigt indexing for matrices.

        Returns
        -------
        int
            An index used internally to access the component.
        """
        vector = {"x": 0, "y": 1, "z": 2}
        matrix = {
            "xx": 0,
            "xy": 1,
            "xz": 2,
            "yx": 3,
            "yy": 4,
            "yz": 5,
            "zx": 6,
            "zy": 7,
            "zz": 8,
        }
        matrix_voigt = {"xx": 0, "yy": 1, "zz": 2, "yz": 3, "xz": 4, "xy": 5}

        if voigt:
            return matrix_voigt[component]
        if component in vector:
            return vector[component]
        return matrix[component]

    @abstractmethod
    # pylint: disable=C0103
    def get(self, md: Union[Langevin, VelocityVerlet, ASE_NPT]) -> float:
        """
        Get an observable component from md.

        Parameters
        ----------
        md : Union[Langevin, VelocityVerlet, ASE_NPT]
            A MolecularDynamics object.

        Returns
        -------
        float
            A scalar value of the observable (component).
        """
        return


class Correlation:
    """
    Represents a user correlation.

    Parameters
    ----------
    observable_a : Observable
        Quantity a correlated.
    observable_b : Observable
        Quantity b correlated.
    blocks : int
        Number of correlation blocks.
    points : int
        Number of points per block.
    window : int
        Averaging window per block level.
    update_frequency : int
        Frequency to update the correlation, md steps.
    """

    def __init__(
        self,
        observable_a: Observable,
        observable_b: Observable,
        blocks: int,
        points: int,
        window: int,
        update_frequency: int,
    ):
        """
        Initialise a correlation.

        Parameters
        ----------
        observable_a : Observable
            Quantity a correlated.
        observable_b : Observable
            Quantity b correlated.
        blocks : int
            Number of correlation blocks.
        points : int
            Number of points per block.
        window : int
            Averaging window per block level.
        update_frequency : int
            Frequency to update the correlation, md steps.
        """
        self.observable_a = observable_a
        self.observable_b = observable_b
        self._correlator = Correlator(blocks=blocks, points=points, window=window)
        self._update_frequency = update_frequency

    @property
    def update_frequency(self) -> int:
        """
        Get update frequency.

        Returns
        -------
        int
            Correlation update frequency.
        """

        return self._update_frequency

    # pylint: disable=C0103
    def update(self, md: Union[Langevin, VelocityVerlet, ASE_NPT]):
        """
        Update a correlation.

        Parameters
        ----------
        md : Union[Langevin, VelocityVerlet, ASE_NPT]
            MolecularDynamics object to observe values from.
        """
        self._correlator.update(self.observable_a.get(md), self.observable_b.get(md))

    def get(self) -> tuple[Iterable[float], Iterable[float]]:
        """
        Get the correlation value and lags.

        Returns
        -------
        tuple[Iterable[float], Iterable[float]]
            Correlation value and lags.
        """
        return self._correlator.get()

    def __str__(self) -> str:
        """
        String representation of correlation.

        Returns
        -------
        str
            String representation.
        """
        return f"{str(self.observable_a)}-{str(self.observable_b)}"


class Stress(Observable):
    """
    A stress observable.

    Parameters
    ----------
    component : str
        Component of the stress tensor.
    """

    def __init__(self, component: str):
        """
        A stress observable.

        Parameters
        ----------
        component : str
            Component of the stress tensor.
        """
        self.component = component
        self._index = self._component_to_index(component, voigt=False)

    def get(self, md: Union[Langevin, VelocityVerlet, ASE_NPT]) -> float:
        """
        Get an observable component from md.

        Parameters
        ----------
        md : Union[Langevin, VelocityVerlet, ASE_NPT]
            The md object to observe.

        Returns
        -------
        float
            The observed stress value.
        """
        return (
            md.atoms.get_stress(include_ideal_gas=True, voigt=False).flatten()[
                self._index
            ]
            / units.bar
        )

    def __str__(self) -> str:
        """
        String representation of Stress observable.

        Returns
        -------
        str
            String representation.
        """
        return f"stress_{self.component}"


def option_to_observable(option: str) -> Observable:
    """
    Convert an option string into an Observable.

    Parameters
    ----------
    option : str
        A user correlation option.

    Returns
    -------
    Observable
        The internal representation.
    """
    observable, component = option.split("_")
    if observable.lower() in ["s", "stress"]:
        return Stress(component)

    raise ValueError(f"Could not match {option} to correlation observable")
