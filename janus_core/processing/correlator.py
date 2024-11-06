"""Module to correlate scalar data on-the-fly."""

from __future__ import annotations

from collections.abc import Iterable

from ase import Atoms
import numpy as np

from janus_core.helpers.janus_types import Observable


class Correlator:
    """
    Correlate scalar real values, <ab>.

    Parameters
    ----------
    blocks : int
        Number of correlation blocks.
    points : int
        Number of points per block.
    averaging : int
        Averaging window per block level.
    """

    def __init__(self, *, blocks: int, points: int, averaging: int) -> None:
        """
        Initialise an empty Correlator.

        Parameters
        ----------
        blocks : int
            Number of correlation blocks.
        points : int
            Number of points per block.
        averaging : int
            Averaging window per block level.
        """
        self._blocks = blocks
        self._points = points
        self._averaging = averaging
        self._max_block_used = 0
        self._min_dist = self._points / self._averaging

        self._accumulator = np.zeros((self._blocks, 2))
        self._count_accumulated = np.zeros(self._blocks, dtype=int)
        self._shift_index = np.zeros(self._blocks, dtype=int)
        self._shift = np.zeros((self._blocks, self._points, 2))
        self._shift_not_null = np.zeros((self._blocks, self._points), dtype=bool)
        self._correlation = np.zeros((self._blocks, self._points))
        self._count_correlated = np.zeros((self._blocks, self._points), dtype=int)

    def update(self, a: float, b: float) -> None:
        """
        Update the correlation, <ab>, with new values a and b.

        Parameters
        ----------
        a : float
            Newly observed value of left correland.
        b : float
            Newly observed value of right correland.
        """
        self._propagate(a, b, 0)

    def _propagate(self, a: float, b: float, block: int) -> None:
        """
        Propagate update down block hierarchy.

        Parameters
        ----------
        a : float
            Newly observed value of left correland/average.
        b : float
            Newly observed value of right correland/average.
        block : int
            Block in the hierachy being updated.
        """
        if block == self._blocks:
            return

        shift = self._shift_index[block]
        self._max_block_used = max(self._max_block_used, block)
        self._shift[block, shift, :] = a, b
        self._accumulator[block, :] += a, b
        self._shift_not_null[block, shift] = True
        self._count_accumulated[block] += 1

        if self._count_accumulated[block] == self._averaging:
            self._propagate(
                self._accumulator[block, 0] / self._averaging,
                self._accumulator[block, 1] / self._averaging,
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
        Return True if the shift registers have data.

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
        correlation : Iterable[float]
            The correlation values <a(t)b(t+t')>.
        lags : Iterable[float]]
            The correlation lag times t'.
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
                    lags[lag] = float(i) * float(self._averaging) ** k
                    lag += 1
        return (correlation[0:lag], lags[0:lag])


class Correlation:
    """
    Represents a user correlation, <ab>.

    Parameters
    ----------
    a : tuple[Observable, dict]
        Getter for a and kwargs.
    b : tuple[Observable, dict]
        Getter for b and kwargs.
    name : str
        Name of correlation.
    blocks : int
        Number of correlation blocks.
    points : int
        Number of points per block.
    averaging : int
        Averaging window per block level.
    update_frequency : int
        Frequency to update the correlation, md steps.
    """

    def __init__(
        self,
        a: Observable | tuple[Observable, tuple, dict],
        b: Observable | tuple[Observable, tuple, dict],
        name: str,
        blocks: int,
        points: int,
        averaging: int,
        update_frequency: int,
    ) -> None:
        """
        Initialise a correlation.

        Parameters
        ----------
        a : tuple[Observable, tuple, dict]
            Getter for a and kwargs.
        b : tuple[Observable, tuple, dict]
            Getter for b and kwargs.
        name : str
            Name of correlation.
        blocks : int
            Number of correlation blocks.
        points : int
            Number of points per block.
        averaging : int
            Averaging window per block level.
        update_frequency : int
            Frequency to update the correlation, md steps.
        """
        self.name = name
        if isinstance(a, tuple):
            self._get_a, self._a_args, self._a_kwargs = a
        else:
            self._get_a = a
            self._a_args, self._a_kwargs = (), {}

        if isinstance(b, tuple):
            self._get_b, self._b_args, self._b_kwargs = b
        else:
            self._get_b = b
            self._b_args, self._b_kwargs = (), {}

        self._correlator = Correlator(blocks=blocks, points=points, averaging=averaging)
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

    def update(self, atoms: Atoms) -> None:
        """
        Update a correlation.

        Parameters
        ----------
        atoms : Atoms
            Atoms object to observe values from.
        """
        self._correlator.update(
            self._get_a(atoms, *self._a_args, **self._a_kwargs),
            self._get_b(atoms, *self._b_args, **self._b_kwargs),
        )

    def get(self) -> tuple[Iterable[float], Iterable[float]]:
        """
        Get the correlation value and lags.

        Returns
        -------
        correlation : Iterable[float]
            The correlation values <a(t)b(t+t')>.
        lags : Iterable[float]]
            The correlation lag times t'.
        """
        return self._correlator.get()

    def __str__(self) -> str:
        """
        Return string representation of correlation.

        Returns
        -------
        str
            String representation.
        """
        return self.name
