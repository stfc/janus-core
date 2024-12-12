"""Module to correlate scalar data on-the-fly."""

from __future__ import annotations

from collections.abc import Iterable

from ase import Atoms
import numpy as np

from janus_core.processing.observables import Observable


class Correlator:
    """
    Correlate scalar real values, <ab>.

    Implements the algorithm detailed in https://doi.org/10.1063/1.3491098.

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
        # Number of resoluion levels.
        self._blocks = blocks
        # Data points a each resolution.
        self._points = points
        # Coarse-graining between resolution levels.
        self._averaging = averaging
        # Which levels have been updated with data.
        self._max_block_used = 0
        # First point in a block relevant for correlation updates. 
        self._min_dist = self._points / self._averaging

        # Sum of data seen for calculating the average between blocks.
        self._accumulator = np.zeros((self._blocks, 2))
        # Data points accumulated at each block.
        self._count_accumulated = np.zeros(self._blocks, dtype=int)
        # Current position in each blocks data store. 
        self._shift_index = np.zeros(self._blocks, dtype=int)
        # Rolling data store for each block.
        self._shift = np.zeros((self._blocks, self._points, 2))
        # If data is stored in this block's rolling data store, at a given index.
        self._shift_not_null = np.zeros((self._blocks, self._points), dtype=bool)
        # Running correlation values.
        self._correlation = np.zeros((self._blocks, self._points))
        # Count of correlation updates.
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
            # Hit the end of the data structure.
            return

        shift = self._shift_index[block]
        self._max_block_used = max(self._max_block_used, block)
        
        # Update the rolling data store, and accumulate
        self._shift[block, shift, :] = a, b
        self._accumulator[block, :] += a, b
        self._shift_not_null[block, shift] = True
        self._count_accumulated[block] += 1

        if self._count_accumulated[block] == self._averaging:
            # Hit the coarse graining threshold, the next block can be updated.
            self._propagate(
                self._accumulator[block, 0] / self._averaging,
                self._accumulator[block, 1] / self._averaging,
                block + 1,
            )
            # Reset the accumulator at this block level.
            self._accumulator[block, :] = 0.0
            self._count_accumulated[block] = 0

        # Update the correlation.
        i = self._shift_index[block]
        if block == 0:
            # Need to multiply all (full resolution).
            j = i
            for point in range(self._points):
                if self._shifts_valid(block, i, j):
                    # Correlate at this lag.
                    self._correlation[block, point] += (
                        self._shift[block, i, 0] * self._shift[block, j, 1]
                    )
                    self._count_correlated[block, point] += 1
                j -= 1
                if j < 0:
                    # Wrap to start of rolling data store.
                    j += self._points
        else:
            # Only need to update after points/averaging.
            for point in range(self._min_dist, self._points):
                if j < 0:
                    j = j + self._points
                if self._shifts_valid(block, i, j):
                   # Correlate at this lag.
                    self._correlation[block, point] += (
                        self._shift[block, i, 0] * self._shift[block, j, 1]
                    )
                    self._count_correlated[block, point] += 1
                j = j - 1
        # Update the rolling data store.
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

    def get_lags(self) -> Iterable[float]:
        """
        Obtain the correlation lag times.

        Returns
        -------
        Iterable[float]
            The correlation lag times.
        """
        lags = np.zeros(self._points * self._blocks)

        lag = 0
        for i in range(self._points):
            if self._count_correlated[0, i] > 0:
                # Data has been correlated, at full resolution.
                lags[lag] = i
                lag += 1
        for k in range(1, self._max_block_used):
            for i in range(self._min_dist, self._points):
                if self._count_correlated[k, i] > 0:
                    # Data has been correlated at a coarse-grained level.
                    lags[lag] = float(i) * float(self._averaging) ** k
                    lag += 1
        return lags[0:lag]

    def get_value(self) -> Iterable[float]:
        """
        Obtain the correlation value.

        Returns
        -------
        Iterable[float]
            The correlation values <a(t)b(t+t')>.
        """
        correlation = np.zeros(self._points * self._blocks)

        lag = 0
        for i in range(self._points):
            if self._count_correlated[0, i] > 0:
                # Data has been correlated at full resolution.
                correlation[lag] = (
                    self._correlation[0, i] / self._count_correlated[0, i]
                )
                lag += 1
        for k in range(1, self._max_block_used):
            for i in range(self._min_dist, self._points):
                # Indices at less than points/averaging are accounted in the previous block.
                if self._count_correlated[k, i] > 0:
                    # Data has been correlated at a coarse-grained level.
                    correlation[lag] = (
                        self._correlation[k, i] / self._count_correlated[k, i]
                    )
                    lag += 1
        return correlation[0:lag]


class Correlation:
    """
    Represents a user correlation, <ab>.

    Parameters
    ----------
    a : Observable
        Observable for a.
    b : Observable
        Observable for b.
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
        *,
        a: Observable,
        b: Observable,
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
        a : Observable
            Observable for a.
        b : Observable
            Observable for b.
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
        self.blocks = blocks
        self.points = points
        self.averaging = averaging
        self._get_a = a
        self._get_b = b

        self._correlators = None
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
        # All pairs of data to be correlated.
        value_pairs = zip(self._get_a(atoms), self._get_b(atoms))
        if self._correlators is None:
            # Initialise correlators automatically.
            self._correlators = [
                Correlator(
                    blocks=self.blocks, points=self.points, averaging=self.averaging
                )
                for _ in range(len(self._get_a(atoms)))
            ]
        for corr, values in zip(self._correlators, value_pairs):
            corr.update(*values)

    def get(self) -> tuple[Iterable[float], Iterable[float]]:
        """
        Get the correlation value and lags, averaging over atoms if applicable.

        Returns
        -------
        correlation : Iterable[float]
            The correlation values <a(t)b(t+t')>.
        lags : Iterable[float]]
            The correlation lag times t'.
        """
        if self._correlators:
            lags = self._correlators[0].get_lags()
            return np.mean([cor.get_value() for cor in self._correlators], axis=0), lags
        return [], []

    def __str__(self) -> str:
        """
        Return string representation of correlation.

        Returns
        -------
        str
            String representation.
        """
        return self.name
