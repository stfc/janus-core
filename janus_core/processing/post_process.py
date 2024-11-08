"""Module for post-processing trajectories."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations_with_replacement

from ase import Atoms
from ase.geometry.analysis import Analysis
import numpy as np
from numpy import float64
from numpy.typing import NDArray

from janus_core.helpers.janus_types import (
    MaybeSequence,
    PathLike,
    SliceLike,
    StartStopStep,
)


def _process_index(index: SliceLike) -> StartStopStep:
    """
    Standarize `SliceLike`s into tuple of `start`, `stop`, `step`.

    Parameters
    ----------
    index : SliceLike
        `SliceLike` to standardize.

    Returns
    -------
    StartStopStep
        Standardized `SliceLike` as `start`, `stop`, `step` triplet.
    """
    if isinstance(index, int):
        if index == -1:
            return (index, None, 1)
        return (index, index + 1, 1)

    if isinstance(index, (slice, range)):
        return (index.start, index.stop, index.step)

    return index


def compute_rdf(
    data: MaybeSequence[Atoms],
    ana: Analysis | None = None,
    /,
    *,
    filenames: MaybeSequence[PathLike] | None = None,
    by_elements: bool = False,
    rmax: float = 2.5,
    nbins: int = 50,
    elements: MaybeSequence[int | str] | None = None,
    index: SliceLike = (0, None, 1),
    volume: float | None = None,
) -> NDArray[float64] | dict[tuple[str, str] | NDArray[float64]]:
    """
    Compute the rdf of data.

    Parameters
    ----------
    data : MaybeSequence[Atoms]
        Dataset to compute RDF of.
    ana : Analysis | None
        ASE Analysis object for data reuse.
    filenames : MaybeSequence[PathLike] | None
        Filenames to output data to. Must match number of RDFs computed.
    by_elements : bool
        Split RDF into pairwise by elements group. Default is False.
        N.B. mixed RDFs (e.g. C-H) include all self-RDFs (e.g. C-C),
        to get the pure (C-H) RDF subtract the self-RDFs.
    rmax : float
        Maximum distance of RDF.
    nbins : int
        Number of bins to divide RDF.
    elements : MaybeSequence[int | str] | None
        Make partial RDFs. If `by_elements` is true will filter to
        only display pairs in list.
    index : SliceLike
        Images to analyze as:
        `index` if `int`,
        `start`, `stop`, `step` if `tuple`,
        `slice` if `slice` or `range`.
    volume : float | None
        Volume of cell for normalisation. Only needs to be provided
        if aperiodic cell. Default is (2*rmax)**3.

    Returns
    -------
    NDArray[float64] | dict[tuple[str, str] | NDArray[float64]]
        If `by_elements` is true returns a `dict` of RDF by element pairs.
        Otherwise returns RDF of total system filtered by elements.
    """
    index = _process_index(index)

    if not isinstance(data, Sequence):
        data = [data]

    if elements is not None and not isinstance(elements, Sequence):
        elements = (elements,)

    if (  # If aperiodic, assume volume of a cube encompassing rmax sphere.
        not all(data[0].get_pbc()) and volume is None
    ):
        volume = (2 * rmax) ** 3

    if ana is None:
        ana = Analysis(data)

    if by_elements:
        elements = (
            tuple(sorted(set(data[0].get_chemical_symbols())))
            if elements is None
            else elements
        )

        rdf = {
            element: ana.get_rdf(
                rmax=rmax,
                nbins=nbins,
                elements=element,
                imageIdx=slice(*index),
                return_dists=True,
                volume=volume,
            )
            for element in combinations_with_replacement(elements, 2)
        }

        # Compute RDF average
        rdf = {
            element: (rdf[0][1], np.average([rdf_i[0] for rdf_i in rdf], axis=0))
            for element, rdf in rdf.items()
        }

        if filenames is not None:
            if isinstance(filenames, str) or not isinstance(filenames, Sequence):
                filenames = (filenames,)

            assert isinstance(filenames, Sequence)

            if len(filenames) != len(rdf):
                raise ValueError(
                    f"Different number of file names ({len(filenames)}) "
                    f"to number of samples ({len(rdf)})"
                )

            for (dists, rdfs), out_path in zip(rdf.values(), filenames):
                with open(out_path, "w", encoding="utf-8") as out_file:
                    for dist, rdf_i in zip(dists, rdfs):
                        print(dist, rdf_i, file=out_file)

    else:
        rdf = ana.get_rdf(
            rmax=rmax,
            nbins=nbins,
            elements=elements,
            imageIdx=slice(*index),
            return_dists=True,
            volume=volume,
        )

        assert isinstance(rdf, list)

        # Compute RDF average
        rdf = rdf[0][1], np.average([rdf_i[0] for rdf_i in rdf], axis=0)

        if filenames is not None:
            if isinstance(filenames, Sequence):
                if len(filenames) != 1:
                    raise ValueError(
                        f"Different number of file names ({len(filenames)}) "
                        "to number of samples (1)"
                    )
                filenames = filenames[0]

            with open(filenames, "w", encoding="utf-8") as out_file:
                for dist, rdf_i in zip(*rdf):
                    print(dist, rdf_i, file=out_file)

    return rdf


def compute_vaf(
    data: Sequence[Atoms],
    filenames: MaybeSequence[PathLike] | None = None,
    *,
    use_velocities: bool = False,
    fft: bool = False,
    index: SliceLike = (0, None, 1),
    filter_atoms: MaybeSequence[MaybeSequence[int | None]] = ((None),),
    time_step: float = 1.0,
) -> tuple[NDArray[float64], list[NDArray[float64]]]:
    """
    Compute the velocity autocorrelation function (VAF) of `data`.

    Parameters
    ----------
    data : Sequence[Atoms]
        Dataset to compute VAF of.
    filenames : MaybeSequence[PathLike] | None
        If present, dump resultant VAF to file.
    use_velocities : bool
        Compute VAF using velocities rather than momenta.
        Default is False.
    fft : bool
        Compute the fourier transformed VAF.
        Default is False.
    index : SliceLike
        Images to analyze as `start`, `stop`, `step`.
        Default is all images.
    filter_atoms : MaybeSequence[MaybeSequence[int | None]]
        Compute the VAF averaged over subsets of the system.
        Default is all atoms.
    time_step : float
        Time step for scaling lags to align with input data.
        Default is 1 (i.e. no scaling).

    Returns
    -------
    lags : numpy.ndarray
        Lags at which the VAFs have been computed.
    vafs : list[numpy.ndarray]
        Computed VAF(s).

    Notes
    -----
    `filter_atoms` is given as a series of sequences of atoms, where
    each element in the series denotes a VAF subset to calculate and
    each sequence determines the atoms (by index) to be included in that VAF.

    E.g.

    .. code-block: Python

        # Species indices in cell
        na = (1, 3, 5, 7)
        cl = (2, 4, 6, 8)

        compute_vaf(..., filter_atoms=(na, cl))

    Would compute separate VAFs for each species.

    By default, one VAF will be computed for all atoms in the structure.
    """
    # Ensure if passed scalars they are turned into correct dimensionality
    if not isinstance(filter_atoms, Sequence):
        filter_atoms = (filter_atoms,)
    if not isinstance(filter_atoms[0], Sequence):
        filter_atoms = (filter_atoms,)

    if filenames and not isinstance(filenames, Sequence):
        filenames = (filenames,)

        if len(filenames) != len(filter_atoms):
            raise ValueError(
                f"Different number of file names ({len(filenames)}) "
                f"to number of samples ({len(filter_atoms)})"
            )

    # Extract requested data
    index = _process_index(index)
    data = data[slice(*index)]

    if use_velocities:
        momenta = np.asarray([datum.get_velocities() for datum in data])
    else:
        momenta = np.asarray([datum.get_momenta() for datum in data])

    n_steps = len(momenta)
    n_atoms = len(momenta[0])

    # If filter_atoms not specified use all atoms
    filter_atoms = [
        atom if atom[0] is not None else range(n_atoms) for atom in filter_atoms
    ]

    used_atoms = {atom for atoms in filter_atoms for atom in atoms}
    used_atoms = {j: i for i, j in enumerate(used_atoms)}

    vafs = np.sum(
        np.asarray(
            [
                [
                    np.correlate(momenta[:, j, i], momenta[:, j, i], "full")[
                        n_steps - 1 :
                    ]
                    for i in range(3)
                ]
                for j in used_atoms
            ]
        ),
        axis=1,
    )

    vafs /= n_steps - np.arange(n_steps)

    lags = np.arange(n_steps) * time_step

    if fft:
        vafs = np.fft.fft(vafs, axis=0)
        lags = np.fft.fftfreq(n_steps, time_step)

    vafs = (
        lags,
        [
            np.average([vafs[used_atoms[i]] for i in atoms], axis=0)
            for atoms in filter_atoms
        ],
    )

    if filenames:
        for vaf, filename in zip(vafs[1], filenames):
            with open(filename, "w", encoding="utf-8") as out_file:
                for lag, dat in zip(lags, vaf):
                    print(lag, dat, file=out_file)

    return vafs
