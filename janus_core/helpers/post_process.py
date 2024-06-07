"""Module for post-processing trajectories."""

from collections.abc import Sequence
from itertools import combinations_with_replacement
from typing import Optional, Union

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


def compute_rdf(  # pylint: disable=too-many-locals,too-many-branches
    data: MaybeSequence[Atoms],
    ana: Optional[Analysis] = None,
    /,
    *,
    filename: Optional[MaybeSequence[PathLike]] = None,
    by_elements: bool = False,
    rmax: float = 2.5,
    nbins: int = 50,
    elements: Optional[MaybeSequence[Union[int, str]]] = None,
    index: SliceLike = (0, None, 1),
    volume: Optional[float] = None,
) -> Union[NDArray[float64], dict[tuple[str, str], NDArray[float64]]]:
    """
    Compute the rdf of data.

    Parameters
    ----------
    data : MaybeSequence[Atoms]
        Dataset to compute RDF of.
    ana : Optional[Analysis]
        ASE Analysis object for data reuse.
    filename : Optional[MaybeSequence[PathLike]]
        Filename(s) to output data to. Must match number of RDFs computed.
    by_elements : bool
        Split RDF into pairwise by elements group. Default is False.
        N.B. mixed RDFs (e.g. C-H) include all self-RDFs (e.g. C-C),
        to get the pure (C-H) RDF subtract the self-RDFs.
    rmax : float
        Maximum distance of RDF.
    nbins : int
        Number of bins to divide RDF.
    elements : Optional[MaybeSequence[Union[int, str]]]
        Make partial RDFs. If `by_elements` is true will filter to
        only display pairs in list.
    index : SliceLike
        Images to analyze as:
        `index` if `int`,
        `start`, `stop`, `step` if `tuple`,
        `slice` if `slice` or `range`.
    volume : Optional[float]
        Volume of cell for normalisation. Only needs to be provided
        if aperiodic cell. Default is (2*rmax)**3.

    Returns
    -------
    Union[NDArray[float64], dict[tuple[str, str], NDArray[float64]]]
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

        if filename is not None:
            if not isinstance(filename, Sequence):
                filename = (filename,)

            assert isinstance(filename, Sequence)

            if len(filename) != len(rdf):
                raise ValueError(
                    f"Different number of file names ({len(filename)}) "
                    f"to number of samples ({len(rdf)})"
                )

            for (dists, rdfs), out_path in zip(rdf.values(), filename):
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

        if filename is not None:
            if isinstance(filename, Sequence):
                if len(filename) != 1:
                    raise ValueError(
                        f"Different number of file names ({len(filename)}) "
                        "to number of samples (1)"
                    )
                filename = filename[0]

            with open(filename, "w", encoding="utf-8") as out_file:
                for dist, rdf_i in zip(*rdf):
                    print(dist, rdf_i, file=out_file)

    return rdf


def compute_vaf(
    data: Sequence[Atoms],
    filenames: Optional[MaybeSequence[PathLike]] = None,
    *,
    use_velocities: bool = False,
    fft: bool = False,
    index: SliceLike = (0, None, 1),
    filter_atoms: MaybeSequence[MaybeSequence[int]] = ((),),
) -> NDArray[float64]:
    """
    Compute the velocity autocorrelation function (VAF) of `data`.

    Parameters
    ----------
    data : Sequence[Atoms]
        Dataset to compute VAF of.
    filenames : Optional[MaybeSequence[PathLike]]
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
    filter_atoms : MaybeSequence[MaybeSequence[int]]
        Compute the VAF averaged over subsets of the system.
        Default is all atoms.

    Returns
    -------
    MaybeSequence[NDArray[float64]]
        Computed VAF(s).
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
        momenta = np.asarray(
            [datum.get_momenta() / datum.get_masses() for datum in data]
        )
    else:
        momenta = np.asarray([datum.get_momenta() for datum in data])

    n_steps = len(momenta)
    n_atoms = len(momenta[0])

    # If filter_atoms not specified use all atoms
    filter_atoms = [
        atom if atom and atom[0] else range(n_atoms) for atom in filter_atoms
    ]

    used_atoms = {atom for atoms in filter_atoms for atom in atoms}
    used_atoms = {j: i for i, j in enumerate(used_atoms)}

    vafs = np.sum(
        np.asarray(
            [
                [
                    np.correlate(momenta[:, j, i], momenta[:, j, i], "same")
                    for i in range(3)
                ]
                for j in used_atoms
            ]
        ),
        axis=1,
    )

    vafs /= n_steps - np.arange(n_steps)

    if fft:
        vafs = np.fft.fft(vafs, axis=0)

    vafs = [
        np.average([vafs[used_atoms[i]] for i in atoms], axis=0)
        for atoms in filter_atoms
    ]

    if filenames:
        for filename, vaf in zip(filenames, vafs):
            with open(filename, "w", encoding="utf-8") as out_file:
                print(*vaf, file=out_file, sep="\n")

    return vafs
