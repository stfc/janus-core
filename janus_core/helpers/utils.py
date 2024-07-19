"""Utility functions for janus_core."""

from abc import ABC
from collections.abc import Collection
from pathlib import Path
from typing import Optional, get_args

from ase import Atoms
from ase.io import write
from spglib import get_spacegroup

from janus_core.helpers.janus_types import MaybeSequence, PathLike, Properties


class FileNameMixin(ABC):  # pylint: disable=too-few-public-methods
    """
    Provide mixin functions for standard filename handling.

    Parameters
    ----------
    struct : Atoms
        Structure from which to derive the default name if struct_name not provided.
    struct_name : Optional[str]
        Struct name to use.
    file_prefix : Optional[PathLike]
        Default prefix to use.
    *additional
        Components to add to file_prefix (joined by hyphens).

    Methods
    -------
    _get_default_struct_name(struct, struct_name)
         Return the name from the provided struct_name or generate from struct.
    _get_default_prefix(file_prefix, struct_name)
         Return a prefix from the provided file_prefix or from struct_name.
    _build_filename(suffix, *additional, filename, prefix_override)
         Return a standard format filename if filename not provided.
    """

    def __init__(
        self,
        struct: Atoms,
        struct_name: Optional[str],
        file_prefix: Optional[PathLike],
        *additional,
    ):
        """
        Provide mixin functions for standard filename handling.

        Parameters
        ----------
        struct : Atoms
            Structure from which to derive the default name if struct_name not provided.
        struct_name : Optional[str]
            Struct name to use.
        file_prefix : Optional[PathLike]
            Default prefix to use.
        *additional
            Components to add to file_prefix (joined by hyphens).
        """
        self.struct_name = self._get_default_struct_name(struct, struct_name)

        self.file_prefix = Path(
            self._get_default_prefix(file_prefix, self.struct_name, *additional)
        )

    @staticmethod
    def _get_default_struct_name(struct: Atoms, struct_name: Optional[str]) -> str:
        """
        Determine the default struct name from the structure or provided struct_name.

        Parameters
        ----------
        struct : Atoms
            Structure of system.
        struct_name : Optional[str]
            Name of structure.

        Returns
        -------
        str
            Structure name.
        """

        if struct_name is not None:
            return struct_name
        return struct.get_chemical_formula()

    @staticmethod
    def _get_default_prefix(
        file_prefix: Optional[PathLike], struct_name: str, *additional
    ) -> str:
        """
        Determine the default prefix from the structure name or provided file_prefix.

        Parameters
        ----------
        file_prefix : str
            Given file_prefix.
        struct_name : str
            Name of structure.
        *additional
            Components to add to file_prefix (joined by hyphens).

        Returns
        -------
        str
            File prefix.
        """
        if file_prefix is not None:
            return str(file_prefix)
        return "-".join((struct_name, *additional))

    def _build_filename(
        self,
        suffix: str,
        *additional,
        filename: Optional[PathLike] = None,
        prefix_override: Optional[str] = None,
    ) -> Path:
        """
        Set filename using the file prefix and suffix if not specified otherwise.

        Parameters
        ----------
        suffix : str
            Default suffix to use if `filename` is not specified.
        *additional
            Extra components to add to suffix (joined with hyphens).
        filename : Optional[PathLike]
            Filename to use, if specified. Default is None.
        prefix_override : Optional[str]
            Replace file_prefix if not None.

        Returns
        -------
        Path
            Filename specified, or default filename.
        """
        if filename:
            return Path(filename)
        prefix = (
            prefix_override if prefix_override is not None else str(self.file_prefix)
        )
        return Path("-".join((prefix, *filter(None, additional), suffix)))


def spacegroup(
    struct: Atoms, sym_tolerance: float = 0.001, angle_tolerance: float = -1.0
) -> str:
    """
    Determine the spacegroup for a structure.

    Parameters
    ----------
    struct : Atoms
        Structure as an ase Atoms object.
    sym_tolerance : float
        Atom displacement tolerance for spglib symmetry determination, in Å.
        Default is 0.001.
    angle_tolerance : float
        Angle precision for spglib symmetry determination, in degrees. Default is -1.0,
        which means an internally optimized routine is used to judge symmetry.

    Returns
    -------
    str
        Spacegroup name.
    """
    return get_spacegroup(
        cell=(
            struct.get_cell(),
            struct.get_scaled_positions(),
            struct.get_atomic_numbers(),
        ),
        symprec=sym_tolerance,
        angle_tolerance=angle_tolerance,
    )


def none_to_dict(dictionaries: list[Optional[dict]]) -> list[dict]:
    """
    Ensure dictionaries that may be None are dictionaires.

    Parameters
    ----------
    dictionaries : list[dict]
        List of dictionaries that be be None.

    Returns
    -------
    list[dict]
        Dictionaries set to {} if previously None.
    """
    for i, dictionary in enumerate(dictionaries):
        dictionaries[i] = dictionary if dictionary else {}
    return dictionaries


def dict_paths_to_strs(dictionary: dict) -> None:
    """
    Recursively iterate over dictionary, converting Path values to strings.

    Parameters
    ----------
    dictionary : dict
        Dictionary to be converted.
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dict_paths_to_strs(value)
        elif isinstance(value, Path):
            dictionary[key] = str(value)


def dict_remove_hyphens(dictionary: dict) -> dict:
    """
    Recursively iterate over dictionary, replacing hyphens with underscores in keys.

    Parameters
    ----------
    dictionary : dict
        Dictionary to be converted.

    Returns
    -------
    dict
        Dictionary with hyphens in keys replaced with underscores.
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = dict_remove_hyphens(value)
    return {k.replace("-", "_"): v for k, v in dictionary.items()}


def results_to_info(
    struct: Atoms,
    *,
    properties: Collection[Properties] = (),
    invalidate_calc: bool = False,
) -> None:
    """
    Copy or move MLIP calculated results to Atoms.info dict.

    Parameters
    ----------
    struct : Atoms
        Atoms object to copy or move calculated results to info dict.
    properties : Collection[Properties]
        Properties to copy from results to info dict. Default is ().
    invalidate_calc : bool
        Whether to remove all calculator results after copying properties to info dict.
        Default is False.
    """
    if not properties:
        properties = get_args(Properties)

    if struct.calc:
        # Set default architecture from calculator name
        arch = struct.calc.parameters["arch"]
        struct.info["arch"] = arch

        for key in properties & struct.calc.results.keys():
            tag = f"{arch}_{key}"
            value = struct.calc.results[key]
            if key == "forces":
                struct.arrays[tag] = value
            else:
                struct.info[tag] = value

        # Remove all calculator results
        if invalidate_calc:
            struct.calc.results = {}


def output_structs(
    images: MaybeSequence[Atoms],
    *,
    set_info: bool = True,
    write_results: bool = False,
    properties: Collection[Properties] = (),
    invalidate_calc: bool = False,
    **kwargs,
) -> None:
    """
    Copy or move calculated results to Atoms.info dict and/or write structures to file.

    Parameters
    ----------
    images : MaybeSequence[Atoms]
        Atoms object or a list of Atoms objects to interact with.
    set_info : bool
        True to set info dict from calculated results. Default is True.
    write_results : bool
        True to write out structure with results of calculations. Default is False.
    properties : Collection[Properties]
        Properties to copy from calculated results to info dict. Default is ().
    invalidate_calc : bool
        Whether to remove all calculator results after copying properties to info dict.
        Default is False.
    **kwargs
        Keyword arguments passed to ase.io.write.
    """
    if isinstance(images, Atoms):
        images = (images,)

    if set_info:
        for image in images:
            results_to_info(
                image, properties=properties, invalidate_calc=invalidate_calc
            )
    else:
        # Label architecture even if not copying results to info
        for image in images:
            if image.calc:
                image.info["arch"] = image.calc.parameters["arch"]

    if write_results:
        write(images=images, write_results=not invalidate_calc, **kwargs)
