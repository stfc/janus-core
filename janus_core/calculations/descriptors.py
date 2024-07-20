"""Calculate MLIP descriptors for structures."""

from collections.abc import Sequence
from typing import Any, Optional

from ase import Atoms
from ase.io import write
import numpy as np

from janus_core.helpers.janus_types import ASEWriteArgs, MaybeSequence
from janus_core.helpers.log import config_logger
from janus_core.helpers.utils import FileNameMixin, none_to_dict


class Descriptors(FileNameMixin):  # pylint: disable=too-few-public-methods
    """
    Prepare and calculate MLIP descriptors for structures.

    Parameters
    ----------
    struct : MaybeSequence[Atoms]
        Structure(s) to calculate descriptors for.
    struct_name : Optional[str]
        Name of structure. Default is None.
    invariants_only : bool
        Whether only the invariant descriptors should be returned. Default is True.
    calc_per_element : bool
        Whether to calculate mean descriptors for each element. Default is False.
    write_results : bool
        True to write out structure with results of calculations. Default is False.
    write_kwargs : Optional[ASEWriteArgs],
        Keyword arguments to pass to ase.io.write if saving structure with
        results of calculations. Default is {}.
    log_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to `config_logger`. Default is {}.
    """

    def __init__(
        self,
        struct: MaybeSequence[Atoms],
        struct_name: Optional[str] = None,
        invariants_only: bool = True,
        calc_per_element: bool = False,
        write_results: bool = False,
        write_kwargs: Optional[ASEWriteArgs] = None,
        log_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialise class.

        Parameters
        ----------
        struct : MaybeSequence[Atoms]
            Structure(s) to calculate descriptors for.
        struct_name : Optional[str]
            Name of structure. Default is None.
        invariants_only : bool
            Whether only the invariant descriptors should be returned. Default is True.
        calc_per_element : bool
            Whether to calculate mean descriptors for each element. Default is False.
        write_results : bool
            True to write out structure with results of calculations. Default is False.
        write_kwargs : Optional[ASEWriteArgs],
            Keyword arguments to pass to ase.io.write if saving structure with
            results of calculations. Default is {}.
        log_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to `config_logger`. Default is {}.
        """
        self.struct = struct
        self.struct_name = struct_name
        self.invariants_only = invariants_only
        self.calc_per_element = calc_per_element
        self.write_results = write_results

        if isinstance(self.struct, Sequence):
            if any(not image.calc for image in struct):
                raise ValueError(
                    "Please attach a calculator to all images in `struct`."
                )
        else:
            if not self.struct.calc:
                raise ValueError("Please attach a calculator to `struct`.")

        [write_kwargs, log_kwargs] = none_to_dict([write_kwargs, log_kwargs])
        self.write_kwargs = write_kwargs

        FileNameMixin.__init__(self, self.struct, self.struct_name, None)

        self.write_kwargs.setdefault(
            "filename",
            self._build_filename("descriptors.extxyz").absolute(),
        )

        log_kwargs.setdefault("name", __name__)
        self.logger = config_logger(**log_kwargs)

        if self.logger:
            self.logger.info("Starting descriptors calculation")

    def run(self) -> None:
        """Calculate."""

        if isinstance(self.struct, Sequence):
            for struct in self.struct:
                self._calc_descriptors(struct)
        else:
            self._calc_descriptors(self.struct)

        if self.logger:
            self.logger.info("Descriptors calculation complete")

        if self.write_results:
            write(images=self.struct, **self.write_kwargs, write_info=True)

    def _calc_descriptors(self, struct: Atoms) -> None:
        """
        Calculate MLIP descriptors for the given structure(s).

        Parameters
        ----------
        struct : Atoms
            Structure to calculate descriptors for.
        """
        if self.logger:
            self.logger.info("invariants_only: %s", self.invariants_only)
            self.logger.info("calc_per_element: %s", self.calc_per_element)

        # Calculate mean descriptor and save mean
        descriptors = struct.calc.get_descriptors(
            struct, invariants_only=self.invariants_only
        )
        descriptor = np.mean(descriptors)
        struct.info["descriptor"] = descriptor

        if self.calc_per_element:
            elements = set(struct.get_chemical_symbols())
            for element in elements:
                pattern = [atom.index for atom in struct if atom.symbol == element]
                struct.info[f"{element}_descriptor"] = np.mean(descriptors[pattern, :])
