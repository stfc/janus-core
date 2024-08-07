"""Calculate MLIP descriptors for structures."""

from collections.abc import Sequence
from typing import Any, Optional

from ase import Atoms
from ase.io import write
import numpy as np

from janus_core.helpers.janus_types import ASEWriteArgs, MaybeSequence
from janus_core.helpers.log import config_logger, config_tracker
from janus_core.helpers.utils import FileNameMixin, none_to_dict


class Descriptors(FileNameMixin):
    # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """
    Prepare and calculate MLIP descriptors for structures.

    Parameters
    ----------
    struct : MaybeSequence[Atoms]
        Structure(s) to calculate descriptors for.
    invariants_only : bool
        Whether only the invariant descriptors should be returned. Default is True.
    calc_per_element : bool
        Whether to calculate mean descriptors for each element. Default is False.
    calc_per_atom : bool
        Whether to calculate descriptors for each atom. Default is False.
    write_results : bool
        True to write out structure with results of calculations. Default is False.
    write_kwargs : Optional[ASEWriteArgs],
        Keyword arguments to pass to ase.io.write if saving structure with
        results of calculations. Default is {}.
    log_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to `config_logger`. Default is {}.
    tracker_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to `config_tracker`. Default is {}.

    Attributes
    ----------
    logger : Optional[logging.Logger]
        Logger if log file has been specified.
    tracker : Optional[OfflineEmissionsTracker]
        Tracker if logging is enabled.

    Methods
    -------
    run()
        Calculate descriptors for structure(s)
    """

    def __init__(
        self,
        struct: MaybeSequence[Atoms],
        invariants_only: bool = True,
        calc_per_element: bool = False,
        calc_per_atom: bool = False,
        write_results: bool = False,
        write_kwargs: Optional[ASEWriteArgs] = None,
        log_kwargs: Optional[dict[str, Any]] = None,
        tracker_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialise class.

        Parameters
        ----------
        struct : MaybeSequence[Atoms]
            Structure(s) to calculate descriptors for.
        invariants_only : bool
            Whether only the invariant descriptors should be returned. Default is True.
        calc_per_element : bool
            Whether to calculate mean descriptors for each element. Default is False.
        calc_per_atom : bool
            Whether to calculate descriptors for each atom. Default is False.
        write_results : bool
            True to write out structure with results of calculations. Default is False.
        write_kwargs : Optional[ASEWriteArgs],
            Keyword arguments to pass to ase.io.write if saving structure with
            results of calculations. Default is {}.
        log_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to `config_logger`. Default is {}.
        tracker_kwargs : Optional[dict[str, Any]]
                Keyword arguments to pass to `config_tracker`. Default is {}.
        """
        (write_kwargs, log_kwargs, tracker_kwargs) = none_to_dict(
            (write_kwargs, log_kwargs, tracker_kwargs)
        )

        self.struct = struct
        self.invariants_only = invariants_only
        self.calc_per_element = calc_per_element
        self.calc_per_atom = calc_per_atom
        self.write_results = write_results
        self.write_kwargs = write_kwargs

        # Validate parameters
        if isinstance(self.struct, Sequence):
            if any(not image.calc for image in struct):
                raise ValueError(
                    "Please attach a calculator to all images in `struct`."
                )
        else:
            if not self.struct.calc:
                raise ValueError("Please attach a calculator to `struct`.")

        # Configure logging
        log_kwargs.setdefault("name", __name__)
        self.logger = config_logger(**log_kwargs)
        self.tracker = config_tracker(self.logger, **tracker_kwargs)

        # Set output file
        FileNameMixin.__init__(self, struct, None)
        self.write_kwargs.setdefault(
            "filename",
            self._build_filename("descriptors.extxyz").absolute(),
        )

    def run(self) -> None:
        """Calculate descriptors for structure(s)."""
        if self.logger:
            self.logger.info("Starting descriptors calculation")
            self.tracker.start()

        if isinstance(self.struct, Sequence):
            for struct in self.struct:
                self._calc_descriptors(struct)
        else:
            self._calc_descriptors(self.struct)

        if self.logger:
            self.tracker.stop()
            self.logger.info("Descriptors calculation complete")

        if self.write_results:
            write(images=self.struct, **self.write_kwargs, write_info=True)

    def _calc_descriptors(self, struct: Atoms) -> None:
        """
        Calculate MLIP descriptors a given structure.

        Parameters
        ----------
        struct : Atoms
            Structure to calculate descriptors for.
        """
        if self.logger:
            self.logger.info("invariants_only: %s", self.invariants_only)
            self.logger.info("calc_per_element: %s", self.calc_per_element)
            self.logger.info("calc_per_atom: %s", self.calc_per_atom)

        arch = struct.calc.parameters["arch"]

        # Calculate mean descriptor and save mean
        descriptors = struct.calc.get_descriptors(
            struct, invariants_only=self.invariants_only
        )
        descriptor = np.mean(descriptors)
        struct.info[f"{arch}_descriptor"] = descriptor

        if self.calc_per_element:
            elements = set(struct.get_chemical_symbols())
            for element in elements:
                pattern = [atom.index for atom in struct if atom.symbol == element]
                struct.info[f"{arch}_{element}_descriptor"] = np.mean(
                    descriptors[pattern, :]
                )

        if self.calc_per_atom:
            struct.arrays[f"{arch}_descriptors"] = np.mean(descriptors, axis=1)
