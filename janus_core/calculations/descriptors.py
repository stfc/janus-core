"""Calculate MLIP descriptors for structures."""

from collections.abc import Sequence
from typing import Any, Optional

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.mixing import SumCalculator
import numpy as np

from janus_core.calculations.base import BaseCalculation
from janus_core.helpers.janus_types import (
    Architectures,
    ASEReadArgs,
    ASEWriteArgs,
    Devices,
    MaybeSequence,
    PathLike,
)
from janus_core.helpers.utils import none_to_dict, output_structs


class Descriptors(BaseCalculation):
    """
    Prepare and calculate MLIP descriptors for structures.

    Parameters
    ----------
    struct : Optional[MaybeSequence[Atoms]]
        ASE Atoms structure(s) to calculate descriptors for. Required if `struct_path`
        is None. Default is None.
    struct_path : Optional[PathLike]
        Path of structure to calculate descriptors for. Required if `struct` is None.
        Default is None.
    arch : Architectures
        MLIP architecture to use for calculations. Default is "mace_mp".
    device : Devices
        Device to run MLIP model on. Default is "cpu".
    model_path : Optional[PathLike]
        Path to MLIP model. Default is `None`.
    read_kwargs : Optional[ASEReadArgs]
        Keyword arguments to pass to ase.io.read. By default,
        read_kwargs["index"] is -1.
    calc_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to the selected calculator. Default is {}.
    set_calc : Optional[bool]
        Whether to set (new) calculators for structures. Default is None.
    attach_logger : bool
        Whether to attach a logger. Default is False.
    log_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to `config_logger`. Default is {}.
    tracker_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to `config_tracker`. Default is {}.
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

    Methods
    -------
    run()
        Calculate descriptors for structure(s)
    """

    def __init__(
        self,
        struct: Optional[MaybeSequence[Atoms]] = None,
        struct_path: Optional[PathLike] = None,
        arch: Architectures = "mace_mp",
        device: Devices = "cpu",
        model_path: Optional[PathLike] = None,
        read_kwargs: Optional[ASEReadArgs] = None,
        calc_kwargs: Optional[dict[str, Any]] = None,
        set_calc: Optional[bool] = None,
        attach_logger: bool = False,
        log_kwargs: Optional[dict[str, Any]] = None,
        tracker_kwargs: Optional[dict[str, Any]] = None,
        invariants_only: bool = True,
        calc_per_element: bool = False,
        calc_per_atom: bool = False,
        write_results: bool = False,
        write_kwargs: Optional[ASEWriteArgs] = None,
    ) -> None:
        """
        Initialise class.

        Parameters
        ----------
        struct : Optional[MaybeSequence[Atoms]]
            ASE Atoms structure(s) to calculate descriptors for. Required if
            `struct_path` is None. Default is None.
        struct_path : Optional[PathLike]
            Path of structure to calculate descriptors for. Required if `struct` is
            None. Default is None.
        arch : Architectures
            MLIP architecture to use for calculations. Default is "mace_mp".
        device : Devices
            Device to run MLIP model on. Default is "cpu".
        model_path : Optional[PathLike]
            Path to MLIP model. Default is `None`.
        read_kwargs : Optional[ASEReadArgs]
            Keyword arguments to pass to ase.io.read. By default,
            read_kwargs["index"] is -1.
        calc_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to the selected calculator. Default is {}.
        set_calc : Optional[bool]
            Whether to set (new) calculators for structures. Default is None.
        attach_logger : bool
            Whether to attach a logger. Default is False.
        log_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to `config_logger`. Default is {}.
        tracker_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to `config_tracker`. Default is {}.
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
        """
        (read_kwargs, write_kwargs) = none_to_dict((read_kwargs, write_kwargs))

        self.invariants_only = invariants_only
        self.calc_per_element = calc_per_element
        self.calc_per_atom = calc_per_atom
        self.write_results = write_results
        self.write_kwargs = write_kwargs

        # Read last image by default
        read_kwargs.setdefault("index", ":")

        # Initialise structures and logging
        super().__init__(
            calc_name=__name__,
            struct=struct,
            struct_path=struct_path,
            arch=arch,
            device=device,
            model_path=model_path,
            read_kwargs=read_kwargs,
            sequence_allowed=True,
            calc_kwargs=calc_kwargs,
            set_calc=set_calc,
            attach_logger=attach_logger,
            log_kwargs=log_kwargs,
            tracker_kwargs=tracker_kwargs,
        )

        if isinstance(self.struct, Atoms) and not self.struct.calc:
            raise ValueError("Please attach a calculator to `struct`.")
        if isinstance(self.struct, Sequence) and not any(
            image.calc for image in self.struct
        ):
            raise ValueError("Please attach a calculator to `struct`.")

        if isinstance(self.struct, Atoms):
            self._check_calculator(self.struct.calc)
        if isinstance(self.struct, Sequence):
            for image in self.struct:
                self._check_calculator(image.calc)

        # Set output file
        self.write_kwargs.setdefault("filename", None)
        self.write_kwargs["filename"] = self._build_filename(
            "descriptors.extxyz", filename=self.write_kwargs["filename"]
        ).absolute()

    @staticmethod
    def _check_calculator(calc: Calculator) -> None:
        """
        Ensure calculator has ability to calculate descriptors.

        Parameters
        ----------
        calc : Calculator
            ASE Calculator to calculate descriptors.
        """
        # If dispersion added to MLIP calculator, use MLIP calculator for descriptors
        if isinstance(calc, SumCalculator):
            if (
                len(calc.mixer.calcs) == 2
                and calc.mixer.calcs[1].name == "TorchDFTD3Calculator"
                and hasattr(calc.mixer.calcs[0], "get_descriptors")
            ):
                calc.get_descriptors = calc.mixer.calcs[0].get_descriptors

        if not hasattr(calc, "get_descriptors") or not callable(calc.get_descriptors):
            raise NotImplementedError(
                "The attached calculator does not currently support calculating "
                "descriptors"
            )

    def run(self) -> None:
        """Calculate descriptors for structure(s)."""
        if self.logger:
            self.logger.info("Starting descriptors calculation")
            self.logger.info("invariants_only: %s", self.invariants_only)
            self.logger.info("calc_per_element: %s", self.calc_per_element)
            self.logger.info("calc_per_atom: %s", self.calc_per_atom)
            self.tracker.start_task("Descriptors")

        if isinstance(self.struct, Sequence):
            for struct in self.struct:
                self._calc_descriptors(struct)
        else:
            self._calc_descriptors(self.struct)

        if self.logger:
            emissions = self.tracker.stop_task().emissions
            if isinstance(self.struct, Sequence):
                for image in self.struct:
                    image.info["emissions"] = emissions
            else:
                self.struct.info["emissions"] = emissions
            self.tracker.stop()
            self.logger.info("Descriptors calculation complete")

        output_structs(
            self.struct,
            struct_path=self.struct_path,
            write_results=self.write_results,
            write_kwargs=self.write_kwargs,
        )

    def _calc_descriptors(self, struct: Atoms) -> None:
        """
        Calculate MLIP descriptors a given structure.

        Parameters
        ----------
        struct : Atoms
            Structure to calculate descriptors for.
        """
        if "arch" in struct.calc.parameters:
            arch = struct.calc.parameters["arch"]
            label = f"{arch}_"
        else:
            label = ""

        # Calculate mean descriptor and save mean
        descriptors = struct.calc.get_descriptors(
            struct, invariants_only=self.invariants_only
        )
        descriptor = np.mean(descriptors)
        struct.info[f"{label}descriptor"] = descriptor

        if self.calc_per_element:
            elements = set(struct.get_chemical_symbols())
            for element in elements:
                pattern = [atom.index for atom in struct if atom.symbol == element]
                struct.info[f"{arch}_{element}_descriptor"] = np.mean(
                    descriptors[pattern, :]
                )

        if self.calc_per_atom:
            struct.arrays[f"{arch}_descriptors"] = np.mean(descriptors, axis=1)
