"""Calculate MLIP descriptors for structures."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ase import Atoms
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
from janus_core.helpers.mlip_calculators import check_calculator
from janus_core.helpers.struct_io import output_structs
from janus_core.helpers.utils import none_to_dict, track_progress


class Descriptors(BaseCalculation):
    """
    Prepare and calculate MLIP descriptors for structures.

    Parameters
    ----------
    struct
        ASE Atoms structure(s), or filepath to structure(s) to simulate.
    arch
        MLIP architecture to use for calculations. Default is `None`.
    device
        Device to run MLIP model on. Default is "cpu".
    model
        MLIP model label, path to model, or loaded model. Default is `None`.
    model_path
        Deprecated. Please use `model`.
    file_prefix
        Prefix for output filenames. Default is inferred from structure.
    read_kwargs
        Keyword arguments to pass to ase.io.read. By default,
        read_kwargs["index"] is -1.
    calc_kwargs
        Keyword arguments to pass to the selected calculator. Default is {}.
    attach_logger
        Whether to attach a logger. Default is True if "filename" is passed in
        log_kwargs, else False.
    log_kwargs
        Keyword arguments to pass to `config_logger`. Default is {}.
    track_carbon
        Whether to track carbon emissions of calculation. Default is True if
        attach_logger is True, else False.
    tracker_kwargs
        Keyword arguments to pass to `config_tracker`. Default is {}.
    invariants_only
        Whether only the invariant descriptors should be returned. Default is True.
    calc_per_element
        Whether to calculate mean descriptors for each element. Default is False.
    calc_per_atom
        Whether to calculate descriptors for each atom. Default is False.
    write_results
        True to write out structure with results of calculations. Default is False.
    write_kwargs
        Keyword arguments to pass to ase.io.write if saving structure with
        results of calculations. Default is {}.
    enable_progress_bar
        Whether to show a progress bar when applied to a file containing many
        structures. Default is False.
    """

    def __init__(
        self,
        struct: MaybeSequence[Atoms] | PathLike,
        arch: Architectures | None = None,
        device: Devices = "cpu",
        model: PathLike | None = None,
        model_path: PathLike | None = None,
        file_prefix: PathLike | None = None,
        read_kwargs: ASEReadArgs | None = None,
        calc_kwargs: dict[str, Any] | None = None,
        attach_logger: bool | None = None,
        log_kwargs: dict[str, Any] | None = None,
        track_carbon: bool | None = None,
        tracker_kwargs: dict[str, Any] | None = None,
        invariants_only: bool = True,
        calc_per_element: bool = False,
        calc_per_atom: bool = False,
        write_results: bool = False,
        write_kwargs: ASEWriteArgs | None = None,
        enable_progress_bar: bool = False,
    ) -> None:
        """
        Initialise class.

        Parameters
        ----------
        struct
            ASE Atoms structure(s), or filepath to structure(s) to simulate.
        arch
            MLIP architecture to use for calculations. Default is `None`.
        device
            Device to run MLIP model on. Default is "cpu".
        model
            MLIP model label, path to model, or loaded model. Default is `None`.
        model_path
            Deprecated. Please use `model`.
        file_prefix
            Prefix for output filenames. Default is inferred from structure.
        read_kwargs
            Keyword arguments to pass to ase.io.read. By default,
            read_kwargs["index"] is -1.
        calc_kwargs
            Keyword arguments to pass to the selected calculator. Default is {}.
        attach_logger
            Whether to attach a logger. Default is True if "filename" is passed in
            log_kwargs, else False.
        log_kwargs
            Keyword arguments to pass to `config_logger`. Default is {}.
        track_carbon
            Whether to track carbon emissions of calculation. Requires attach_logger.
            Default is True if attach_logger is True, else False.
        tracker_kwargs
            Keyword arguments to pass to `config_tracker`. Default is {}.
        invariants_only
            Whether only the invariant descriptors should be returned. Default is True.
        calc_per_element
            Whether to calculate mean descriptors for each element. Default is False.
        calc_per_atom
            Whether to calculate descriptors for each atom. Default is False.
        write_results
            True to write out structure with results of calculations. Default is False.
        write_kwargs
            Keyword arguments to pass to ase.io.write if saving structure with
            results of calculations. Default is {}.
        enable_progress_bar
            Whether to show a progress bar when applied to a file containing many
            structures. Default is False.
        """
        read_kwargs, write_kwargs = none_to_dict(read_kwargs, write_kwargs)

        self.invariants_only = invariants_only
        self.calc_per_element = calc_per_element
        self.calc_per_atom = calc_per_atom
        self.write_results = write_results
        self.write_kwargs = write_kwargs
        self.enable_progress_bar = enable_progress_bar

        # Read last image by default
        read_kwargs.setdefault("index", ":")

        # Initialise structures and logging
        super().__init__(
            struct=struct,
            calc_name=__name__,
            arch=arch,
            device=device,
            model=model,
            model_path=model_path,
            read_kwargs=read_kwargs,
            sequence_allowed=True,
            calc_kwargs=calc_kwargs,
            attach_logger=attach_logger,
            log_kwargs=log_kwargs,
            track_carbon=track_carbon,
            tracker_kwargs=tracker_kwargs,
            file_prefix=file_prefix,
        )

        if isinstance(self.struct, Atoms) and not self.struct.calc:
            raise ValueError("Please attach a calculator to `struct`.")
        if isinstance(self.struct, Sequence) and not any(
            image.calc for image in self.struct
        ):
            raise ValueError("Please attach a calculator to `struct`.")

        if isinstance(self.struct, Atoms):
            check_calculator(self.struct.calc, "get_descriptors")
        if isinstance(self.struct, Sequence):
            for image in self.struct:
                check_calculator(image.calc, "get_descriptors")

        # Set output file
        self.write_kwargs["filename"] = self._build_filename(
            "descriptors.extxyz", filename=self.write_kwargs.get("filename")
        )

    @property
    def output_files(self) -> None:
        """
        Dictionary of output file labels and paths.

        Returns
        -------
        dict[str, PathLike]
            Output file labels and paths.
        """
        return {
            "log": self.log_kwargs["filename"] if self.logger else None,
            "results": self.write_kwargs["filename"] if self.write_results else None,
        }

    def run(self) -> None:
        """Calculate descriptors for structure(s)."""
        if self.logger:
            self.logger.info("Starting descriptors calculation")
            self.logger.info("invariants_only: %s", self.invariants_only)
            self.logger.info("calc_per_element: %s", self.calc_per_element)
            self.logger.info("calc_per_atom: %s", self.calc_per_atom)
        if self.tracker:
            self.tracker.start_task("Descriptors")

        if isinstance(self.struct, Sequence):
            struct_sequence = self.struct

            if self.enable_progress_bar:
                struct_sequence = track_progress(
                    struct_sequence, "Calculating descriptors..."
                )

            for struct in struct_sequence:
                self._calc_descriptors(struct)

        else:
            self._calc_descriptors(self.struct)

        if self.logger:
            self.logger.info("Descriptors calculation complete")
        if self.tracker:
            emissions = self.tracker.stop_task().emissions
            if isinstance(self.struct, Sequence):
                for image in self.struct:
                    image.info["emissions"] = emissions
            else:
                self.struct.info["emissions"] = emissions
            self.tracker.stop()

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
        struct
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
