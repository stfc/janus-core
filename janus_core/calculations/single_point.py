"""Prepare and perform single point calculations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, get_args

from ase import Atoms
from numpy import ndarray

from janus_core.calculations.base import BaseCalculation
from janus_core.helpers.janus_types import (
    Architectures,
    ASEReadArgs,
    CalcResults,
    Devices,
    MaybeList,
    MaybeSequence,
    OutputKwargs,
    PathLike,
    Properties,
)
from janus_core.helpers.mlip_calculators import check_calculator
from janus_core.helpers.struct_io import output_structs
from janus_core.helpers.utils import none_to_dict


class SinglePoint(BaseCalculation):
    """
    Prepare and perform single point calculations.

    Parameters
    ----------
    struct : MaybeSequence[Atoms] | None
        ASE Atoms structure(s) to simulate. Required if `struct_path` is None.
        Default is None.
    struct_path : PathLike | None
        Path of structure to simulate. Required if `struct` is None.
        Default is None.
    arch : Architectures
        MLIP architecture to use for single point calculations.
        Default is "mace_mp".
    device : Devices
        Device to run model on. Default is "cpu".
    model_path : PathLike | None
        Path to MLIP model. Default is `None`.
    read_kwargs : ASEReadArgs
        Keyword arguments to pass to ase.io.read. By default,
        read_kwargs["index"] is ":".
    calc_kwargs : dict[str, Any] | None
        Keyword arguments to pass to the selected calculator. Default is {}.
    set_calc : bool | None
        Whether to set (new) calculators for structures. Default is None.
    attach_logger : bool
        Whether to attach a logger. Default is False.
    log_kwargs : dict[str, Any] | None
            Keyword arguments to pass to `config_logger`. Default is {}.
    track_carbon : bool
        Whether to track carbon emissions of calculation. Default is True.
    tracker_kwargs : dict[str, Any] | None
            Keyword arguments to pass to `config_tracker`. Default is {}.
    properties : MaybeSequence[Properties]
        Physical properties to calculate. If not specified, "energy",
        "forces", and "stress" will be returned.
    write_results : bool
        True to write out structure with results of calculations. Default is False.
    write_kwargs : OutputKwargs | None
        Keyword arguments to pass to ase.io.write if saving structure with results of
        calculations. Default is {}.

    Attributes
    ----------
    results : CalcResults
        Dictionary of calculated results, with keys from `properties`.

    Methods
    -------
    run()
        Run single point calculations.
    """

    def __init__(
        self,
        *,
        struct: MaybeSequence[Atoms] | None = None,
        struct_path: PathLike | None = None,
        arch: Architectures = "mace_mp",
        device: Devices = "cpu",
        model_path: PathLike | None = None,
        read_kwargs: ASEReadArgs | None = None,
        calc_kwargs: dict[str, Any] | None = None,
        set_calc: bool | None = None,
        attach_logger: bool = False,
        log_kwargs: dict[str, Any] | None = None,
        track_carbon: bool = True,
        tracker_kwargs: dict[str, Any] | None = None,
        properties: MaybeSequence[Properties] = (),
        write_results: bool = False,
        write_kwargs: OutputKwargs | None = None,
    ) -> None:
        """
        Read the structure being simulated and attach an MLIP calculator.

        Parameters
        ----------
        struct : MaybeSequence[Atoms] | None
            ASE Atoms structure(s) to simulate. Required if `struct_path`
            is None. Default is None.
        struct_path : PathLike | None
            Path of structure to simulate. Required if `struct` is None.
            Default is None.
        arch : Architectures
            MLIP architecture to use for single point calculations.
            Default is "mace_mp".
        device : Devices
            Device to run MLIP model on. Default is "cpu".
        model_path : PathLike | None
            Path to MLIP model. Default is `None`.
        read_kwargs : ASEReadArgs | None
            Keyword arguments to pass to ase.io.read. By default,
            read_kwargs["index"] is ":".
        calc_kwargs : dict[str, Any] | None
            Keyword arguments to pass to the selected calculator. Default is {}.
        set_calc : bool | None
            Whether to set (new) calculators for structures. Default is None.
        attach_logger : bool
            Whether to attach a logger. Default is False.
        log_kwargs : dict[str, Any] | None
            Keyword arguments to pass to `config_logger`. Default is {}.
        track_carbon : bool
            Whether to track carbon emissions of calculation. Default is True.
        tracker_kwargs : dict[str, Any] | None
            Keyword arguments to pass to `config_tracker`. Default is {}.
        properties : MaybeSequence[Properties]
            Physical properties to calculate. If not specified, "energy",
            "forces", and "stress" will be returned.
        write_results : bool
            True to write out structure with results of calculations. Default is False.
        write_kwargs : OutputKwargs | None
            Keyword arguments to pass to ase.io.write if saving structure with results
            of calculations. Default is {}.
        """
        read_kwargs, write_kwargs = none_to_dict(read_kwargs, write_kwargs)

        self.write_results = write_results
        self.write_kwargs = write_kwargs
        self.log_kwargs = log_kwargs

        # Read full trajectory by default
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
            track_carbon=track_carbon,
            tracker_kwargs=tracker_kwargs,
        )

        # Properties validated using calculator
        self.properties = properties

        # Set output file
        self.write_kwargs.setdefault("filename", None)
        self.write_kwargs["filename"] = self._build_filename(
            "results.extxyz", filename=self.write_kwargs["filename"]
        ).absolute()

        self.results = {}

    @property
    def properties(self) -> Sequence[Properties]:
        """
        Physical properties to be calculated.

        Returns
        -------
        Sequence[Properties]
            Physical properties.
        """
        return self._properties

    @properties.setter
    def properties(self, value: MaybeSequence[Properties]) -> None:
        """
        Setter for `properties`.

        Parameters
        ----------
        value : MaybeSequence[Properties]
            Physical properties to be calculated.
        """
        if isinstance(value, str):
            value = (value,)

        if isinstance(value, Sequence):
            for prop in value:
                if prop not in get_args(Properties):
                    raise NotImplementedError(
                        f"Property '{prop}' cannot currently be calculated."
                    )

        # If none specified, get energy, forces and stress
        if not value:
            value = ("energy", "forces", "stress")

        # Validate properties
        if "hessian" in value:
            if isinstance(self.struct, Sequence):
                for image in self.struct:
                    check_calculator(image.calc, "get_hessian")
            else:
                check_calculator(self.struct.calc, "get_hessian")

        self._properties = value

    def _get_potential_energy(self) -> MaybeList[float]:
        """
        Calculate potential energy using MLIP.

        Returns
        -------
        MaybeList[float]
            Potential energy of structure(s).
        """
        if isinstance(self.struct, Sequence):
            return [struct.get_potential_energy() for struct in self.struct]

        return self.struct.get_potential_energy()

    def _get_forces(self) -> MaybeList[ndarray]:
        """
        Calculate forces using MLIP.

        Returns
        -------
        MaybeList[ndarray]
            Forces of structure(s).
        """
        if isinstance(self.struct, Sequence):
            return [struct.get_forces() for struct in self.struct]

        return self.struct.get_forces()

    def _get_stress(self) -> MaybeList[ndarray]:
        """
        Calculate stress using MLIP.

        Returns
        -------
        MaybeList[ndarray]
            Stress of structure(s).
        """
        if isinstance(self.struct, Sequence):
            return [struct.get_stress() for struct in self.struct]

        return self.struct.get_stress()

    def _calc_hessian(self, struct: Atoms) -> ndarray:
        """
        Calculate analytical Hessian for a given structure.

        Parameters
        ----------
        struct : Atoms
            Structure to calculate Hessian for.

        Returns
        -------
        ndarray
            Analytical Hessian.
        """
        if "arch" in struct.calc.parameters:
            arch = struct.calc.parameters["arch"]
            label = f"{arch}_"
        else:
            label = ""

        # Calculate hessian
        hessian = struct.calc.get_hessian(struct)
        struct.info[f"{label}hessian"] = hessian
        return hessian

    def _get_hessian(self) -> MaybeList[ndarray]:
        """
        Calculate hessian using MLIP.

        Returns
        -------
        MaybeList[ndarray]
            Hessian of structure(s).
        """
        if isinstance(self.struct, Sequence):
            return [self._calc_hessian(struct) for struct in self.struct]

        return self._calc_hessian(self.struct)

    def run(self) -> CalcResults:
        """
        Run single point calculations.

        Returns
        -------
        CalcResults
            Dictionary of calculated results, with keys from `properties`.
        """
        self.results = {}

        if self.logger:
            self.logger.info("Starting single point calculation")
        if self.tracker:
            self.tracker.start_task("Single point")

        if "energy" in self.properties:
            self.results["energy"] = self._get_potential_energy()
        if "forces" in self.properties:
            self.results["forces"] = self._get_forces()
        if "stress" in self.properties:
            self.results["stress"] = self._get_stress()
        if "hessian" in self.properties:
            self.results["hessian"] = self._get_hessian()

        if self.logger:
            self.logger.info("Single point calculation complete")
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
            properties=self.properties,
            write_kwargs=self.write_kwargs,
        )

        return self.results
