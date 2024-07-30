"""Prepare and perform single point calculations."""

from collections.abc import Sequence
from copy import copy
from pathlib import Path
from typing import Any, Optional, get_args

from ase import Atoms
from ase.io import read
from numpy import ndarray

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
from janus_core.helpers.log import config_logger, config_tracker
from janus_core.helpers.mlip_calculators import choose_calculator
from janus_core.helpers.utils import FileNameMixin, none_to_dict, output_structs


class SinglePoint(FileNameMixin):  # pylint: disable=too-many-instance-attributes
    """
    Prepare and perform single point calculations.

    Parameters
    ----------
    struct : Optional[MaybeSequence[Atoms]]
        ASE Atoms structure(s) to simulate. Required if `struct_path` is None.
        Default is None.
    struct_path : Optional[PathLike]
        Path of structure to simulate. Required if `struct` is None.
        Default is None.
    properties : MaybeSequence[Properties]
        Physical properties to calculate. If not specified, "energy",
        "forces", and "stress" will be returned.
    write_results : bool
        True to write out structure with results of calculations. Default is False.
    architecture : Literal[architectures]
        MLIP architecture to use for single point calculations.
        Default is "mace_mp".
    device : Devices
        Device to run model on. Default is "cpu".
    model_path : Optional[PathLike]
        Path to MLIP model. Default is `None`.
    read_kwargs : ASEReadArgs
        Keyword arguments to pass to ase.io.read. By default,
        read_kwargs["index"] is ":".
    calc_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to the selected calculator. Default is {}.
    write_kwargs : Optional[OutputKwargs]
        Keyword arguments to pass to ase.io.write if saving structure with results of
        calculations. Default is {}.
    log_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to `config_logger`. Default is {}.
    tracker_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to `config_tracker`. Default is {}.

    Attributes
    ----------
    results : CalcResults
        Dictionary of calculated results, with keys from `properties`.
    logger : Optional[logging.Logger]
        Logger if log file has been specified.
    tracker : Optional[OfflineEmissionsTracker]
        Tracker if logging is enabled.

    Methods
    -------
    read_structure()
        Read structure and structure name.
    set_calculator()
        Configure calculator and attach to structure.
    run()
        Run single point calculations.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        struct: Optional[MaybeSequence[Atoms]] = None,
        struct_path: Optional[PathLike] = None,
        properties: MaybeSequence[Properties] = (),
        write_results: bool = False,
        architecture: Architectures = "mace_mp",
        device: Devices = "cpu",
        model_path: Optional[PathLike] = None,
        read_kwargs: Optional[ASEReadArgs] = None,
        calc_kwargs: Optional[dict[str, Any]] = None,
        write_kwargs: Optional[OutputKwargs] = None,
        log_kwargs: Optional[dict[str, Any]] = None,
        tracker_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Read the structure being simulated and attach an MLIP calculator.

        Parameters
        ----------
        struct : Optional[MaybeSequence[Atoms]]
            ASE Atoms structure(s) to simulate. Required if `struct_path`
            is None. Default is None.
        struct_path : Optional[PathLike]
            Path of structure to simulate. Required if `struct` is None.
            Default is None.
        properties : MaybeSequence[Properties]
            Physical properties to calculate. If not specified, "energy",
            "forces", and "stress" will be returned.
        write_results : bool
            True to write out structure with results of calculations. Default is False.
        architecture : Architectures
            MLIP architecture to use for single point calculations.
            Default is "mace_mp".
        device : Devices
            Device to run MLIP model on. Default is "cpu".
        model_path : Optional[PathLike]
            Path to MLIP model. Default is `None`.
        read_kwargs : Optional[ASEReadArgs]
            Keyword arguments to pass to ase.io.read. By default,
            read_kwargs["index"] is ":".
        calc_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to the selected calculator. Default is {}.
        write_kwargs : Optional[OutputKwargs],
            Keyword arguments to pass to ase.io.write if saving structure with results
            of calculations. Default is {}.
        log_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to `config_logger`. Default is {}.
        tracker_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to `config_tracker`. Default is {}.
        """
        (read_kwargs, calc_kwargs, write_kwargs, log_kwargs, tracker_kwargs) = (
            none_to_dict(
                (read_kwargs, calc_kwargs, write_kwargs, log_kwargs, tracker_kwargs)
            )
        )

        self.struct = struct
        self.struct_path = struct_path
        self.properties = properties
        self.write_results = write_results
        self.architecture = architecture
        self.device = device
        self.model_path = model_path
        self.read_kwargs = read_kwargs
        self.calc_kwargs = calc_kwargs
        self.write_kwargs = write_kwargs

        # Validate parameters
        if not self.struct and not self.struct_path:
            raise ValueError(
                "Please specify either the ASE Atoms structure (`struct`) "
                "or a path to the structure file (`struct_path`)"
            )

        if self.struct and self.struct_path:
            raise ValueError(
                "You cannot specify both the ASE Atoms structure (`struct`) "
                "and a path to the structure file (`struct_path`)"
            )

        if log_kwargs and "filename" not in log_kwargs:
            raise ValueError("'filename' must be included in `log_kwargs`")

        if not self.model_path and "model_path" in self.calc_kwargs:
            raise ValueError("`model_path` must be passed explicitly")

        # Read full trajectory by default
        self.read_kwargs.setdefault("index", ":")

        # Read structure if given as path
        file_prefix = None
        if self.struct_path:
            self.read_structure()
            file_prefix = Path(self.struct_path).stem

        # Configure logging
        log_kwargs.setdefault("name", __name__)
        self.logger = config_logger(**log_kwargs)
        self.tracker = config_tracker(self.logger, **tracker_kwargs)

        # Configure calculator
        self.set_calculator()
        if self.logger:
            self.logger.info("Single point calculator configured")

        # Set output file
        FileNameMixin.__init__(self, self.struct, None, file_prefix)
        self.write_kwargs.setdefault(
            "filename",
            self._build_filename("results.extxyz").absolute(),
        )

        self.results = {}

    def read_structure(self) -> None:
        """
        Read structure and structure name.

        If the file contains multiple structures, only the last configuration
        will be read by default.
        """
        if not self.struct_path:
            raise ValueError("`struct_path` must be defined")

        self.struct = read(self.struct_path, **self.read_kwargs)

    def set_calculator(self) -> None:
        """Configure calculator and attach to structure."""
        calculator = choose_calculator(
            architecture=self.architecture,
            device=self.device,
            model_path=self.model_path,
            **self.calc_kwargs,
        )
        if self.struct is None:
            self.read_structure(**self.read_kwargs)

        if isinstance(self.struct, Sequence):
            for struct in self.struct:
                struct.calc = copy(calculator)
            # Return single Atoms object if only one image in list
            if len(self.struct) == 1:
                self.struct = self.struct[0]
        else:
            self.struct.calc = calculator

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
        self._properties = value

        if isinstance(self._properties, str):
            self._properties = (self._properties,)

        if isinstance(self._properties, Sequence):
            for prop in self._properties:
                if prop not in get_args(Properties):
                    raise NotImplementedError(
                        f"Property '{prop}' cannot currently be calculated."
                    )

        # If none specified, get all valid properties
        if not self._properties:
            self._properties = get_args(Properties)

    def _get_potential_energy(self) -> MaybeList[float]:
        """
        Calculate potential energy using MLIP.

        Returns
        -------
        MaybeList[float]
            Potential energy of structure(s).
        """
        if isinstance(self.struct, Sequence):
            energies = [struct.get_potential_energy() for struct in self.struct]
            return energies

        energy = self.struct.get_potential_energy()
        return energy

    def _get_forces(self) -> MaybeList[ndarray]:
        """
        Calculate forces using MLIP.

        Returns
        -------
        MaybeList[ndarray]
            Forces of structure(s).
        """
        if isinstance(self.struct, Sequence):
            forces = [struct.get_forces() for struct in self.struct]
            return forces

        force = self.struct.get_forces()
        return force

    def _get_stress(self) -> MaybeList[ndarray]:
        """
        Calculate stress using MLIP.

        Returns
        -------
        MaybeList[ndarray]
            Stress of structure(s).
        """
        if isinstance(self.struct, Sequence):
            stresses = [struct.get_stress() for struct in self.struct]
            return stresses

        stress = self.struct.get_stress()
        return stress

    def run(
        self,
        *,
        properties: Optional[MaybeSequence[Properties]] = None,
        write_results: Optional[bool] = None,
        write_kwargs: Optional[OutputKwargs] = None,
    ) -> CalcResults:
        """
        Run single point calculations.

        Parameters
        ----------
        properties : Optional[MaybeSequence[Properties]]
            Physical properties to calculate. Default is self.properties.
        write_results : bool
            True to write out structure with results of calculations. Default is
            self.write_results.
        write_kwargs : Optional[OutputKwargs],
            Keyword arguments to pass to ase.io.write if saving structure with
            results of calculations. Default is self.write_kwargs.

        Returns
        -------
        CalcResults
            Dictionary of calculated results, with keys from `properties`.
        """
        # Parameters can be overwritten, otherwise default to values from instantiation
        properties = self.properties
        write_results = write_results if write_results else self.write_results
        write_kwargs = write_kwargs if write_kwargs else self.write_kwargs

        self.results = {}

        if self.logger:
            self.logger.info("Starting single point calculation")
            self.tracker.start_task("Single point")

        if "energy" in properties:
            self.results["energy"] = self._get_potential_energy()
        if "forces" in properties:
            self.results["forces"] = self._get_forces()
        if "stress" in properties:
            self.results["stress"] = self._get_stress()

        if self.logger:
            self.tracker.stop_task()
            self.tracker.stop()
            self.logger.info("Single point calculation complete")

        output_structs(
            self.struct,
            write_results=write_results,
            properties=properties,
            write_kwargs=write_kwargs,
        )

        return self.results
