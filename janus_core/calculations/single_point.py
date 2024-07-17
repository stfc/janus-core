"""Prepare and perform single point calculations."""

from collections.abc import Collection
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from ase import Atoms
from ase.io import read, write
from numpy import isfinite, ndarray

from janus_core.helpers.janus_types import (
    Architectures,
    ASEReadArgs,
    ASEWriteArgs,
    CalcResults,
    Devices,
    MaybeList,
    MaybeSequence,
    PathLike,
)
from janus_core.helpers.log import config_logger, config_tracker
from janus_core.helpers.mlip_calculators import choose_calculator
from janus_core.helpers.utils import FileNameMixin, none_to_dict


class SinglePoint(FileNameMixin):  # pylint: disable=too-many-instance-attributes
    """
    Prepare and perform single point calculations.

    Parameters
    ----------
    struct : Optional[MaybeSequence[Atoms]]
        ASE Atoms structure(s) to simulate. Required if `struct_path` is None.
        Default is None.
    struct_path : Optional[str]
        Path of structure to simulate. Required if `struct` is None.
        Default is None.
    struct_name : Optional[str]
        Name of structure. Default is inferred from chemical formula if `struct`
        is specified, else inferred from `struct_path`.
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
    log_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to `config_logger`. Default is {}.
    tracker_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to `config_tracker`. Default is {}.

    Attributes
    ----------
    architecture : Architectures
        MLIP architecture to use for single point calculations.
    struct : MaybeSequence[Atoms]
        ASE Atoms structure(s) to simulate.
    device : Devices
        Device to run MLIP model on.
    model_path : Optional[PathLike]
        Path to MLIP model.
    struct_path : Optional[str]
        Path of structure to simulate.
    struct_name : Optional[str]
        Name of structure.
    logger : Optional[logging.Logger]
        Logger if log file has been specified.
    tracker : Optional[OfflineEmissionsTracker]
        Tracker if logging is enabled.

    Methods
    -------
    read_structure(**kwargs)
        Read structure and structure name.
    set_calculator(**kwargs)
        Configure calculator and attach to structure.
    run(properties=None)
        Run single point calculations.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        struct: Optional[MaybeSequence[Atoms]] = None,
        struct_path: Optional[str] = None,
        struct_name: Optional[str] = None,
        architecture: Architectures = "mace_mp",
        device: Devices = "cpu",
        model_path: Optional[PathLike] = None,
        read_kwargs: Optional[ASEReadArgs] = None,
        calc_kwargs: Optional[dict[str, Any]] = None,
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
        struct_path : Optional[str]
            Path of structure to simulate. Required if `struct` is None.
            Default is None.
        struct_name : Optional[str]
            Name of structure. Default is inferred from chemical formula if `struct`
            is specified, else inferred from `struct_path`.
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
        log_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to `config_logger`. Default is {}.
        tracker_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to `config_tracker`. Default is {}.
        """
        if struct and struct_path:
            raise ValueError(
                "You cannot specify both the ASE Atoms structure (`struct`) "
                "and a path to the structure file (`struct_path`)"
            )

        if not struct and not struct_path:
            raise ValueError(
                "Please specify either the ASE Atoms structure (`struct`) "
                "or a path to the structure file (`struct_path`)"
            )

        [read_kwargs, calc_kwargs, log_kwargs, tracker_kwargs] = none_to_dict(
            [read_kwargs, calc_kwargs, log_kwargs, tracker_kwargs]
        )

        if log_kwargs and "filename" not in log_kwargs:
            raise ValueError("'filename' must be included in `log_kwargs`")

        log_kwargs.setdefault("name", __name__)
        self.logger = config_logger(**log_kwargs)
        self.tracker = config_tracker(self.logger, **tracker_kwargs)

        self.architecture = architecture
        self.device = device
        self.model_path = model_path

        self.struct_path = struct_path
        self.struct_name = struct_name

        # Read full trajectory by default
        read_kwargs.setdefault("index", ":")

        # Read structure if given as path
        if self.struct_path:
            self.read_structure(**read_kwargs)
        else:
            self.struct = struct

        FileNameMixin.__init__(self, self.struct, self.struct_name, None)

        # Configure calculator
        self.set_calculator(**calc_kwargs)

        if self.logger:
            self.logger.info("Single point calculator configured")

    def read_structure(self, **kwargs) -> None:
        """
        Read structure and structure name.

        If the file contains multiple structures, only the last configuration
        will be read by default.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to ase.io.read.
        """
        if not self.struct_path:
            raise ValueError("`struct_path` must be defined")

        self.struct = read(self.struct_path, **kwargs)
        if not self.struct_name:
            self.struct_name = Path(self.struct_path).stem

    def set_calculator(
        self, read_kwargs: Optional[ASEReadArgs] = None, **kwargs
    ) -> None:
        """
        Configure calculator and attach to structure.

        Parameters
        ----------
        read_kwargs : Optional[ASEReadArgs]
            Keyword arguments to pass to ase.io.read. Default is {}.
        **kwargs
            Additional keyword arguments passed to the selected calculator.
        """
        calculator = choose_calculator(
            architecture=self.architecture,
            device=self.device,
            model_path=self.model_path,
            **kwargs,
        )
        if self.struct is None:
            read_kwargs = read_kwargs if read_kwargs else {}
            self.read_structure(**read_kwargs)

        if isinstance(self.struct, list):
            for struct in self.struct:
                struct.calc = deepcopy(calculator)
            # Return single Atoms object if only one image in list
            if len(self.struct) == 1:
                self.struct = self.struct[0]
        else:
            self.struct.calc = calculator

    def _get_potential_energy(self) -> MaybeList[float]:
        """
        Calculate potential energy using MLIP.

        Returns
        -------
        MaybeList[float]
            Potential energy of structure(s).
        """
        tag = f"{self.architecture}_energy"
        if isinstance(self.struct, list):
            energies = [struct.get_potential_energy() for struct in self.struct]
            for struct, energy in zip(self.struct, energies):
                struct.info[tag] = energy
            return energies

        energy = self.struct.get_potential_energy()
        self.struct.info[tag] = energy
        return energy

    def _get_forces(self) -> MaybeList[ndarray]:
        """
        Calculate forces using MLIP.

        Returns
        -------
        MaybeList[ndarray]
            Forces of structure(s).
        """
        tag = f"{self.architecture}_forces"
        if isinstance(self.struct, list):
            forces = [struct.get_forces() for struct in self.struct]
            for struct, force in zip(self.struct, forces):
                struct.arrays[tag] = force
            return forces

        force = self.struct.get_forces()
        self.struct.arrays[tag] = force
        return force

    def _get_stress(self) -> MaybeList[ndarray]:
        """
        Calculate stress using MLIP.

        Returns
        -------
        MaybeList[ndarray]
            Stress of structure(s).
        """
        tag = f"{self.architecture}_stress"
        if isinstance(self.struct, list):
            stresses = [struct.get_stress() for struct in self.struct]
            for struct, stress in zip(self.struct, stresses):
                struct.info[tag] = stress
            return stresses

        stress = self.struct.get_stress()
        self.struct.info[tag] = stress
        return stress

    def _remove_invalid_props(
        self,
        struct: Atoms,
        results: CalcResults = None,
        properties: Collection[str] = (),
    ) -> None:
        """
        Remove any invalid properties from calculated results.

        Parameters
        ----------
        struct : Atoms
            ASE Atoms structure with attached calculator results.
        results : CalcResults
            Dictionary of calculated results. Default is {}.
        properties : Collection[str]
            Physical properties requested to be calculated. Default is ().
        """
        results = results if results else {}

        # Find any properties with non-finite values
        rm_keys = [
            prop
            for prop in struct.calc.results
            if not isfinite(struct.calc.results[prop]).all()
        ]
        # Raise error if property was explicitly requested, otherwise remove
        for prop in rm_keys:
            if prop in properties:
                raise ValueError(
                    f"'{prop}' contains non-finite values for this structure."
                )
            if prop in results:
                del struct.info[f"{self.architecture}_{prop}"]
                del struct.calc.results[prop]
                del results[prop]

    def _clean_results(
        self,
        results: CalcResults = None,
        properties: Collection[str] = (),
        invalidate_calc: bool = True,
    ) -> None:
        """
        Remove NaN and inf values from results and calc.results dictionaries.

        Parameters
        ----------
        results : CalcResults
            Dictionary of calculated results. Default is {}.
        properties : Collection[str]
            Physical properties requested to be calculated. Default is ().
        invalidate_calc : bool
            Remove calculator results if True. When True Atoms object loses
            its property methods and true values are in info and arrays.
            Default is True.
        """
        results = results if results else {}

        if isinstance(self.struct, list):
            for image in self.struct:
                self._remove_invalid_props(image, results, properties)
                if invalidate_calc:
                    image.calc.results = {}
        else:
            self._remove_invalid_props(self.struct, results, properties)
            if invalidate_calc:
                self.struct.calc.results = {}

    def run(
        self,
        properties: MaybeSequence[str] = (),
        write_results: bool = False,
        write_kwargs: Optional[ASEWriteArgs] = None,
    ) -> CalcResults:
        """
        Run single point calculations.

        Parameters
        ----------
        properties : MaybeSequence[str]
            Physical properties to calculate. If not specified, "energy",
            "forces", and "stress" will be returned.
        write_results : bool
            True to write out structure with results of calculations. Default is False.
        write_kwargs : Optional[ASEWriteArgs],
            Keyword arguments to pass to ase.io.write if saving structure with
            results of calculations. Default is {}.

        Returns
        -------
        CalcResults
            Dictionary of calculated results.
        """
        results: CalcResults = {}
        if isinstance(properties, str):
            properties = [properties]

        for prop in properties:
            if prop not in ["energy", "forces", "stress"]:
                raise NotImplementedError(
                    f"Property '{prop}' cannot currently be calculated."
                )

        write_kwargs = write_kwargs if write_kwargs else {}

        write_kwargs.setdefault(
            "filename",
            self._build_filename("results.extxyz").absolute(),
        )

        if self.logger:
            self.logger.info("Starting single point calculation")
            self.tracker.start_task("Single point")

        if "energy" in properties or len(properties) == 0:
            results["energy"] = self._get_potential_energy()
        if "forces" in properties or len(properties) == 0:
            results["forces"] = self._get_forces()
        if "stress" in properties or len(properties) == 0:
            results["stress"] = self._get_stress()

        # Remove meaningless values from results e.g. stress for non-periodic systems
        self._clean_results(results, properties=properties)

        if self.logger:
            self.tracker.stop_task()
            self.tracker.stop()
            self.logger.info("Single point calculation complete")

        if write_results:
            write(images=self.struct, **write_kwargs)

        return results
