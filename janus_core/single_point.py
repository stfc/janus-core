"""Prepare and perform single point calculations."""

from collections.abc import Collection
from pathlib import Path
from typing import Any, Optional

from ase import Atoms
from ase.io import read, write
from numpy import isfinite, ndarray

from janus_core.mlip_calculators import choose_calculator

from .janus_types import (
    Architectures,
    ASEReadArgs,
    ASEWriteArgs,
    CalcResults,
    Devices,
    MaybeList,
    MaybeSequence,
)


class SinglePoint:
    """
    Prepare and perform single point calculations.

    Parameters
    ----------
    structure : str
        Structure to simulate.
    architecture : Literal[architectures]
        MLIP architecture to use for single point calculations.
        Default is "mace_mp".
    device : Devices
        Device to run model on. Default is "cpu".
    read_kwargs : ASEReadArgs
        Keyword arguments to pass to ase.io.read. Default is {}.
    calc_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to the selected calculator. Default is {}.

    Attributes
    ----------
    architecture : Architectures
        MLIP architecture to use for single point calculations.
    structure : str
        Path of structure to simulate.
    device : Devices
        Device to run MLIP model on.
    struct : MaybeList[Atoms]
        ASE Atoms or list of Atoms structures to simulate.
    structname : str
        Name of structure from its filename.

    Methods
    -------
    read_structure(**kwargs)
        Read structure and structure name.
    set_calculator(**kwargs)
        Configure calculator and attach to structure.
    run_single_point(properties=None)
        Run single point calculations.
    """

    def __init__(
        self,
        structure: str,
        architecture: Architectures = "mace_mp",
        device: Devices = "cpu",
        read_kwargs: Optional[ASEReadArgs] = None,
        calc_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Read the structure being simulated and attach an MLIP calculator.

        Parameters
        ----------
        structure : str
            Path of structure to simulate.
        architecture : Architectures
            MLIP architecture to use for single point calculations.
            Default is "mace_mp".
        device : Devices
            Device to run MLIP model on. Default is "cpu".
        read_kwargs : Optional[ASEReadArgs]
            Keyword arguments to pass to ase.io.read. Default is {}.
        calc_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to the selected calculator. Default is {}.
        """
        self.architecture = architecture
        self.device = device
        self.structure = structure

        # Read structure and get calculator
        read_kwargs = read_kwargs if read_kwargs else {}
        calc_kwargs = calc_kwargs if calc_kwargs else {}
        self.read_structure(**read_kwargs)
        self.set_calculator(**calc_kwargs)

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
        self.struct = read(self.structure, **kwargs)
        self.structname = Path(self.structure).stem

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
            **kwargs,
        )
        if self.struct is None:
            read_kwargs = read_kwargs if read_kwargs else {}
            self.read_structure(**read_kwargs)

        if isinstance(self.struct, list):
            for struct in self.struct:
                struct.calc = calculator
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
        if isinstance(self.struct, list):
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
        if isinstance(self.struct, list):
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
        if isinstance(self.struct, list):
            return [struct.get_stress() for struct in self.struct]

        return self.struct.get_stress()

    @staticmethod
    def _remove_invalid_props(
        struct: Atoms,
        properties: Collection[str] = (),
    ) -> None:
        """
        Remove any invalid properties from calculator results.

        Parameters
        ----------
        struct : Atoms
            Structure with attached results from calculator.
        properties : Collection[str]
            Physical properties requested to be calculated. Default is ().
        """
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
            del struct.calc.results[prop]

    def _clean_results(self, properties: Collection[str] = ()) -> None:
        """
        Remove results with NaN or inf values from calc.results dictionary.

        Parameters
        ----------
        properties : Collection[str]
            Physical properties requested to be calculated. Default is ().
        """
        if isinstance(self.struct, list):
            for image in self.struct:
                self._remove_invalid_props(image, properties)
        else:
            self._remove_invalid_props(self.struct, properties)

    def run_single_point(
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
        if write_kwargs and "filename" not in write_kwargs:
            raise ValueError("'filename' must be included in write_kwargs")

        if "energy" in properties or len(properties) == 0:
            results["energy"] = self._get_potential_energy()
        if "forces" in properties or len(properties) == 0:
            results["forces"] = self._get_forces()
        if "stress" in properties or len(properties) == 0:
            results["stress"] = self._get_stress()

        self._clean_results(properties=properties)

        if write_results:
            if "filename" not in write_kwargs:
                filename = f"{self.structname}-results.xyz"
                write_kwargs["filename"] = Path(".").absolute() / filename
            write(images=self.struct, **write_kwargs)

        return results
