"""Prepare and perform single point calculations."""

import pathlib
from typing import Any, Literal, Optional, Union

from ase.io import read
from numpy import ndarray

from janus_core.mlip_calculators import architectures, choose_calculator, devices


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
    device : Literal[devices]
        Device to run model on. Default is "cpu".
    read_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ase.io.read. Default is {}.
    calc_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to the selected calculator. Default is {}.

    Attributes
    ----------
    architecture : Literal[architectures]
        MLIP architecture to use for single point calculations.
    structure : str
        Path of structure to simulate.
    device : Literal[devices]
        Device to run MLIP model on.
    struct : Union[Atoms, list[Atoms]
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
        architecture: Literal[architectures] = "mace_mp",
        device: Literal[devices] = "cpu",
        read_kwargs: Optional[dict[str, Any]] = None,
        calc_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Read the structure being simulated and attach an MLIP calculator.

        Parameters
        ----------
        structure : str
            Path of structure to simulate.
        architecture : Literal[architectures]
            MLIP architecture to use for single point calculations.
            Default is "mace_mp".
        device : Literal[devices]
            Device to run MLIP model on. Default is "cpu".
        read_kwargs : Optional[dict[str, Any]]
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
        self.structname = pathlib.Path(self.structure).stem

    def set_calculator(
        self, read_kwargs: Optional[dict[str, Any]] = None, **kwargs
    ) -> None:
        """
        Configure calculator and attach to structure.

        Parameters
        ----------
        read_kwargs : Optional[dict[str, Any]]
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

    def _get_potential_energy(self) -> Union[float, list[float]]:
        """
        Calculate potential energy using MLIP.

        Returns
        -------
        Union[float, list[float]]
            Potential energy of structure(s).
        """
        if isinstance(self.struct, list):
            return [struct.get_potential_energy() for struct in self.struct]

        return self.struct.get_potential_energy()

    def _get_forces(self) -> Union[ndarray, list[ndarray]]:
        """
        Calculate forces using MLIP.

        Returns
        -------
        Union[ndarray, list[ndarray]]
            Forces of structure(s).
        """
        if isinstance(self.struct, list):
            return [struct.get_forces() for struct in self.struct]

        return self.struct.get_forces()

    def _get_stress(self) -> Union[ndarray, list[ndarray]]:
        """
        Calculate stress using MLIP.

        Returns
        -------
        Union[ndarray, list[ndarray]]
            Stress of structure(s).
        """
        if isinstance(self.struct, list):
            return [struct.get_stress() for struct in self.struct]

        return self.struct.get_stress()

    def run_single_point(
        self, properties: Optional[Union[str, list[str]]] = None
    ) -> dict[str, Any]:
        """
        Run single point calculations.

        Parameters
        ----------
        properties : Optional[Union[str, list[str]]]
            Physical properties to calculate. If not specified, "energy",
            "forces", and "stress" will be returned.

        Returns
        -------
        dict[str, Any]
            Dictionary of calculated results.
        """
        results = {}
        if properties is None:
            properties = []
        if isinstance(properties, str):
            properties = [properties]

        if "energy" in properties or len(properties) == 0:
            results["energy"] = self._get_potential_energy()
        if "forces" in properties or len(properties) == 0:
            results["forces"] = self._get_forces()
        if "stress" in properties or len(properties) == 0:
            results["stress"] = self._get_stress()

        return results
