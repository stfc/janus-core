"""Perpare and perform single point calculations."""

from __future__ import annotations

import pathlib
from typing import Any

from ase.io import read
from numpy import ndarray

from janus_core.mlip_calculators import choose_calculator


class SinglePoint:
    """Perpare and perform single point calculations."""

    def __init__(
        self,
        system: str,
        architecture: str = "mace_mp",
        device: str = "cpu",
        read_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """
        Initialise class.

        Attributes
        ----------
        system : str
            System to simulate.
        architecture : str
            MLIP architecture to use for single point calculations.
            Default is "mace_mp".
        device : str
            Device to run model on. Default is "cpu".
        read_kwargs : dict[str, Any] | None
            kwargs to pass to ase.io.read. Default is None.
        """
        self.architecture = architecture
        self.device = device
        self.system = system

        # Read system and get calculator
        read_kwargs = read_kwargs if read_kwargs else {}
        self.read_system(**read_kwargs)
        self.set_calculator(**kwargs)

    def read_system(self, **kwargs) -> None:
        """Read system and system name.

        If the file contains multiple structures, only the last configuration
        will be read by default.
        """
        self.sys = read(self.system, **kwargs)
        self.sysname = pathlib.Path(self.system).stem

    def set_calculator(
        self, read_kwargs: dict[str, Any] | None = None, **kwargs
    ) -> None:
        """Configure calculator and attach to system.

        Parameters
        ----------
        read_kwargs : dict[str, Any] | None
            kwargs to pass to ase.io.read. Default is None.
        """
        calculator = choose_calculator(
            architecture=self.architecture,
            device=self.device,
            **kwargs,
        )
        if self.sys is None:
            read_kwargs = read_kwargs if read_kwargs else {}
            self.read_system(**read_kwargs)

        if isinstance(self.sys, list):
            for sys in self.sys:
                sys.calc = calculator
        else:
            self.sys.calc = calculator

    def _get_potential_energy(self) -> float | list[float]:
        """Calculate potential energy using MLIP.

        Returns
        -------
        potential_energy : float | list[float]
            Potential energy of system(s).
        """
        if isinstance(self.sys, list):
            energies = []
            for sys in self.sys:
                energies.append(sys.get_potential_energy())
            return energies

        return self.sys.get_potential_energy()

    def _get_forces(self) -> ndarray | list[ndarray]:
        """Calculate forces using MLIP.

        Returns
        -------
        forces : ndarray | list[ndarray]
            Forces of system(s).
        """
        if isinstance(self.sys, list):
            forces = []
            for sys in self.sys:
                forces.append(sys.get_forces())
            return forces

        return self.sys.get_forces()

    def _get_stress(self) -> ndarray | list[ndarray]:
        """Calculate stress using MLIP.

        Returns
        -------
        stress : ndarray | list[ndarray]
            Stress of system(s).
        """
        if isinstance(self.sys, list):
            stress = []
            for sys in self.sys:
                stress.append(sys.get_stress())
            return stress

        return self.sys.get_stress()

    def run_single_point(
        self, properties: str | list[str] | None = None
    ) -> dict[str, Any]:
        """Run single point calculations.

        Parameters
        ----------
        properties : str | List[str] | None
            Physical properties to calculate. If not specified, "energy",
            "forces", and "stress" will be returned.

        Returns
        -------
        results : dict[str, Any]
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
