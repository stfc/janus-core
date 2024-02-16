"""Perpare and perform single point calculations."""

from __future__ import annotations

import pathlib

from ase.io import read
from numpy.typing import NDArray

from janus_core.mlip_calculators import choose_calculator


class SinglePoint:
    """Perpare and perform single point calculations."""

    def __init__(
        self,
        system: str,
        architecture: str = "mace_mp",
        device: str = "cpu",
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
        """
        self.architecture = architecture
        self.device = device
        self.system = system

        # Read system and get calculator
        self.read_system()
        self.get_calculator(**kwargs)

    def read_system(self) -> None:
        """Read system and system name."""
        self.sys = read(self.system)
        self.sysname = pathlib.Path(self.system).stem

    def get_calculator(self, **kwargs) -> None:
        """Configure calculator and attach to system."""
        calculator = choose_calculator(
            architecture=self.architecture,
            device=self.device,
            **kwargs,
        )
        if self.sys is None:
            self.read_system()
        self.sys.calc = calculator

    def _get_potential_energy(self) -> float:
        """Calculate potential energy using MLIP.

        Returns
        -------
        potential_energy : float
            Potential energy of system.
        """
        return self.sys.get_potential_energy()

    def _get_forces(self) -> NDArray:
        """Calculate forces using MLIP.

        Returns
        -------
        forces : float
            Forces of system.
        """
        return self.sys.get_forces()

    def _get_stress(self) -> NDArray:
        """Calculate stress using MLIP.

        Returns
        -------
        stress : float
            Stress of system.
        """
        return self.sys.get_stress()

    def run_single_point(self, properties: str | list[str] | None = None) -> dict:
        """Run single point calculations.

        Parameters
        ----------
        properties : str | List[str] | None
            Physical properties to calculate. If not specified, "energy",
            "forces", and "stress" will be returned.

        Returns
        -------
        results : dict
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
