"""Perpare and perform single point calculations."""

from __future__ import annotations

import pathlib

from ase.io import read

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
        System : str
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

    def read_system(self):
        """
        Read system and system name.
        """
        self.sys = read(self.system)
        self.sysname = pathlib.Path(self.system).stem

    def get_calculator(self, **kwargs):
        """
        Configure calculator and attach to system.
        """
        calculator = choose_calculator(
            architecture=self.architecture,
            device=self.device,
            **kwargs,
        )
        if self.sys is None:
            self.read_system()
        self.sys.calc = calculator

    def get_potential_energy(self):
        """
        Calculate potential energy using MLIP.
        """
        return self.sys.get_potential_energy()
