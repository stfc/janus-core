"""Module for built-in correlation observables."""

from __future__ import annotations

from ase import Atoms, units


class Stress:
    """
    Observable for stress components.

    Parameters
    ----------
    component : str
        Symbol for tensor components, xx, yy, etc.
    include_ideal_gas : bool
        Calculate with the ideal gas contribution.
    """

    def __init__(self, component: str, *, include_ideal_gas: bool = True) -> None:
        """
        Initialise the observables from a symbolic str component.

        Parameters
        ----------
        component : str
            Symbol for tensor components, xx, yy, etc.
        include_ideal_gas : bool
            Calculate with the ideal gas contribution.
        """
        components = {
            "xx": 0,
            "yy": 1,
            "zz": 2,
            "yz": 3,
            "zy": 3,
            "xz": 4,
            "zx": 4,
            "xy": 5,
            "yx": 5,
        }
        if component not in components:
            raise ValueError(
                f"'{component}' invalid, must be '{', '.join(list(components.keys()))}'"
            )

        self.component = component
        self._index = components[self.component]
        self.include_ideal_gas = include_ideal_gas

    def __call__(self, atoms: Atoms, *args, **kwargs) -> float:
        """
        Get the stress component.

        Parameters
        ----------
        atoms : Atoms
            Atoms object to extract values from.
        *args : tuple
            Additional positional arguments passed to getter.
        **kwargs : dict
            Additional kwargs passed getter.

        Returns
        -------
        float
            The stress component in GPa units.
        """
        return (
            atoms.get_stress(include_ideal_gas=self.include_ideal_gas, voigt=True)[
                self._index
            ]
            / units.GPa
        )
