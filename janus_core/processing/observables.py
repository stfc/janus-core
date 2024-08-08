"""Module for built-in correlation observables."""

from __future__ import annotations
from typing import Optional

from ase import Atoms, units


# pylint: disable=too-few-public-methods
class Observable:
    """
    Observable data that may be correlated.

    Parameters
    ----------
    dimension : int
        The dimension of the observed data.
    include_ideal_gas : bool
            Calculate with the ideal gas contribution.
    """

    def __init__(self, component: str, *, include_ideal_gas: bool = True) -> None:
        """
        Initialise an observable with a given dimensionality.

        Parameters
        ----------
        dimension : int
            The dimension of the observed data.
        include_ideal_gas : bool
            Calculate with the ideal gas contribution.
        """
        self._dimension = dimension
        self._getter = getter
        self.atoms = None

    def __call__(self, atoms: Atoms, *args, **kwargs) -> list[float]:
        """
        Call the user supplied getter if it exits.

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
        list[float]
            The observed value, with dimensions atoms by self.dimension.

        Raises
        ------
        ValueError
            If user supplied getter is None.
        """
        if self._getter:
            value = self._getter(atoms, *args, **kwargs)
            if not isinstance(value, list):
                return [value]
            return value
        raise ValueError("No user getter supplied")

    @property
    def dimension(self):
        """
        Dimension of the observable. Commensurate with self.__call__.

        Returns
        -------
        int
            Observables dimension.
        """
        return self._dimension

    @property
    def atom_count(self):
        """
        Atom count to average over.

        Returns
        -------
        int
            Atom count averaged over.
        """
        if self.atoms:
            return len(self.atoms)
        return 0


class ComponentMixin:
    """
    Mixin to handle Observables with components.

    Parameters
    ----------
    components : dict[str, int]
        Symbolic components mapped to indices.
    """

    def __init__(self, components: dict[str, int]):
        """
        Initialise the mixin with components.

        Parameters
        ----------
        components : dict[str, int]
            Symbolic components mapped to indices.
        """
        self._components = components

    @property
    def allowed_components(self) -> dict[str, int]:
        """
        Allowed symbolic components with associated indices.

        Returns
        -------
        Dict[str, int]
            The allowed components and associated indices.
        """
        return self._components

    @property
    def _indices(self) -> list[int]:
        """
        Get indices associated with self._components.

        Returns
        -------
        list[int]
            The indices for each self._components.
        """
        return [self._components[c] for c in self.components]

    def _set_components(self, components: list[str]):
        """
        Check if components are valid, if so set them.

        Parameters
        ----------
        components : str
            The component symbols to check.

        Raises
        ------
        ValueError
            If any component is invalid.
        """
        for component in components:
            if component not in self.allowed_components:
                component_names = list(self._components.keys())
                raise ValueError(
                    f"'{component}' invalid, must be '{', '.join(component_names)}'"
                )
        self.components = components


# pylint: disable=too-few-public-methods
class Stress(Observable, ComponentMixin):
    """
    Observable for stress components.

    Parameters
    ----------
    components : list[str]
        Symbols for correlated tensor components, xx, yy, etc.
    include_ideal_gas : bool
        Calculate with the ideal gas contribution.
    """

    def __init__(self, components: list[str], *, include_ideal_gas: bool = True):
        """
        Initialise the observable from a symbolic str component.

        Parameters
        ----------
        components : list[str]
            Symbols for tensor components, xx, yy, etc.
        include_ideal_gas : bool
            Calculate with the ideal gas contribution.
        """
        ComponentMixin.__init__(
            self,
            components={
                "xx": 0,
                "yy": 1,
                "zz": 2,
                "yz": 3,
                "zy": 3,
                "xz": 4,
                "zx": 4,
                "xy": 5,
                "yx": 5,
            },
        )
        self._set_components(components)

        Observable.__init__(self, len(components))
        self.include_ideal_gas = include_ideal_gas

    def __call__(self, atoms: Atoms, *args, **kwargs) -> list[float]:
        """
        Get the stress components.

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
        list[float]
            The stress components in GPa units.
        """
        return (
            atoms.get_stress(include_ideal_gas=self.include_ideal_gas, voigt=True)
            / units.GPa
        )[self._indices]


StressDiagonal = Stress(["xx", "yy", "zz"])
ShearStress = Stress(["xy", "yz", "zx"])


# pylint: disable=too-few-public-methods
class Velocity(Observable, ComponentMixin):
    """
    Observable for per atom velocity components.

    Parameters
    ----------
    components : list[str]
        Symbols for velocity components, x, y, z.
    atoms : list[int]
        List of atoms to observe velocities from.
    """

    def __init__(self, components: list[str], atoms: list[int]):
        """
        Initialise the observable from a symbolic str component and atom index.

        Parameters
        ----------
        components : list[str]
            Symbols for tensor components, x, y, and z.
        atoms : list[int]
            List of atoms to observe velocities from.
        """
        ComponentMixin.__init__(self, components={"x": 0, "y": 1, "z": 2})
        self._set_components(components)

        Observable.__init__(self, len(components))
        self.atoms = atoms

    def __call__(self, atoms: Atoms, *args, **kwargs) -> list[float]:
        """
        Get the velocity components for correlated atoms.

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
        list[float]
            The velocity values.
        """
        return atoms.get_velocities()[self.atoms, :][:, self._indices].flatten()
