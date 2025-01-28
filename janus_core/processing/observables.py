"""Module for built-in correlation observables."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ase import Atoms
from ase.units import create_units

if TYPE_CHECKING:
    from janus_core.helpers.janus_types import SliceLike

from janus_core.helpers.utils import slicelike_to_startstopstep

units = create_units("2014")


# pylint: disable=too-few-public-methods
class Observable(ABC):
    """
    Observable data that may be correlated.

    Parameters
    ----------
    atoms_slice
        A slice of atoms to observe.
    """

    def __init__(self, atoms_slice: list[int] | SliceLike | None = None):
        """
        Initialise an observable with a given dimensionality.

        Parameters
        ----------
        atoms_slice
            A slice of atoms to observe. By default all atoms are included.
        """
        if not atoms_slice:
            self.atoms_slice = slice(0, None, 1)
            return

        if isinstance(atoms_slice, list):
            self.atoms_slice = atoms_slice
        else:
            self.atoms_slice = slice(*slicelike_to_startstopstep(atoms_slice))

    @abstractmethod
    def __call__(self, atoms: Atoms) -> list[float]:
        """
        Signature for returning observed value from atoms.

        Parameters
        ----------
        atoms
            Atoms object to extract values from.

        Returns
        -------
        list[float]
            The observed value, with dimensions atoms by self.dimension.
        """


class ComponentMixin:
    """
    Mixin to handle Observables with components.

    Parameters
    ----------
    components
        Symbolic components mapped to indices.
    """

    def __init__(self, components: dict[str, int]):
        """
        Initialise the mixin with components.

        Parameters
        ----------
        components
            Symbolic components mapped to indices.
        """
        self._allowed_components = components

    @property
    def _indices(self) -> list[int]:
        """
        Get indices associated with self.components.

        Returns
        -------
        list[int]
            The indices for each self.components.
        """
        return [self._allowed_components[c] for c in self.components]

    @property
    def components(self) -> list[str]:
        """
        Get the symbolic components of the observable.

        Returns
        -------
        list[str]
            The observables components.
        """
        return self._components

    @components.setter
    def components(self, components: list[str]):
        """
        Check if components are valid, if so set them.

        Parameters
        ----------
        components
            The component symbols to check.

        Raises
        ------
        ValueError
            If any component is invalid.
        """
        if any(components - self._allowed_components.keys()):
            raise ValueError(
                f"'{components - self._allowed_components.keys()}'"
                f" invalid, must be '{', '.join(self._allowed_components)}'"
            )

        self._components = components


# pylint: disable=too-few-public-methods
class Stress(Observable, ComponentMixin):
    """
    Observable for stress components.

    Parameters
    ----------
    components
        Symbols for correlated tensor components, xx, yy, etc.
    atoms_slice
        List or slice of atoms to observe velocities from.
    include_ideal_gas
        Calculate with the ideal gas contribution.
    """

    def __init__(
        self,
        *,
        components: list[str],
        atoms_slice: list[int] | SliceLike | None = None,
        include_ideal_gas: bool = True,
    ):
        """
        Initialise the observable from a symbolic str component.

        Parameters
        ----------
        components
            Symbols for tensor components, xx, yy, etc.
        atoms_slice
            List or slice of atoms to observe velocities from.
        include_ideal_gas
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
        self.components = components

        Observable.__init__(self, atoms_slice)
        self.include_ideal_gas = include_ideal_gas

    def __call__(self, atoms: Atoms) -> list[float]:
        """
        Get the stress components.

        Parameters
        ----------
        atoms
            Atoms object to extract values from.

        Returns
        -------
        list[float]
            The stress components in GPa units.

        Raises
        ------
        ValueError
            If atoms is not an Atoms object.
        """
        if not isinstance(atoms, Atoms):
            raise ValueError("atoms should be an Atoms object")
        sliced_atoms = atoms[self.atoms_slice]
        # must be re-attached after slicing for get_stress
        sliced_atoms.calc = atoms.calc
        stresses = (
            sliced_atoms.get_stress(
                include_ideal_gas=self.include_ideal_gas, voigt=True
            )
            / units.GPa
        )
        return stresses[self._indices]


StressHydrostatic = Stress(components=["xx", "yy", "zz"])
StressShear = Stress(components=["xy", "yz", "zx"])


# pylint: disable=too-few-public-methods
class Velocity(Observable, ComponentMixin):
    """
    Observable for per atom velocity components.

    Parameters
    ----------
    components
        Symbols for velocity components, x, y, z.
    atoms_slice
        List or slice of atoms to observe velocities from.
    """

    def __init__(
        self,
        *,
        components: list[str],
        atoms_slice: list[int] | SliceLike | None = None,
    ):
        """
        Initialise the observable from a symbolic str component and atom index.

        Parameters
        ----------
        components
            Symbols for tensor components, x, y, and z.
        atoms_slice
            List or slice of atoms to observe velocities from.
        """
        ComponentMixin.__init__(self, components={"x": 0, "y": 1, "z": 2})
        self.components = components

        Observable.__init__(self, atoms_slice)

    def __call__(self, atoms: Atoms) -> list[float]:
        """
        Get the velocity components for correlated atoms.

        Parameters
        ----------
        atoms
            Atoms object to extract values from.

        Returns
        -------
        list[float]
            The velocity values.
        """
        return atoms.get_velocities()[self.atoms_slice, :][:, self._indices]
