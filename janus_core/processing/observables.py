"""Module for built-in correlation observables."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ase import Atoms, units

if TYPE_CHECKING:
    from janus_core.helpers.janus_types import SliceLike
    from janus_core.helpers.utils import slicelike_len_for


# pylint: disable=too-few-public-methods
class Observable(ABC):
    """
    Observable data that may be correlated.

    Parameters
    ----------
    dimension : int
        The dimension of the observed data.
    """

    def __init__(self, dimension: int = 1):
        """
        Initialise an observable with a given dimensionality.

        Parameters
        ----------
        dimension : int
            The dimension of the observed data.
        """
        self._dimension = dimension

    @abstractmethod
    def __call__(self, atoms: Atoms) -> list[float]:
        """
        Signature for returning observed value from atoms.

        Parameters
        ----------
        atoms : Atoms
            Atoms object to extract values from.

        Returns
        -------
        list[float]
            The observed value, with dimensions atoms by self.dimension.
        """

    def value_count(self, n_atoms: int | None = None) -> int:
        """
        Count of values returned by __call__.

        Parameters
        ----------
        n_atoms : int | None
            Atom count to expand atoms_slice.

        Returns
        -------
        int
            The number of values returned by __call__.
        """
        return self.dimension

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
        dict[str, int]
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
        for component in self.allowed_components.keys() - components.keys():
            if component not in self.allowed_components:
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
    atoms_slice : list[int] | SliceLike | None = None
        List or slice of atoms to observe velocities from.
    include_ideal_gas : bool
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
        components : list[str]
            Symbols for tensor components, xx, yy, etc.
        atoms_slice : list[int] | SliceLike | None = None
            List or slice of atoms to observe velocities from.
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

        if atoms_slice:
            self.atoms_slice = atoms_slice
        else:
            self.atoms_slice = slice(0, None, 1)

        Observable.__init__(self, len(components))
        self.include_ideal_gas = include_ideal_gas

    def __call__(self, atoms: Atoms) -> list[float]:
        """
        Get the stress components.

        Parameters
        ----------
        atoms : Atoms
            Atoms object to extract values from.

        Returns
        -------
        list[float]
            The stress components in GPa units.
        """
        sliced_atoms = atoms[self.atoms_slice]
        sliced_atoms.calc = atoms.calc
        return (
            sliced_atoms.get_stress(
                include_ideal_gas=self.include_ideal_gas, voigt=True
            )
            / units.GPa
        )[self._indices]


StressDiagonal = Stress(components=["xx", "yy", "zz"])
ShearStress = Stress(components=["xy", "yz", "zx"])


# pylint: disable=too-few-public-methods
class Velocity(Observable, ComponentMixin):
    """
    Observable for per atom velocity components.

    Parameters
    ----------
    components : list[str]
        Symbols for velocity components, x, y, z.
    atoms_slice : list[int] | SliceLike | None = None
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
        components : list[str]
            Symbols for tensor components, x, y, and z.
        atoms_slice : Union[list[int], SliceLike]
            List or slice of atoms to observe velocities from.
        """
        ComponentMixin.__init__(self, components={"x": 0, "y": 1, "z": 2})
        self._set_components(components)

        Observable.__init__(self, len(components))

        self.atoms_slice = atoms_slice if atoms_slice else slice(0, None, 1)

    def __call__(self, atoms: Atoms) -> list[float]:
        """
        Get the velocity components for correlated atoms.

        Parameters
        ----------
        atoms : Atoms
            Atoms object to extract values from.

        Returns
        -------
        list[float]
            The velocity values.
        """
        return atoms.get_velocities()[self.atoms_slice, :][:, self._indices].flatten()

    def value_count(self, n_atoms: int | None = None) -> int:
        """
        Count of values returned by __call__.

        Parameters
        ----------
        n_atoms : int | None
            Atom count to expand atoms_slice.

        Returns
        -------
        int
            The number of values returned by __call__.
        """
        if isinstance(self.atoms_slice, list):
            return len(self.atoms_slice) * self.dimension
        return slicelike_len_for(self.atoms_slice, self.n_atoms) * self.dimension
