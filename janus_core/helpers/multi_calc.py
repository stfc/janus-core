"""Define MultiCalc ASE Calculator."""

from collections.abc import Sequence
from typing import Any

from ase.calculators.calculator import (
    BaseCalculator,
    Calculator,
    CalculatorSetupError,
    PropertyNotImplementedError,
)


class MultiCalc(BaseCalculator):
    """
    ASE MultiCalc class.

    Parameters
    ----------
    calcs : Sequence[Calculator]
        Calculators to use.
    """

    def __init__(self, calcs: Sequence[Calculator]):
        """
        Initialise class.

        Parameters
        ----------
        calcs : Sequence[Calculator]
            Calculators to use.
        """
        super().__init__()

        if len(calcs) == 0:
            raise CalculatorSetupError("Please provide a list of Calculators")

        common_properties = set.intersection(
            *(set(calc.implemented_properties) for calc in calcs)
        )

        self.implemented_properties = list(common_properties)
        if not self.implemented_properties:
            raise PropertyNotImplementedError(
                "The provided Calculators have" " no properties in common!"
            )

        self.calcs = calcs

    def __str__(self) -> str:
        """
        Return string representation of the calculator.

        Returns
        -------
        str
            String representation.
        """
        calcs = ", ".join(calc.__class__.__name__ for calc in self.calcs)
        return f"{self.__class__.__name__}({calcs})"

    def get_properties(self, properties, atoms) -> dict[str, Any]:
        """
        Get properties from each listed calculator.

        Parameters
        ----------
        properties : list[str]
            List of properties to be calculated.
        atoms : Atoms
            Atoms object to calculate properties for.

        Returns
        -------
        dict
            Dictionary of results.
        """
        results = {}

        def get_property(prop: str) -> None:
            """
            Get property from each listed calculator.

            Parameters
            ----------
            prop : str
                Property to get.
            """
            contribs = [calc.get_property(prop, atoms) for calc in self.calcs]
            results[prop] = contribs

        for prop in properties:  # get requested properties
            get_property(prop)
        for prop in self.implemented_properties:  # cache all available props
            if all(prop in calc.results for calc in self.calcs):
                get_property(prop)
        return results

    def calculate(self, atoms, properties, system_changes) -> None:
        """
        Calculate properties for each calculator and return values as list.

        Parameters
        ----------
        atoms : Atoms
            Atoms object to calculate properties for.
        properties : list[str]
            List of properties to be calculated.
        system_changes : list[str]
            List of what has changed since last calculation.
        """
        self.atoms = atoms.copy()  # for caching of results
        self.results = self.get_properties(properties, atoms)
