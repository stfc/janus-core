"""Phonons."""

from typing import Optional

from ase import Atoms
from numpy import ndarray
import phonopy
from phonopy.structure.atoms import PhonopyAtoms

from janus_core.calculations.geom_opt import optimize
from janus_core.helpers.janus_types import MaybeList


class Phonons:  # pylint: disable=too-many-instance-attributes
    """
    Configure, calculate and output phonon calculations.

    Parameters
    ----------
    struct : Atoms
        Structrure to calculate phonons for.
    struct_name : Optional[str]
        Name of structure. Default is inferred from chemical formula if `struct`.
    supercell : MaybeList[int]
        Size of supercell for calculation. Default is 2.
    displacement : float
        Displacement for force constants calculation, in A. Default is 0.01.
    t_step : float
        Temperature step for CV calculations, in K. Default is 50.0.
    t_min : float
        Start temperature for CV calculations, in K. Default is 0.0.
    t_max : float
        End temperature for CV calculations, in K. Default is 1000.0.
    optimize_struct : bool
        Whether to perform geometry optimisation before calculating phonons.
        Default is False.

    Attributes
    ----------
    calc : ase.calculators.calculator.Calculator
        ASE Calculator attached to strucutre.
    results : dict
        Results of phonon calculations.
    """

    def __init__(
        self,
        struct: Atoms,
        struct_name: Optional[str] = None,
        supercell: MaybeList[int] = 2,
        displacement: float = 0.01,
        t_step: float = 50.0,
        t_min: float = 0.0,
        t_max: float = 1000.0,
        optimize_struct: bool = False,
    ) -> None:
        """
        Initialise Phonons class.

        Parameters
        ----------
        struct : Atoms
            Structrure to calculate phonons for.
        struct_name : Optional[str]
            Name of structure. Default is inferred from chemical formula if `struct`.
        supercell : MaybeList[int]
            Size of supercell for calculation. Default is 2.
        displacement : float
            Displacement for force constants calculation, in A. Default is 0.01.
        t_step : float
            Temperature step for CV calculations, in K. Default is 50.0.
        t_min : float
            Start temperature for CV calculations, in K. Default is 0.0.
        t_max : float
            End temperature for CV calculations, in K. Default is 1000.0.
        optimize_struct : bool
            Whether to perform geometry optimisation before calculating phonons.
            Default is False.
        """
        self.struct = struct
        if struct_name:
            self.struct_name = self.struct_name
        else:
            self.struct_name = self.struct.get_chemical_formula()

        # Ensure supercell is a list
        self.supercell = [supercell] * 3 if isinstance(supercell, int) else supercell

        self.displacement = displacement
        self.t_step = t_step
        self.t_min = t_min
        self.t_max = t_max
        self.optimize_struct = optimize_struct

        if not self.struct.calc:
            raise ValueError("Please attach a calculator to `struct`.")
        self.calc = self.struct.calc
        self.results = {}

    def calculate_phonons(self, write_results: bool = False) -> None:
        """
        Calculate phonons.

        Parameters
        ----------
        write_results : bool
            Whether to write out results to file. Default is False.
        """
        if self.optimize_struct:
            optimize(self.struct)

        cell = self.ASE_to_PhonopyAtoms(self.struct)

        supercell_matrix = (
            (self.supercell[0], 0, 0),
            (0, self.supercell[1], 0),
            (0, 0, self.supercell[2]),
        )
        phonon = phonopy.Phonopy(cell, supercell_matrix)
        phonon.generate_displacements(distance=self.displacement)
        disp_supercells = phonon.supercells_with_displacements

        phonon.forces = [
            self.calc_forces(supercell)
            for supercell in disp_supercells
            if supercell is not None
        ]

        phonon.produce_force_constants()
        phonon.run_mesh()
        phonon.run_thermal_properties(
            t_step=self.t_step, t_max=self.t_max, t_min=self.t_min
        )

        self.results["phonon"] = phonon
        self.results["thermal_properties"] = phonon.get_thermal_properties_dict()

        if write_results:
            self.write_phonon_results()

    def calculate_dos(self, mesh: Optional[MaybeList[float]] = None) -> None:
        """
        Calculate density of states and projected density of states.

        Parameters
        ----------
        mesh : MaybeList[float]
            Mesh for sampling. Default is [10, 10, 10].
        """
        if not mesh:
            mesh = [10, 10, 10]

        # Calculate phonons is not already run
        if "phonon" not in self.results:
            self.calculate_phonons(write_results=False)

        self.results["phonon"].run_mesh(mesh)
        self.results["phonon"].run_total_dos()

        self.results["phonon"].run_mesh(
            mesh, with_eigenvectors=True, is_mesh_symmetry=False
        )
        self.results["phonon"].run_projected_dos()

    def write_phonon_results(self) -> None:
        """Write results of phonon calculations."""
        self.results["phonon"].save(
            f"{self.struct_name}-ase.yml", settings={"force_constants": True}
        )

        self.results["phonon"].auto_band_structure(
            write_yaml=True, filename=f"{self.struct_name}-auto-band.yml"
        )

        with open(f"{self.struct_name}-cv.dat", "w", encoding="utf8") as out:
            temps = self.results["thermal_properties"]["temperatures"]
            c_vs = self.results["thermal_properties"]["heat_capacity"]
            entropies = self.results["thermal_properties"]["entropy"]
            free_energies = self.results["thermal_properties"]["free_energy"]

            print("#Temperature [K] | Cv | H | S ", file=out)
            for temp, c_v, free_energy, entropy in zip(
                temps, c_vs, free_energies, entropies
            ):
                print(f"{temp} {c_v} {free_energy} {entropy}", file=out)

    # No magnetic moments considered
    def Phonopy_to_ASEAtoms(self, struct: PhonopyAtoms) -> Atoms:
        # pylint: disable=invalid-name
        """
        Convert Phonopy Atoms structure to ASE Atoms structure.

        Parameters
        ----------
        struct : PhonopyAtoms
            PhonopyAtoms structure to be converted.

        Returns
        -------
        Atoms
            Converted ASE Atoms structure.
        """
        return Atoms(
            symbols=struct.symbols,
            scaled_positions=struct.scaled_positions,
            cell=struct.cell,
            masses=struct.masses,
            pbc=True,
            calculator=self.calc,
        )

    def ASE_to_PhonopyAtoms(self, struct: Atoms) -> PhonopyAtoms:
        # pylint: disable=invalid-name
        """
        Convert ASE Atoms structure to Phonopy Atoms structure.

        Parameters
        ----------
        struct : Atoms
            ASE Atoms structure to be converted.

        Returns
        -------
        PhonopyAtoms
            Converted PhonopyAtoms structure.
        """
        return PhonopyAtoms(
            symbols=struct.get_chemical_symbols(),
            cell=struct.cell.array,
            scaled_positions=struct.get_scaled_positions(),
            masses=struct.get_masses(),
        )

    def calc_forces(self, struct: PhonopyAtoms) -> ndarray:
        """
        Calculate forces on PhonopyAtoms structure.

        Parameters
        ----------
        struct : PhonopyAtoms
            Structure to calculate forces on.

        Returns
        -------
        ndarray
            Forces on the structure.
        """
        atoms = self.Phonopy_to_ASEAtoms(struct)
        return atoms.get_forces()