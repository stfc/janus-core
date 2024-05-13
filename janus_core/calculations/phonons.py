"""Phonons."""

from typing import Optional

from ase import Atoms
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy import ndarray
import phonopy
from phonopy.phonon.dos import get_pdos_indices
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

    def calculate_phonons(
        self, write_results: bool = False, plot_results: bool = False
    ) -> None:
        """
        Calculate phonons.

        Parameters
        ----------
        write_results : bool
            Whether to write out results to file. Default is False.
        plot_results : bool
            Whether to plot results. Default is False.
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

        if plot_results:
            self.plot_phonon_results()

    def calculate_dos(
        self, mesh: Optional[MaybeList[float]] = None, plot_results=True
    ) -> None:
        """
        Calculate density of states and projected density of states.

        Parameters
        ----------
        mesh : MaybeList[float]
            Mesh for sampling. Default is [10, 10, 10].
        plot_results : bool
            Whether to plot results for DOS and PDOS. Default is True.
        """
        if not mesh:
            mesh = [10, 10, 10]

        # Calculate phonons is not already run
        if "phonon" not in self.results:
            self.calculate_phonons(write_results=False, plot_results=False)

        self.results["phonon"].run_mesh(mesh)
        self.results["phonon"].run_total_dos()

        self.results["phonon"].run_mesh(
            mesh, with_eigenvectors=True, is_mesh_symmetry=False
        )
        self.results["phonon"].run_projected_dos()

        if plot_results:
            self.plot_dos_results()

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

    def plot_phonon_results(
        self,
        ymin: float = 0.0,
        ymax: float = 10.0,
    ) -> None:
        """
        Plot results of phonon calculations.

        Parameters
        ----------
        ymin : float
            Minimum y-value for zoomed band structure. Default is 0.0.
        ymax : float
            Maximum y-value for zoomed band structure. Default is 10.0.
        """
        bands_plot = self.results["phonon"].plot_band_structure()
        bands_plot.savefig(f"{self.struct_name}-bs-auto-ase.pdf")

        ncols = self.results["phonon"].band_structure.path_connections.count(False)
        fig, ax = plt.subplots(ncols=ncols, nrows=1, layout="constrained")
        self.results["phonon"].band_structure.plot(ax)

        for i in range(ncols):
            ax[i].set_ylim(ymin, ymax)
            if i > 0:
                ax[i].get_yaxis().set_visible(False)
        fig.savefig(f"{self.struct_name}-bs-ase-auto-zoom.pdf")

    def plot_dos_results(
        self,
        ymin: float = 0.0,
        ymax: float = 10.0,
    ) -> None:
        """
        Plot results of DOS and PDOS calculations.

        Parameters
        ----------
        ymin : float
            Minimum y-value for zoomed band structure. Default is 0.0.
        ymax : float
            Maximum y-value for zoomed band structure. Default is 10.0.
        """
        dos_plot = self.results["phonon"].plot_total_dos()
        dos_plot.savefig(f"{self.struct_name}-dos.pdf")

        projected_dos_plot = self.results["phonon"].plot_projected_dos()
        projected_dos_plot.savefig(f"{self.struct_name}-pdos.pdf")

        bands_dos_plot = self.results["phonon"].plot_band_structure_and_dos()
        bands_dos_plot.savefig(f"{self.struct_name}-bs-dos.pdf")

        self.plot_band_structure_dos_and_pdos(f"{self.struct_name}-bs-pdos.pdf")

        self.plot_band_structure_dos_and_pdos(
            f"{self.struct_name}-bs-pdos-zoom.pdf", ymin, ymax
        )

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

    def plot_band_structure_dos_and_pdos(
        self,
        fig_name: Optional[str] = None,
        ymin: Optional[float] = None,
        ymax: Optional[float] = None,
    ) -> None:
        """
        Plot band structure DOS and PDOS.

        Parameters
        ----------
        fig_name : Optional[str]
            Name for saved figure. Default is None.
        ymin : Optional[float]
            Minimum y-value for zoomed plot. Default is None.
        ymax : Optional[float]
            Maximum y-value for zoomed plot. Default is None.
        """
        pdi = get_pdos_indices(self.results["phonon"].primitive_symmetry)
        legend = ["Total"] + [
            self.results["phonon"].primitive.symbols[x[0]] for x in pdi
        ]
        ncols = self.results["phonon"].band_structure.path_connections.count(False) + 1
        fig = plt.figure()
        axs = ImageGrid(
            fig,
            111,  # similar to subplot(111)
            nrows_ncols=(1, ncols),
            axes_pad=0.11,
            label_mode="L",
        )
        if ymin is None:
            self.results["phonon"].band_structure.plot(axs[:-1])
        else:
            ncols = self.results["phonon"].band_structure.path_connections.count(False)
            self.results["phonon"].band_structure.plot(axs[:-1])
            for i in range(ncols):
                axs[i].set_ylim(ymin, ymax)
            if i > 0:
                axs[i].get_yaxis().set_visible(False)

        self.results["phonon"].total_dos.plot(
            axs[-1], xlabel="", ylabel="", draw_grid=False, flip_xy=True
        )
        self.results["phonon"].pdos.plot(
            axs[-1],
            indices=pdi,
            xlabel="",
            ylabel="",
            legend=legend,
            draw_grid=False,
            flip_xy=True,
        )
        ylim = axs[-1].get_ylim()
        if ymin is None:
            xlim = axs[-1].get_xlim()
        else:
            xlim = [ymin, ymax]

        aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) * 3
        axs[-1].set_aspect(aspect)
        axs[-1].axhline(y=0, linestyle=":", linewidth=0.5, color="b")
        axs[-1].set_xlim((xlim[0], xlim[1]))
        plt.savefig(fig_name)
        plt.show()
