"""Phonon calculations."""

from typing import Any, Optional

from ase import Atoms
from numpy import ndarray
import phonopy
from phonopy.file_IO import write_force_constants_to_hdf5
from phonopy.structure.atoms import PhonopyAtoms

from janus_core.calculations.geom_opt import optimize
from janus_core.helpers.janus_types import MaybeList, PathLike
from janus_core.helpers.log import config_logger
from janus_core.helpers.utils import none_to_dict


class Phonons:  # pylint: disable=too-many-instance-attributes
    """
    Configure, perform phonon calculations and write out results.

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
        Temperature step for thermal properties calculations, in K. Default is 50.0.
    t_min : float
        Start temperature for thermal properties calculations, in K. Default is 0.0.
    t_max : float
        End temperature for thermal properties calculations, in K. Default is 1000.0.
    minimize : bool
        Whether to perform geometry optimisation before calculating phonons.
        Default is False.
    hdf5 : bool
        Whether to write force constants in hdf format or not.
        Default is True.
    plot : bool
        Whether to plot various graphs as band stuctures, dos/pdos in svg.
        Default is False.
    minimize_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to geometry optimizer. Default is {}.
    file_prefix : Optional[PathLike]
        Prefix for output filenames. Default is inferred from structure name, or
        chemical formula of the structure.
    log_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to `config_logger`. Default is {}.

    Attributes
    ----------
    calc : ase.calculators.calculator.Calculator
        ASE Calculator attached to strucutre.
    results : dict
        Results of phonon calculations.
    logger : logging.Logger
        Logger if log file has been specified.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        struct: Atoms,
        struct_name: Optional[str] = None,
        supercell: MaybeList[int] = 2,
        displacement: float = 0.01,
        t_step: float = 50.0,
        t_min: float = 0.0,
        t_max: float = 1000.0,
        minimize: bool = False,
        hdf5: bool = False,
        plot: bool = False,
        minimize_kwargs: Optional[dict[str, Any]] = None,
        file_prefix: Optional[PathLike] = None,
        log_kwargs: Optional[dict[str, Any]] = None,
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
        minimize : bool
            Whether to perform geometry optimisation before calculating phonons.
            Default is False.
        hdf5 : bool
            Whether to write force constants in hdf format or not.
            Default is True.
        plot : bool
            Whether to plot various graphs as band stuctures, dos/pdos in svg.
            Default is False.
        minimize_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to geometry optimizer. Default is {}.
        file_prefix : Optional[PathLike]
            Prefix for output filenames. Default is inferred from structure name, or
            chemical formula of the structure.
        log_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to `config_logger`. Default is {}.
        """
        [minimize_kwargs, log_kwargs] = none_to_dict([minimize_kwargs, log_kwargs])

        self.struct = struct
        if struct_name:
            self.struct_name = struct_name
        else:
            self.struct_name = self.struct.get_chemical_formula()

        self.file_prefix = file_prefix if file_prefix else self.struct_name

        # Ensure supercell is a valid list
        self.supercell = [supercell] * 3 if isinstance(supercell, int) else supercell
        if len(self.supercell) != 3:
            raise ValueError("`supercell` must be an integer, or list of length 3")

        self.displacement = displacement
        self.t_step = t_step
        self.t_min = t_min
        self.t_max = t_max
        self.minimize = minimize
        self.minimize_kwargs = minimize_kwargs

        self.log_kwargs = log_kwargs
        self.log_kwargs.setdefault("name", __name__)
        self.logger = config_logger(**self.log_kwargs)

        self.hdf5 = hdf5
        self.plot = plot

        if not self.struct.calc:
            raise ValueError("Please attach a calculator to `struct`.")
        self.calc = self.struct.calc
        self.results = {}

    def _set_filename(
        self, default_suffix: str, filename: Optional[PathLike] = None
    ) -> str:
        """
        Set filename using the file prefix and suffix if not specified otherwise.

        Parameters
        ----------
        default_suffix : str
            Default suffix to use if `filename` is not specified.
        filename : Optional[PathLike]
            Filename to use, if specified. Default is None.

        Returns
        -------
        str
            Filename specified, or default filename.
        """
        if filename:
            return filename
        return f"{self.file_prefix}-{default_suffix}"

    def calc_force_constants(self, write_results: bool = True) -> None:
        """
        Calculate force constants and optionally write results.

        Parameters
        ----------
        write_results : bool
            Whether to write out results to file. Default is True.
        """
        if self.minimize:
            if self.logger:
                self.minimize_kwargs["log_kwargs"] = {
                    "filename": self.log_kwargs["filename"],
                    "name": self.logger.name,
                    "filemode": "a",
                }
            optimize(self.struct, **self.minimize_kwargs)

        if self.logger:
            self.logger.info("Beginning phonons calculation")

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
        self.results["phonon"] = phonon

        if self.logger:
            self.logger.info("Phonons calculation complete")

        if write_results:
            self.write_force_constants(force_consts_to_hdf5=self.hdf5)

    def calc_bands(self, write_results: bool = True) -> None:
        """
        Calculate band structure and optionally write and plot results.

        Parameters
        ----------
        write_results : bool
            Whether to write out results to file. Default is True.
        """
        # Calculate phonons is not already run
        if "phonon" not in self.results:
            self.calc_force_constants(write_results=False)
        self.write_band_structure(write_bands=write_results)

    def write_band_structure(
        self,
        *,
        write_bands: bool = None,
        bands_file: Optional[PathLike] = None,
        plot_file: Optional[PathLike] = None,
    ) -> None:
        """
        Write results of band structure calculations.

        Parameters
        ----------
        write_bands : bool
            Whether to write out results to file. Default is True.
        bands_file : Optional[PathLike]
            Name of yaml file to save band structure. Default is inferred from
            `file_prefix`.
        plot_file : Optional[PathLike]
            Name of svg file to save band structure. Default is inferred from
            `file_prefix`.
        """

        bands_file = self._set_filename("auto_bands.yml", bands_file)
        self.results["phonon"].auto_band_structure(
            write_yaml=write_bands, filename=bands_file
        )
        if self.plot:
            bplt = self.results["phonon"].plot_band_structure()
            plot_file = self._set_filename("auto_bands.svg", plot_file)
            bplt.savefig(plot_file)

    def write_force_constants(
        self,
        *,
        phonopy_file: Optional[PathLike] = None,
        force_consts_to_hdf5: bool = False,
        force_consts_file: Optional[PathLike] = None,
    ) -> None:
        """
        Write results of force constants calculations.

        Parameters
        ----------
        phonopy_file : Optional[PathLike]
            Name of yaml file to save params of phonopy and optionally force constants.
            Default is inferred from `file_prefix`.
        force_consts_to_hdf5 : bool
            Whether to save the force constants separately to an hdf5 file. Default is
            False.
        force_consts_file : Optional[PathLike]
            Name of hdf5 file to save force constants. Unused if `force_consts_to_hdf5`
            is False. Default is inferred from `file_prefix`.
        """
        phonopy_file = self._set_filename("phonopy.yml", phonopy_file)
        force_consts_file = self._set_filename(
            "force_constants.hdf5", force_consts_file
        )

        phonon = self.results["phonon"]

        save_force_consts = not force_consts_to_hdf5
        phonon.save(phonopy_file, settings={"force_constants": save_force_consts})

        if force_consts_to_hdf5:
            write_force_constants_to_hdf5(
                phonon.force_constants, filename=force_consts_file
            )

    def calc_thermal_props(self, write_results: bool = True) -> None:
        """
        Calculate thermal properties and optionally write results.

        Parameters
        ----------
        write_results : bool
            Whether to write out results to file. Default is True.
        """
        # Calculate phonons is not already run
        if "phonon" not in self.results:
            self.calc_force_constants(write_results=False)

        if self.logger:
            self.logger.info("Beginning thermal properties calculation")

        self.results["phonon"].run_mesh()
        self.results["phonon"].run_thermal_properties(
            t_step=self.t_step, t_max=self.t_max, t_min=self.t_min
        )
        self.results["thermal_properties"] = self.results[
            "phonon"
        ].get_thermal_properties_dict()

        if self.logger:
            self.logger.info("Thermal properties calculation complete")

        if write_results:
            self.write_thermal_props()

    def write_thermal_props(self, filename: Optional[PathLike] = None) -> None:
        """
        Write results of thermal properties calculations.

        Parameters
        ----------
        filename : Optional[PathLike]
            Name of data file to save thermal properties. Default is inferred from
            `file_prefix`.
        """
        filename = self._set_filename("thermal.dat", filename)

        with open(filename, "w", encoding="utf8") as out:
            temps = self.results["thermal_properties"]["temperatures"]
            c_vs = self.results["thermal_properties"]["heat_capacity"]
            entropies = self.results["thermal_properties"]["entropy"]
            free_energies = self.results["thermal_properties"]["free_energy"]

            print("#Temperature [K] | Cv | H | S ", file=out)
            for properties in zip(temps, c_vs, free_energies, entropies):
                print(*properties, file=out)

    def calc_dos(
        self, mesh: MaybeList[float] = (10, 10, 10), write_results=True
    ) -> None:
        """
        Calculate density of states and optionally write results.

        Parameters
        ----------
        mesh : MaybeList[float]
            Mesh for sampling. Default is (10, 10, 10).
        write_results : bool
            Whether to write out results to file. Default is True.
        """
        # Calculate phonons is not already run
        if "phonon" not in self.results:
            self.calc_force_constants(write_results=False)

        if self.logger:
            self.logger.info("Beginning DOS calculation")

        self.results["phonon"].run_mesh(mesh)
        self.results["phonon"].run_total_dos()

        if self.logger:
            self.logger.info("DOS calculation complete")

        if write_results:
            self.write_dos()

    def write_dos(
        self,
        filename: Optional[PathLike] = None,
        plot_file: Optional[PathLike] = None,
        plot_bs_file: Optional[PathLike] = None,
    ) -> None:
        """
        Write results of DOS calculation.

        Parameters
        ----------
        filename : Optional[PathLike]
            Name of data file to save the calculated DOS. Default is inferred from
            `file_prefix`.
        plot_file : Optional[PathLike]
            Name of svg file to plot the DOS. Default is inferred from
            `file_prefix`.
        plot_bs_file : Optional[PathLike]
            Name of svg file to plot the band structure and DOS.
            Default is inferred from `file_prefix`.
        """
        filename = self._set_filename("dos.dat", filename)
        self.results["phonon"].total_dos.write(filename)
        if self.plot:
            bplt = self.results["phonon"].plot_total_dos()
            plot_file = self._set_filename("dos.svg", plot_file)
            bplt.savefig(plot_file)

            bplt = self.results["phonon"].plot_band_structure_and_dos()
            plot_bs_file = self._set_filename("bs-dos.svg", plot_bs_file)
            bplt.savefig(plot_bs_file)

    def calc_pdos(
        self, mesh: MaybeList[float] = (10, 10, 10), write_results: bool = True
    ) -> None:
        """
        Calculate projected density of states and optionally write results.

        Parameters
        ----------
        mesh : MaybeList[float]
            Mesh for sampling. Default is (10, 10, 10).
        write_results : bool
            Whether to write out results to file. Default is True.
        """
        # Calculate phonons is not already run
        if "phonon" not in self.results:
            self.calc_force_constants(write_results=False)

        if self.logger:
            self.logger.info("Beginning PDOS calculation")

        self.results["phonon"].run_mesh(
            mesh, with_eigenvectors=True, is_mesh_symmetry=False
        )
        self.results["phonon"].run_projected_dos()

        if self.logger:
            self.logger.info("PDOS calculation complete")

        if write_results:
            self.write_pdos()

    def write_pdos(
        self, filename: Optional[PathLike] = None, plot_file: Optional[PathLike] = None
    ) -> None:
        """
        Write results of PDOS calculation.

        Parameters
        ----------
        filename : Optional[PathLike]
            Name of data file to save the calculated PDOS. Default is inferred from
            `file_prefix`.
        plot_file : Optional[PathLike]
            Name of svg file to plot the calculated PDOS. Default is inferred from
            `file_prefix`.
        """
        filename = self._set_filename("pdos.dat", filename)
        self.results["phonon"].projected_dos.write(filename)
        if self.plot:
            bplt = self.results["phonon"].plot_projected_dos()
            plot_file = self._set_filename("pdos.svg", plot_file)
            bplt.savefig(plot_file)

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
