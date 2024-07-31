"""Equation of State."""

from copy import deepcopy
from typing import Any, Optional

from ase import Atoms
from ase.eos import EquationOfState
from ase.units import kJ
from numpy import cbrt, empty, linspace

from janus_core.calculations.geom_opt import GeomOpt
from janus_core.helpers.janus_types import EoSNames, EoSResults, OutputKwargs, PathLike
from janus_core.helpers.log import config_logger, config_tracker
from janus_core.helpers.utils import FileNameMixin, none_to_dict, output_structs


class EoS(FileNameMixin):
    # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """
    Prepare and calculate equation of state of a structure.

    Parameters
    ----------
    struct : Atoms
        Structure.
    struct_name : Optional[str]
        Name of structure. Default is None.
    min_volume : float
        Minimum volume scale factor. Default is 0.95.
    max_volume : float
        Maximum volume scale factor. Default is 1.05.
    n_volumes : int
        Number of volumes to use. Default is 7.
    eos_type : EoSNames
        Type of fit for equation of state. Default is "birchmurnaghan".
    minimize : bool
        Whether to minimize initial structure before calculations. Default is True.
    minimize_all : bool
        Whether to optimize geometry for all generated structures. Default is False.
    minimize_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to optimize. Default is None.
    write_results : bool
        True to write out results of equation of state calculations. Default is True.
    write_structures : bool
        True to write out all genereated structures. Default is False.
    write_kwargs : Optional[OutputKwargs],
        Keyword arguments to pass to ase.io.write to save generated structures.
        Default is {}.
    file_prefix : Optional[PathLike]
        Prefix for output filenames. Default is inferred from structure name, or
        chemical formula of the structure.
    log_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to `config_logger`. Default is {}.
    tracker_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to `config_tracker`. Default is {}.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        struct: Atoms,
        struct_name: Optional[str] = None,
        min_volume: float = 0.95,
        max_volume: float = 1.05,
        n_volumes: int = 7,
        eos_type: EoSNames = "birchmurnaghan",
        minimize: bool = True,
        minimize_all: bool = False,
        minimize_kwargs: Optional[dict[str, Any]] = None,
        write_results: bool = True,
        write_structures: bool = False,
        write_kwargs: Optional[OutputKwargs] = None,
        file_prefix: Optional[PathLike] = None,
        log_kwargs: Optional[dict[str, Any]] = None,
        tracker_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialise class.

        Parameters
        ----------
        struct : Atoms
            Structure.
        struct_name : Optional[str]
            Name of structure. Default is None.
        min_volume : float
            Minimum volume scale factor. Default is 0.95.
        max_volume : float
            Maximum volume scale factor. Default is 1.05.
        n_volumes : int
            Number of volumes to use. Default is 7.
        eos_type : EoSNames
            Type of fit for equation of state. Default is "birchmurnaghan".
        minimize : bool
            Whether to minimize initial structure before calculations. Default is True.
        minimize_all : bool
            Whether to optimize geometry for all generated structures. Default is
            False.
        minimize_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to optimize. Default is None.
        write_results : bool
            True to write out results of equation of state calculations. Default is
            True.
        write_structures : bool
            True to write out all genereated structures. Default is False.
        write_kwargs : Optional[OutputKwargs],
            Keyword arguments to pass to ase.io.write to save generated structures.
            Default is {}.
        file_prefix : Optional[PathLike]
            Prefix for output filenames. Default is inferred from structure name, or
            chemical formula of the structure.
        log_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to `config_logger`. Default is {}.
        tracker_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to `config_tracker`. Default is {}.

        Attributes
        ----------
        logger : Optional[logging.Logger]
            Logger if log file has been specified.
        tracker : Optional[OfflineEmissionsTracker]
            Tracker if logging is enabled.
        results : EoSResults
            Dictionary containing equation of state ASE object, and the fitted minimum
            bulk modulus, volume, and energy.
        volumes : list[float]
            List of volumes of generated structures.
        energies : list[float]
            List of energies of generated structures.
        lattice_scalars : NDArray[float64]
            Lattice scalars of generated structures.
        """
        self.struct = struct
        self.min_volume = min_volume
        self.max_volume = max_volume
        self.n_volumes = n_volumes
        self.eos_type = eos_type
        self.minimize = minimize
        self.minimize_all = minimize_all
        self.write_results = write_results
        self.write_structures = write_structures
        self.file_prefix = file_prefix

        [minimize_kwargs, write_kwargs, log_kwargs, tracker_kwargs] = none_to_dict(
            [minimize_kwargs, write_kwargs, log_kwargs, tracker_kwargs]
        )
        self.minimize_kwargs = minimize_kwargs
        self.write_kwargs = write_kwargs
        self.log_kwargs = log_kwargs

        log_kwargs.setdefault("name", __name__)
        self.logger = config_logger(**log_kwargs)
        self.tracker = config_tracker(self.logger, **tracker_kwargs)

        FileNameMixin.__init__(self, struct, struct_name, file_prefix)

        self.write_kwargs.setdefault(
            "filename",
            self._build_filename("generated.extxyz").absolute(),
        )

        if (
            (self.minimize or self.minimize_all)
            and "write_results" in self.minimize_kwargs
            and self.minimize_kwargs["write_results"]
        ):
            raise ValueError(
                "Please set the `write_structures` parameter to `True` to save "
                "optimized structures, instead of passing `write_results` through "
                "`minimize_kwargs`"
            )

        if not self.struct.calc:
            raise ValueError("Please attach a calculator to `struct`.")

        # Ensure lattice constants span correct range
        if self.n_volumes <= 1:
            raise ValueError("`n_volumes` must be greater than 1.")
        if not 0 < self.min_volume < 1:
            raise ValueError("`min_volume` must be between 0 and 1.")
        if self.max_volume <= 1:
            raise ValueError("`max_volume` must be greater than 1.")

        self.results = {}
        self.volumes = []
        self.energies = []
        self.lattice_scalars = empty(0)

    def run(self) -> EoSResults:
        """
        Calculate equation of state.

        Returns
        -------
        EoSResults
            Dictionary containing equation of state ASE object, and the fitted minimum
            bulk modulus, volume, and energy.
        """
        if self.minimize:
            if self.logger:
                self.logger.info("Minimising initial structure")
                self.minimize_kwargs["log_kwargs"] = {
                    "filename": self.log_kwargs["filename"],
                    "name": self.logger.name,
                    "filemode": "a",
                }
            optimizer = GeomOpt(self.struct, **self.minimize_kwargs)
            optimizer.run()

            # Optionally write structure to file
            output_structs(
                images=self.struct,
                write_results=self.write_structures,
                write_kwargs=self.write_kwargs,
            )

        # Set constant volume for geometry optimization of generated structures
        if "filter_kwargs" in self.minimize_kwargs:
            self.minimize_kwargs["filter_kwargs"]["constant_volume"] = True
        else:
            self.minimize_kwargs["filter_kwargs"] = {"constant_volume": True}

        self._calc_volumes_energies()

        if self.write_results:
            with open(f"{self.file_prefix}-eos-raw.dat", "w", encoding="utf8") as out:
                print("#Lattice Scalar | Energy [eV] | Volume [Å^3] ", file=out)
                for eos_data in zip(self.lattice_scalars, self.energies, self.volumes):
                    print(*eos_data, file=out)

        eos = EquationOfState(self.volumes, self.energies, self.eos_type)

        if self.logger:
            self.logger.info("Starting of fitting equation of state")
            self.tracker.start_task("Fit EoS")

        v_0, e_0, bulk_modulus = eos.fit()
        # transform bulk modulus unit in GPa
        bulk_modulus *= 1.0e24 / kJ

        if self.logger:
            self.tracker.stop_task()
            self.tracker.stop()
            self.logger.info("Equation of state fitting complete")

        if self.write_results:
            with open(f"{self.file_prefix}-eos-fit.dat", "w", encoding="utf8") as out:
                print("#Bulk modulus [GPa] | Energy [eV] | Volume [Å^3] ", file=out)
                print(bulk_modulus, e_0, v_0, file=out)

        self.results = {
            "eos": eos,
            "bulk_modulus": bulk_modulus,
            "e_0": e_0,
            "v_0": v_0,
        }

        return self.results

    def _calc_volumes_energies(self) -> None:
        """Calculate volumes and energies for all lattice constants."""

        if self.logger:
            self.logger.info("Starting calculations for configurations")
            self.tracker.start_task("Calculate configurations")

        cell = self.struct.get_cell()

        self.lattice_scalars = cbrt(
            linspace(self.min_volume, self.max_volume, self.n_volumes)
        )
        for lattice_scalar in self.lattice_scalars:
            c_struct = self.struct.copy()
            c_struct.calc = deepcopy(self.struct.calc)
            c_struct.set_cell(cell * lattice_scalar, scale_atoms=True)

            # Minimize new structure
            if self.minimize_all:
                if self.logger:
                    self.logger.info("Minimising lattice scalar = %s", lattice_scalar)
                optimizer = GeomOpt(c_struct, **self.minimize_kwargs)
                optimizer.run()

            self.volumes.append(c_struct.get_volume())
            self.energies.append(c_struct.get_potential_energy())

            # Always append first original structure
            self.write_kwargs["append"] = True
            # Write structures, but no need to set info c_struct is not used elsewhere
            output_structs(
                images=c_struct,
                write_results=self.write_structures,
                set_info=False,
                write_kwargs=self.write_kwargs,
            )

        if self.logger:
            self.tracker.stop_task()
            self.logger.info("Calculations for configurations complete")
