"""Equation of State."""

from __future__ import annotations

from copy import copy
from typing import Any

from ase import Atoms
from ase.eos import EquationOfState
from ase.units import kJ
from numpy import cbrt, empty, linspace

from janus_core.calculations.base import BaseCalculation
from janus_core.calculations.geom_opt import GeomOpt
from janus_core.helpers.janus_types import (
    Architectures,
    ASEReadArgs,
    Devices,
    EoSNames,
    EoSResults,
    OutputKwargs,
    PathLike,
)
from janus_core.helpers.struct_io import output_structs
from janus_core.helpers.utils import build_file_dir, none_to_dict, set_minimize_logging


class EoS(BaseCalculation):
    """
    Prepare and calculate equation of state of a structure.

    Parameters
    ----------
    struct
        ASE Atoms structure, or filepath to structure to simulate.
    arch
        MLIP architecture to use for calculations. Default is `None`.
    device
        Device to run MLIP model on. Default is "cpu".
    model
        MLIP model label, path to model, or loaded model. Default is `None`.
    model_path
        Deprecated. Please use `model`.
    read_kwargs
        Keyword arguments to pass to ase.io.read. By default,
        read_kwargs["index"] is -1.
    calc_kwargs
        Keyword arguments to pass to the selected calculator. Default is {}.
    attach_logger
        Whether to attach a logger. Default is True if "filename" is passed in
        log_kwargs, else False.
    log_kwargs
        Keyword arguments to pass to `config_logger`. Default is {}.
    track_carbon
        Whether to track carbon emissions of calculation. Default is True if
        attach_logger is True, else False.
    tracker_kwargs
        Keyword arguments to pass to `config_tracker`. Default is {}.
    min_volume
        Minimum volume scale factor. Default is 0.95.
    max_volume
        Maximum volume scale factor. Default is 1.05.
    n_volumes
        Number of volumes to use. Default is 7.
    eos_type
        Type of fit for equation of state. Default is "birchmurnaghan".
    minimize
        Whether to minimize initial structure before calculations. Default is True.
    minimize_all
        Whether to optimize geometry for all generated structures. Default is False.
    minimize_kwargs
        Keyword arguments to pass to optimize. Default is `None`.
    write_results
        True to write out results of equation of state calculations. Default is True.
    write_structures
        True to write out all genereated structures. Default is False.
    write_kwargs
        Keyword arguments to pass to ase.io.write to save generated structures.
        Default is {}.
    plot_to_file
        Whether to save plot equation of state to svg. Default is False.
    plot_kwargs
        Keyword arguments to pass to EquationOfState.plot. Default is {}.
    file_prefix
        Prefix for output filenames. Default is inferred from structure name, or
        chemical formula of the structure.

    Attributes
    ----------
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

    def __init__(
        self,
        struct: Atoms | PathLike,
        arch: Architectures | None = None,
        device: Devices = "cpu",
        model: PathLike | None = None,
        model_path: PathLike | None = None,
        read_kwargs: ASEReadArgs | None = None,
        calc_kwargs: dict[str, Any] | None = None,
        attach_logger: bool | None = None,
        log_kwargs: dict[str, Any] | None = None,
        track_carbon: bool | None = None,
        tracker_kwargs: dict[str, Any] | None = None,
        min_volume: float = 0.95,
        max_volume: float = 1.05,
        n_volumes: int = 7,
        eos_type: EoSNames = "birchmurnaghan",
        minimize: bool = True,
        minimize_all: bool = False,
        minimize_kwargs: dict[str, Any] | None = None,
        write_results: bool = True,
        write_structures: bool = False,
        write_kwargs: OutputKwargs | None = None,
        plot_to_file: bool = False,
        plot_kwargs: dict[str, Any] | None = None,
        file_prefix: PathLike | None = None,
    ) -> None:
        """
        Initialise class.

        Parameters
        ----------
        struct
            ASE Atoms structure, or filepath to structure to simulate.
        arch
            MLIP architecture to use for optimization. Default is `None`.
        device
            Device to run MLIP model on. Default is "cpu".
        model
            MLIP model label, path to model, or loaded model. Default is `None`.
        model_path
            Deprecated. Please use `model`.
        read_kwargs
            Keyword arguments to pass to ase.io.read. By default,
            read_kwargs["index"] is -1.
        calc_kwargs
            Keyword arguments to pass to the selected calculator. Default is {}.
        attach_logger
            Whether to attach a logger. Default is True if "filename" is passed in
            log_kwargs, else False.
        log_kwargs
            Keyword arguments to pass to `config_logger`. Default is {}.
        track_carbon
            Whether to track carbon emissions of calculation. Requires attach_logger.
            Default is True if attach_logger is True, else False.
        tracker_kwargs
            Keyword arguments to pass to `config_tracker`. Default is {}.
        min_volume
            Minimum volume scale factor. Default is 0.95.
        max_volume
            Maximum volume scale factor. Default is 1.05.
        n_volumes
            Number of volumes to use. Default is 7.
        eos_type
            Type of fit for equation of state. Default is "birchmurnaghan".
        minimize
            Whether to minimize initial structure before calculations. Default is True.
        minimize_all
            Whether to optimize geometry for all generated structures. Default is
            False.
        minimize_kwargs
            Keyword arguments to pass to optimize. Default is `None`.
        write_results
            True to write out results of equation of state calculations. Default is
            True.
        write_structures
            True to write out all genereated structures. Default is False.
        write_kwargs
            Keyword arguments to pass to ase.io.write to save generated structures.
            Default is {}.
        plot_to_file
            Whether to save plot equation of state to svg. Default is False.
        plot_kwargs
            Keyword arguments to pass to EquationOfState.plot. Default is {}.
        file_prefix
            Prefix for output filenames. Default is inferred from structure name, or
            chemical formula of the structure.
        """
        read_kwargs, minimize_kwargs, write_kwargs, plot_kwargs = none_to_dict(
            read_kwargs, minimize_kwargs, write_kwargs, plot_kwargs
        )

        self.min_volume = min_volume
        self.max_volume = max_volume
        self.n_volumes = n_volumes
        self.eos_type = eos_type
        self.minimize = minimize
        self.minimize_all = minimize_all
        self.minimize_kwargs = minimize_kwargs
        self.write_results = write_results
        self.write_structures = write_structures
        self.write_kwargs = write_kwargs
        self.plot_to_file = plot_to_file
        self.plot_kwargs = plot_kwargs

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

        # Ensure lattice constants span correct range
        if self.n_volumes <= 1:
            raise ValueError("`n_volumes` must be greater than 1.")
        if not 0 < self.min_volume < 1:
            raise ValueError("`min_volume` must be between 0 and 1.")
        if self.max_volume <= 1:
            raise ValueError("`max_volume` must be greater than 1.")

        # Read last image by default
        read_kwargs.setdefault("index", -1)

        # Initialise structures and logging
        super().__init__(
            struct=struct,
            calc_name=__name__,
            arch=arch,
            device=device,
            model=model,
            model_path=model_path,
            read_kwargs=read_kwargs,
            sequence_allowed=False,
            calc_kwargs=calc_kwargs,
            attach_logger=attach_logger,
            log_kwargs=log_kwargs,
            track_carbon=track_carbon,
            tracker_kwargs=tracker_kwargs,
            file_prefix=file_prefix,
        )

        if not self.struct.calc:
            raise ValueError("Please attach a calculator to `struct`.")

        set_minimize_logging(
            self.logger, self.minimize_kwargs, self.log_kwargs, track_carbon
        )

        # Set output files
        self.write_kwargs["filename"] = self._build_filename(
            "generated.extxyz", filename=self.write_kwargs.get("filename")
        )

        self.plot_kwargs["filename"] = self._build_filename(
            "eos-plot.svg", filename=self.plot_kwargs.get("filename")
        )
        self.fit_file = self._build_filename("eos-fit.dat")
        self.raw_file = self._build_filename("eos-raw.dat")

        self.results = {}
        self.volumes = []
        self.energies = []
        self.lattice_scalars = empty(0)

    @property
    def output_files(self) -> None:
        """
        Dictionary of output file labels and paths.

        Returns
        -------
        dict[str, PathLike]
            Output file labels and paths.
        """
        return {
            "log": self.log_kwargs["filename"] if self.logger else None,
            "generated_structures": self.write_kwargs["filename"]
            if self.write_structures
            else None,
            "plot": self.plot_kwargs["filename"] if self.plot_to_file else None,
            "fit": self.fit_file if self.write_results else None,
            "raw": self.raw_file if self.write_results else None,
        }

    def run(self) -> EoSResults:
        """
        Calculate equation of state.

        Returns
        -------
        EoSResults
            Dictionary containing equation of state ASE object, and the fitted minimum
            bulk modulus, volume, and energy.
        """
        self._set_info_units()

        if self.minimize:
            if self.logger:
                self.logger.info("Minimising initial structure")
            optimizer = GeomOpt(self.struct, **self.minimize_kwargs)
            optimizer.run()

            # Optionally write structure to file
            output_structs(
                images=self.struct,
                struct_path=self.struct_path,
                write_results=self.write_structures,
                write_kwargs=self.write_kwargs,
                config_type="eos",
            )

        # Set constant volume for geometry optimization of generated structures
        if "filter_kwargs" in self.minimize_kwargs:
            self.minimize_kwargs["filter_kwargs"]["constant_volume"] = True
        else:
            self.minimize_kwargs["filter_kwargs"] = {"constant_volume": True}

        self._calc_volumes_energies()

        if self.write_results:
            build_file_dir(self.raw_file)
            with open(self.raw_file, "w", encoding="utf8") as out:
                print("#Lattice Scalar | Energy [eV] | Volume [Å^3] ", file=out)
                for eos_data in zip(
                    self.lattice_scalars, self.energies, self.volumes, strict=True
                ):
                    print(*eos_data, file=out)

        eos = EquationOfState(self.volumes, self.energies, self.eos_type)

        if self.logger:
            self.logger.info("Starting of fitting equation of state")
        if self.tracker:
            self.tracker.start_task("Fit EoS")

        v_0, e_0, bulk_modulus = eos.fit()
        # transform bulk modulus unit in GPa
        bulk_modulus *= 1.0e24 / kJ

        if self.logger:
            self.logger.info("Equation of state fitting complete")
        if self.tracker:
            emissions = self.tracker.stop_task().emissions
            self.struct.info["emissions"] = emissions
            self.tracker.stop()

        if self.write_results:
            build_file_dir(self.fit_file)
            with open(self.fit_file, "w", encoding="utf8") as out:
                print("#Bulk modulus [GPa] | Energy [eV] | Volume [Å^3] ", file=out)
                print(bulk_modulus, e_0, v_0, file=out)

        self.results = {
            "eos": eos,
            "bulk_modulus": bulk_modulus,
            "e_0": e_0,
            "v_0": v_0,
        }

        if self.plot_to_file:
            build_file_dir(self.plot_kwargs["filename"])
            eos.plot(**self.plot_kwargs)

        return self.results

    def _calc_volumes_energies(self) -> None:
        """Calculate volumes and energies for all lattice constants."""
        if self.logger:
            self.logger.info("Starting calculations for configurations")
        if self.tracker:
            self.tracker.start_task("Calculate configurations")

        cell = self.struct.get_cell()

        self.lattice_scalars = cbrt(
            linspace(self.min_volume, self.max_volume, self.n_volumes)
        )
        for lattice_scalar in self.lattice_scalars:
            c_struct = self.struct.copy()
            c_struct.calc = copy(self.struct.calc)
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
                struct_path=self.struct_path,
                write_results=self.write_structures,
                set_info=False,
                write_kwargs=self.write_kwargs,
                config_type="eos",
            )

        if self.logger:
            self.logger.info("Calculations for configurations complete")
        if self.tracker:
            emissions = self.tracker.stop_task().emissions
            self.struct.info["emissions"] = emissions
