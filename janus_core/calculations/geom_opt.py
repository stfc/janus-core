"""Prepare and run geometry optimization."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
import warnings

from ase import Atoms, filters, units
from ase.filters import FrechetCellFilter
from ase.io import read
import ase.optimize
from ase.optimize import LBFGS
from numpy import linalg

from janus_core.calculations.base import BaseCalculation
from janus_core.helpers.janus_types import (
    Architectures,
    ASEOptArgs,
    ASEReadArgs,
    Devices,
    OutputKwargs,
    PathLike,
)
from janus_core.helpers.struct_io import output_structs
from janus_core.helpers.utils import none_to_dict
from janus_core.processing.symmetry import snap_symmetry, spacegroup


class GeomOpt(BaseCalculation):
    """
    Prepare and run geometry optimization.

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
    file_prefix
        Prefix for output filenames. Default is inferred from structure.
    fmax
        Set force convergence criteria for optimizer in units eV/Å. Default is 0.1.
    steps
        Set maximum number of optimization steps to run. Default is 1000.
    symmetrize
        Whether to refine symmetry after geometry optimization. Default is False.
    symmetry_tolerance
        Atom displacement tolerance for spglib symmetry determination, in Å.
        Default is 0.001.
    angle_tolerance
        Angle precision for spglib symmetry determination, in degrees. Default is -1.0,
        which means an internally optimized routine is used to judge symmetry.
    filter_class
        Filter class, or name of class from ase.filters to wrap around atoms.
        Default is `FrechetCellFilter`.
    filter_func
        Deprecated. Please use `filter_class`.
    filter_kwargs
        Keyword arguments to pass to filter_class. Default is {}.
    optimizer
        Optimization function, or name of function from ase.optimize. Default is
        `LBFGS`.
    opt_kwargs
        Keyword arguments to pass to optimizer. Default is {}.
    write_results
        True to write out optimized structure. Default is False.
    write_kwargs
        Keyword arguments to pass to ase.io.write to save optimized structure.
        Default is {}.
    write_traj
        Whether to save a trajectory file of optimization frames.
    traj_kwargs
        Keyword arguments to pass to ase.io.write to save optimization trajectory.
        "filename" keyword is inferred from `file_prefix` if not given. Default is {}.
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
        file_prefix: PathLike | None = None,
        fmax: float = 0.1,
        steps: int = 1000,
        symmetrize: bool = False,
        symmetry_tolerance: float = 0.001,
        angle_tolerance: float = -1.0,
        filter_class: Callable | str | None = FrechetCellFilter,
        filter_func: Callable | str | None = None,
        filter_kwargs: dict[str, Any] | None = None,
        optimizer: Callable | str = LBFGS,
        opt_kwargs: ASEOptArgs | None = None,
        write_results: bool = False,
        write_kwargs: OutputKwargs | None = None,
        write_traj: bool = False,
        traj_kwargs: OutputKwargs | None = None,
    ) -> None:
        """
        Initialise GeomOpt class.

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
        file_prefix
            Prefix for output filenames. Default is inferred from structure.
        fmax
            Set force convergence criteria for optimizer in units eV/Å. Default is 0.1.
        steps
            Set maximum number of optimization steps to run. Default is 1000.
        symmetrize
            Whether to refine symmetry after geometry optimization. Default is False.
        symmetry_tolerance
            Atom displacement tolerance for spglib symmetry determination, in Å.
            Default is 0.001.
        angle_tolerance
            Angle precision for spglib symmetry determination, in degrees. Default is
            -1.0, which means an internally optimized routine is used to judge symmetry.
        filter_class
            Filter class, or name of class from ase.filters to wrap around atoms.
            Default is `FrechetCellFilter`.
        filter_func
            Deprecated. Please use `filter_class`.
        filter_kwargs
            Keyword arguments to pass to filter_class. Default is {}.
        optimizer
            Optimization function, or name of function from ase.optimize. Default is
            `LBFGS`.
        opt_kwargs
            Keyword arguments to pass to optimizer. Default is {}.
        write_results
            True to write out optimized structure. Default is False.
        write_kwargs
            Keyword arguments to pass to ase.io.write to save optimized structure.
            Default is {}.
        write_traj
            Whether to save a trajectory file of optimization frames.
        traj_kwargs
            Keyword arguments to pass to ase.io.write to save optimization trajectory.
            "filename" keyword is inferred from `file_prefix` if not given.
            Default is {}.
        """
        read_kwargs, filter_kwargs, opt_kwargs, write_kwargs, traj_kwargs = (
            none_to_dict(
                read_kwargs, filter_kwargs, opt_kwargs, write_kwargs, traj_kwargs
            )
        )

        # Handle deprecated filter_func, but warn later after logging is set up
        if filter_func:
            filter_class = filter_func

        self.fmax = fmax
        self.steps = steps
        self.symmetrize = symmetrize
        self.symmetry_tolerance = symmetry_tolerance
        self.angle_tolerance = angle_tolerance
        self.filter_class = filter_class
        self.filter_kwargs = filter_kwargs
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        self.write_results = write_results
        self.write_kwargs = write_kwargs
        self.write_traj = write_traj
        self.traj_kwargs = traj_kwargs

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

        # Warn for deprecated filter_func now that logging is set up
        if filter_func:
            warnings.warn(
                "`filter_func` has been deprecated, but currently overrides "
                "`filter_class`. Please only set `filter_class`.",
                FutureWarning,
                stacklevel=2,
            )

        # Set output files
        self.write_kwargs["filename"] = self._build_filename(
            "opt.extxyz", filename=self.write_kwargs.get("filename")
        )

        if self.write_traj:
            if "trajectory" in self.opt_kwargs:
                raise ValueError(
                    "Please use traj_kwargs['filename'] to save the trajectory"
                )

            # Set filenames for trajectory, and ensure directories exist
            self.traj_kwargs.setdefault(
                "filename", self._build_filename("traj.extxyz").absolute()
            )
            Path(self.traj_kwargs["filename"]).parent.mkdir(parents=True, exist_ok=True)
            self.opt_kwargs["trajectory"] = str(self.traj_kwargs["filename"])

        elif self.traj_kwargs:
            raise ValueError(
                "traj_kwargs given, but trajectory writing not enabled via write_traj."
            )

        elif "trajectory" in self.opt_kwargs:
            raise ValueError(
                "Please use write_traj, and optionally traj_kwargs['filename'] to "
                "save the trajectory"
            )

        # Configure optimizer dynamics
        self.set_optimizer()

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
            "optimized_structure": self.write_kwargs["filename"]
            if self.write_results
            else None,
            "trajectory": self.traj_kwargs.get("filename"),
        }

    def set_optimizer(self) -> None:
        """Set optimizer for geometry optimization."""
        self._set_functions()
        if self.logger:
            self.logger.info("Using optimizer: %s", self.optimizer.__name__)

        if self.filter_class is not None:
            if "scalar_pressure" in self.filter_kwargs:
                self.filter_kwargs["scalar_pressure"] *= units.GPa
            self.filtered_struct = self.filter_class(self.struct, **self.filter_kwargs)
            self.dyn = self.optimizer(self.filtered_struct, **self.opt_kwargs)
            if self.logger:
                self.logger.info("Using filter: %s", self.filter_class.__name__)
                if "hydrostatic_strain" in self.filter_kwargs:
                    self.logger.info(
                        "hydrostatic_strain: %s",
                        self.filter_kwargs["hydrostatic_strain"],
                    )
                if "constant_volume" in self.filter_kwargs:
                    self.logger.info(
                        "constant_volume: %s", self.filter_kwargs["constant_volume"]
                    )
                if "scalar_pressure" in self.filter_kwargs:
                    self.logger.info(
                        "scalar_pressure: %s GPa",
                        self.filter_kwargs["scalar_pressure"] / units.GPa,
                    )
        else:
            self.dyn = self.optimizer(self.struct, **self.opt_kwargs)

    def _set_functions(self) -> None:
        """Set optimizer and filter."""
        if isinstance(self.optimizer, str):
            try:
                self.optimizer = getattr(ase.optimize, self.optimizer)
            except AttributeError as e:
                raise AttributeError(f"No such optimizer: {self.optimizer}") from e

        if self.filter_class is not None and isinstance(self.filter_class, str):
            try:
                self.filter_class = getattr(filters, self.filter_class)
            except AttributeError as e:
                raise AttributeError(f"No such filter: {self.filter_class}") from e

    def run(self) -> None:
        """Run geometry optimization."""
        s_grp = spacegroup(self.struct, self.symmetry_tolerance, self.angle_tolerance)
        self.struct.info["initial_spacegroup"] = s_grp
        if self.logger:
            self.logger.info("Before optimisation spacegroup: %s", s_grp)

        if self.logger:
            self.logger.info("Starting geometry optimization")
        if self.tracker:
            self.tracker.start_task("Geometry optimization")

        self._set_info_units()

        converged = self.dyn.run(fmax=self.fmax, steps=self.steps)

        # Calculate current maximum force
        if self.filter_class is not None:
            max_force = linalg.norm(self.filtered_struct.get_forces(), axis=1).max()
        else:
            max_force = linalg.norm(self.struct.get_forces(), axis=1).max()

        if self.symmetrize:
            snap_symmetry(self.struct, self.symmetry_tolerance)

            # Update max force
            old_max_force = max_force
            struct = (
                self.filtered_struct if self.filter_class is not None else self.struct
            )
            max_force = linalg.norm(struct.get_forces(), axis=1).max()

            if max_force >= old_max_force:
                warnings.warn(
                    "Refining symmetry increased the maximum force", stacklevel=2
                )

        s_grp = spacegroup(self.struct, self.symmetry_tolerance, self.angle_tolerance)
        self.struct.info["final_spacegroup"] = s_grp

        if self.logger:
            self.logger.info("After optimization spacegroup: %s", s_grp)
            self.logger.info("Max force: %s", max_force)
            self.logger.info("Final energy: %s", self.struct.get_potential_energy())

        if not converged:
            warnings.warn(
                f"Optimization has not converged after {self.steps} steps. "
                f"Current max force {max_force} > target force {self.fmax}",
                stacklevel=2,
            )

        # Write out optimized structure
        output_structs(
            self.struct,
            struct_path=self.struct_path,
            write_results=self.write_results,
            write_kwargs=self.write_kwargs,
            config_type="geom_opt",
        )

        # Reformat trajectory file from binary
        if self.write_traj:
            traj = read(self.opt_kwargs["trajectory"], index=":")
            output_structs(
                traj,
                struct_path=self.struct_path,
                write_results=True,
                write_kwargs=self.traj_kwargs,
                config_type="geom_opt",
            )

        if self.logger:
            self.logger.info("Geometry optimization complete")
        if self.tracker:
            emissions = self.tracker.stop_task().emissions
            self.struct.info["emissions"] = emissions
            self.tracker.stop()
