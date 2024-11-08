"""Prepare and run geometry optimization."""

from __future__ import annotations

from typing import Any, Callable
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
    struct : Atoms | None
        ASE Atoms structure to optimize geometry for. Required if `struct_path` is
        None. Default is None.
    struct_path : PathLike | None
        Path of structure to optimize. Required if `struct` is None. Default is None.
    arch : Architectures
        MLIP architecture to use for optimization. Default is "mace_mp".
    device : Devices
        Device to run MLIP model on. Default is "cpu".
    model_path : PathLike | None
        Path to MLIP model. Default is `None`.
    read_kwargs : ASEReadArgs | None
        Keyword arguments to pass to ase.io.read. By default,
        read_kwargs["index"] is -1.
    calc_kwargs : dict[str, Any] | None
        Keyword arguments to pass to the selected calculator. Default is {}.
    set_calc : bool | None
        Whether to set (new) calculators for structures. Default is None.
    attach_logger : bool
        Whether to attach a logger. Default is False.
    log_kwargs : dict[str, Any] | None
        Keyword arguments to pass to `config_logger`. Default is {}.
    track_carbon : bool
        Whether to track carbon emissions of calculation. Default is True.
    tracker_kwargs : dict[str, Any] | None
        Keyword arguments to pass to `config_tracker`. Default is {}.
    fmax : float
        Set force convergence criteria for optimizer in units eV/Å. Default is 0.1.
    steps : int
        Set maximum number of optimization steps to run. Default is 1000.
    symmetrize : bool
        Whether to refine symmetry after geometry optimization. Default is False.
    symmetry_tolerance : float
        Atom displacement tolerance for spglib symmetry determination, in Å.
        Default is 0.001.
    angle_tolerance : float
        Angle precision for spglib symmetry determination, in degrees. Default is -1.0,
        which means an internally optimized routine is used to judge symmetry.
    filter_func : Callable | str | None
        Filter function, or name of function from ase.filters to apply constraints to
        atoms. Default is `FrechetCellFilter`.
    filter_kwargs : dict[str, Any] | None
        Keyword arguments to pass to filter_func. Default is {}.
    optimizer : Callable | str
        Optimization function, or name of function from ase.optimize. Default is
        `LBFGS`.
    opt_kwargs : ASEOptArgs | None
        Keyword arguments to pass to optimizer. Default is {}.
    write_results : bool
        True to write out optimized structure. Default is False.
    write_kwargs : OutputKwargs | None
        Keyword arguments to pass to ase.io.write to save optimized structure.
        Default is {}.
    traj_kwargs : OutputKwargs | None
        Keyword arguments to pass to ase.io.write to save optimization trajectory.
        Must include "filename" keyword. Default is {}.

    Methods
    -------
    set_optimizer()
        Set optimizer for geometry optimization.
    run()
        Run geometry optimization.
    """

    def __init__(
        self,
        struct: Atoms | None = None,
        struct_path: PathLike | None = None,
        arch: Architectures = "mace_mp",
        device: Devices = "cpu",
        model_path: PathLike | None = None,
        read_kwargs: ASEReadArgs | None = None,
        calc_kwargs: dict[str, Any] | None = None,
        set_calc: bool | None = None,
        attach_logger: bool = False,
        log_kwargs: dict[str, Any] | None = None,
        track_carbon: bool = True,
        tracker_kwargs: dict[str, Any] | None = None,
        fmax: float = 0.1,
        steps: int = 1000,
        symmetrize: bool = False,
        symmetry_tolerance: float = 0.001,
        angle_tolerance: float = -1.0,
        filter_func: Callable | str | None = FrechetCellFilter,
        filter_kwargs: dict[str, Any] | None = None,
        optimizer: Callable | str = LBFGS,
        opt_kwargs: ASEOptArgs | None = None,
        write_results: bool = False,
        write_kwargs: OutputKwargs | None = None,
        traj_kwargs: OutputKwargs | None = None,
    ) -> None:
        """
        Initialise GeomOpt class.

        Parameters
        ----------
        struct : Atoms | None
            ASE Atoms structure to optimize geometry for. Required if `struct_path` is
            None. Default is None.
        struct_path : PathLike | None
            Path of structure to optimize. Required if `struct` is None. Default is
            None.
        arch : Architectures
            MLIP architecture to use for optimization. Default is "mace_mp".
        device : Devices
            Device to run MLIP model on. Default is "cpu".
        model_path : PathLike | None
            Path to MLIP model. Default is `None`.
        read_kwargs : ASEReadArgs | None
            Keyword arguments to pass to ase.io.read. By default,
            read_kwargs["index"] is -1.
        calc_kwargs : dict[str, Any] | None
            Keyword arguments to pass to the selected calculator. Default is {}.
        set_calc : bool | None
            Whether to set (new) calculators for structures. Default is None.
        attach_logger : bool
            Whether to attach a logger. Default is False.
        log_kwargs : dict[str, Any] | None
            Keyword arguments to pass to `config_logger`. Default is {}.
        track_carbon : bool
            Whether to track carbon emissions of calculation. Default is True.
        tracker_kwargs : dict[str, Any] | None
            Keyword arguments to pass to `config_tracker`. Default is {}.
        fmax : float
            Set force convergence criteria for optimizer in units eV/Å. Default is 0.1.
        steps : int
            Set maximum number of optimization steps to run. Default is 1000.
        symmetrize : bool
            Whether to refine symmetry after geometry optimization. Default is False.
        symmetry_tolerance : float
            Atom displacement tolerance for spglib symmetry determination, in Å.
            Default is 0.001.
        angle_tolerance : float
            Angle precision for spglib symmetry determination, in degrees. Default is
            -1.0, which means an internally optimized routine is used to judge symmetry.
        filter_func : Callable | str | None
            Filter function, or name of function from ase.filters to apply constraints
            to atoms. Default is `FrechetCellFilter`.
        filter_kwargs : dict[str, Any] | None
            Keyword arguments to pass to filter_func. Default is {}.
        optimizer : Callable | str
            Optimization function, or name of function from ase.optimize. Default is
            `LBFGS`.
        opt_kwargs : ASEOptArgs | None
            Keyword arguments to pass to optimizer. Default is {}.
        write_results : bool
            True to write out optimized structure. Default is False.
        write_kwargs : OutputKwargs | None
            Keyword arguments to pass to ase.io.write to save optimized structure.
            Default is {}.
        traj_kwargs : OutputKwargs | None
            Keyword arguments to pass to ase.io.write to save optimization trajectory.
            Must include "filename" keyword. Default is {}.
        """
        read_kwargs, filter_kwargs, opt_kwargs, write_kwargs, traj_kwargs = (
            none_to_dict(
                read_kwargs, filter_kwargs, opt_kwargs, write_kwargs, traj_kwargs
            )
        )

        self.fmax = fmax
        self.steps = steps
        self.symmetrize = symmetrize
        self.symmetry_tolerance = symmetry_tolerance
        self.angle_tolerance = angle_tolerance
        self.filter_func = filter_func
        self.filter_kwargs = filter_kwargs
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        self.write_results = write_results
        self.write_kwargs = write_kwargs
        self.traj_kwargs = traj_kwargs

        # Validate parameters
        if self.traj_kwargs and "filename" not in self.traj_kwargs:
            raise ValueError("'filename' must be included in `traj_kwargs`")

        if self.traj_kwargs and "trajectory" not in self.opt_kwargs:
            raise ValueError(
                "'trajectory' must be a key in `opt_kwargs` to save the trajectory."
            )

        # Read last image by default
        read_kwargs.setdefault("index", -1)

        # Initialise structures and logging
        super().__init__(
            calc_name=__name__,
            struct=struct,
            struct_path=struct_path,
            arch=arch,
            device=device,
            model_path=model_path,
            read_kwargs=read_kwargs,
            sequence_allowed=False,
            calc_kwargs=calc_kwargs,
            set_calc=set_calc,
            attach_logger=attach_logger,
            log_kwargs=log_kwargs,
            track_carbon=track_carbon,
            tracker_kwargs=tracker_kwargs,
        )

        if not self.struct.calc:
            raise ValueError("Please attach a calculator to `struct`.")

        # Set output file
        self.write_kwargs.setdefault("filename", None)
        self.write_kwargs["filename"] = self._build_filename(
            "opt.extxyz", filename=self.write_kwargs["filename"]
        ).absolute()

        # Configure optimizer dynamics
        self.set_optimizer()

    def set_optimizer(self) -> None:
        """Set optimizer for geometry optimization."""
        self._set_functions()
        if self.logger:
            self.logger.info("Using optimizer: %s", self.optimizer.__name__)

        if self.filter_func is not None:
            if "scalar_pressure" in self.filter_kwargs:
                self.filter_kwargs["scalar_pressure"] *= units.GPa
            self.filtered_struct = self.filter_func(self.struct, **self.filter_kwargs)
            self.dyn = self.optimizer(self.filtered_struct, **self.opt_kwargs)
            if self.logger:
                self.logger.info("Using filter: %s", self.filter_func.__name__)
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
        """Set optimizer and filter functions."""
        if isinstance(self.optimizer, str):
            try:
                self.optimizer = getattr(ase.optimize, self.optimizer)
            except AttributeError as e:
                raise AttributeError(f"No such optimizer: {self.optimizer}") from e

        if self.filter_func is not None and isinstance(self.filter_func, str):
            try:
                self.filter_func = getattr(filters, self.filter_func)
            except AttributeError as e:
                raise AttributeError(f"No such filter: {self.filter_func}") from e

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

        converged = self.dyn.run(fmax=self.fmax, steps=self.steps)

        # Calculate current maximum force
        if self.filter_func is not None:
            max_force = linalg.norm(self.filtered_struct.get_forces(), axis=1).max()
        else:
            max_force = linalg.norm(self.struct.get_forces(), axis=1).max()

        if self.symmetrize:
            snap_symmetry(self.struct, self.symmetry_tolerance)

            # Update max force
            old_max_force = max_force
            struct = (
                self.filtered_struct if self.filter_func is not None else self.struct
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
        )

        # Reformat trajectory file from binary
        if self.traj_kwargs:
            traj = read(self.opt_kwargs["trajectory"], index=":")
            output_structs(
                traj,
                struct_path=self.struct_path,
                write_results=True,
                write_kwargs=self.traj_kwargs,
            )

        if self.logger:
            self.logger.info("Geometry optimization complete")
        if self.tracker:
            emissions = self.tracker.stop_task().emissions
            self.struct.info["emissions"] = emissions
            self.tracker.stop()
