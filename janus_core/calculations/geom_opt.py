"""Geometry optimization."""

from typing import Any, Callable, Optional, Union
import warnings

from ase import Atoms, filters, units
from ase.filters import FrechetCellFilter
from ase.io import read
import ase.optimize
from ase.optimize import LBFGS
from numpy import linalg

from janus_core.helpers.janus_types import ASEOptArgs, OutputKwargs
from janus_core.helpers.log import config_logger, config_tracker
from janus_core.helpers.utils import (
    FileNameMixin,
    none_to_dict,
    output_structs,
    spacegroup,
)


class GeomOpt(FileNameMixin):  # pylint: disable=too-many-instance-attributes
    """
    Prepare and run geometry optimization.

    Parameters
    ----------
    struct : Atoms
        Atoms object to optimize geometry for.
    fmax : float
        Set force convergence criteria for optimizer in units eV/Å.
        Default is 0.1.
    steps : int
        Set maximum number of optimization steps to run. Default is 1000.
    symmetry_tolerance : float
        Atom displacement tolerance for spglib symmetry determination, in Å.
        Default is 0.001.
    angle_tolerance : float
        Angle precision for spglib symmetry determination, in degrees. Default is -1.0,
        which means an internally optimized routine is used to judge symmetry.
    filter_func : Optional[Union[Callable, str]]
        Filter function, or name of function from ase.filters to apply constraints to
        atoms. Default is `FrechetCellFilter`.
    filter_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to filter_func. Default is {}.
    optimizer : Union[Callable, str]
        Optimization function, or name of function from ase.optimize. Default is
        `LBFGS`.
    opt_kwargs : Optional[ASEOptArgs]
        Keyword arguments to pass to optimizer. Default is {}.
    write_results : bool
        True to write out optimized structure. Default is False.
    write_kwargs : Optional[OutputKwargs],
        Keyword arguments to pass to ase.io.write to save optimized structure.
        Default is {}.
    traj_kwargs : Optional[OutputKwargs]
        Keyword arguments to pass to ase.io.write to save optimization trajectory.
        Must include "filename" keyword. Default is {}.
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
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        struct: Atoms,
        fmax: float = 0.1,
        steps: int = 1000,
        symmetry_tolerance: float = 0.001,
        angle_tolerance: float = -1.0,
        filter_func: Optional[Union[Callable, str]] = FrechetCellFilter,
        filter_kwargs: Optional[dict[str, Any]] = None,
        optimizer: Union[Callable, str] = LBFGS,
        opt_kwargs: Optional[ASEOptArgs] = None,
        write_results: bool = False,
        write_kwargs: Optional[OutputKwargs] = None,
        traj_kwargs: Optional[OutputKwargs] = None,
        log_kwargs: Optional[dict[str, Any]] = None,
        tracker_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialise GeomOpt class.

        Parameters
        ----------
        struct : Atoms
            Atoms object to optimize geometry for.
        fmax : float
            Set force convergence criteria for optimizer in units eV/Å.
            Default is 0.1.
        steps : int
            Set maximum number of optimization steps to run. Default is 1000.
        symmetry_tolerance : float
            Atom displacement tolerance for spglib symmetry determination, in Å.
            Default is 0.001.
        angle_tolerance : float
            Angle precision for spglib symmetry determination, in degrees. Default is
            -1.0, which means an internally optimized routine is used to judge
            symmetry.
        filter_func : Optional[Union[Callable, str]]
            Filter function, or name of function from ase.filters to apply constraints
            to atoms. Default is `FrechetCellFilter`.
        filter_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to filter_func. Default is {}.
        optimizer : Union[Callable, str]
            Optimization function, or name of function from ase.optimize. Default is
            `LBFGS`.
        opt_kwargs : Optional[ASEOptArgs]
            Keyword arguments to pass to optimizer. Default is {}.
        write_results : bool
            True to write out optimized structure. Default is False.
        write_kwargs : Optional[OutputKwargs],
            Keyword arguments to pass to ase.io.write to save optimized structure.
            Default is {}.
        traj_kwargs : Optional[OutputKwargs]
            Keyword arguments to pass to ase.io.write to save optimization trajectory.
            Must include "filename" keyword. Default is {}.
        log_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to `config_logger`. Default is {}.
        tracker_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to `config_tracker`. Default is {}.
        """
        self.struct = struct
        self.fmax = fmax
        self.steps = steps
        self.symmetry_tolerance = symmetry_tolerance
        self.angle_tolerance = angle_tolerance
        self.filter_func = filter_func
        self.optimizer = optimizer
        self.write_results = write_results

        [
            filter_kwargs,
            opt_kwargs,
            write_kwargs,
            traj_kwargs,
            log_kwargs,
            tracker_kwargs,
        ] = none_to_dict(
            [
                filter_kwargs,
                opt_kwargs,
                write_kwargs,
                traj_kwargs,
                log_kwargs,
                tracker_kwargs,
            ]
        )
        self.filter_kwargs = filter_kwargs
        self.opt_kwargs = opt_kwargs
        self.write_kwargs = write_kwargs
        self.traj_kwargs = traj_kwargs

        FileNameMixin.__init__(self, self.struct, None, None)

        self.write_kwargs.setdefault(
            "filename",
            self._build_filename("opt.extxyz").absolute(),
        )

        if traj_kwargs and "filename" not in traj_kwargs:
            raise ValueError("'filename' must be included in `traj_kwargs`")

        if traj_kwargs and "trajectory" not in opt_kwargs:
            raise ValueError(
                "'trajectory' must be a key in `opt_kwargs` to save the trajectory."
            )

        if log_kwargs and "filename" not in log_kwargs:
            raise ValueError("'filename' must be included in `log_kwargs`")

        log_kwargs.setdefault("name", __name__)
        self.logger = config_logger(**log_kwargs)
        self.tracker = config_tracker(self.logger, **tracker_kwargs)

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

    def _set_functions(self):
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

    def run(self):
        """Run geometry optimization."""
        s_grp = spacegroup(self.struct, self.symmetry_tolerance, self.angle_tolerance)
        self.struct.info["initial_spacegroup"] = s_grp
        if self.logger:
            self.logger.info("Before optimisation spacegroup: %s", s_grp)

        if self.logger:
            self.logger.info("Starting geometry optimization")
            self.tracker.start()

        converged = self.dyn.run(fmax=self.fmax, steps=self.steps)

        s_grp = spacegroup(self.struct, self.symmetry_tolerance, self.angle_tolerance)
        self.struct.info["final_spacegroup"] = s_grp

        # Calculate current maximum force
        if self.filter_func is not None:
            max_force = linalg.norm(self.filtered_struct.get_forces(), axis=1).max()
        else:
            max_force = linalg.norm(self.struct.get_forces(), axis=1).max()

        if self.logger:
            self.logger.info("After optimization spacegroup: %s", s_grp)
            self.logger.info("Max force: %.6f", max_force)

        if not converged:
            warnings.warn(
                f"Optimization has not converged after {self.steps} steps. "
                f"Current max force {max_force} > target force {self.fmax}"
            )

        # Write out optimized structure
        output_structs(
            self.struct,
            write_results=self.write_results,
            write_kwargs=self.write_kwargs,
        )

        # Reformat trajectory file from binary
        if self.traj_kwargs:
            traj = read(self.opt_kwargs["trajectory"], index=":")
            output_structs(
                traj,
                write_results=True,
                write_kwargs=self.traj_kwargs,
            )

        if self.logger:
            self.tracker.stop()
            self.logger.info("Geometry optimization complete")
