"""Run molecular dynamics simulations."""

from __future__ import annotations

from collections.abc import Sequence
import datetime
from functools import partial
from itertools import combinations_with_replacement
from math import isclose
from os.path import getmtime
from pathlib import Path
import random
from typing import Any
from warnings import warn

from ase import Atoms
from ase.geometry.analysis import Analysis
from ase.io import read
from ase.md.bussi import Bussi
from ase.md.langevin import Langevin
from ase.md.npt import NPT as ASE_NPT
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet
from ase.units import create_units
import numpy as np
from rich.progress import Progress, TaskID
import yaml

from janus_core.calculations.base import UNITS as JANUS_UNITS
from janus_core.calculations.base import BaseCalculation
from janus_core.calculations.geom_opt import GeomOpt
from janus_core.helpers.janus_types import (
    Architectures,
    ASEReadArgs,
    CorrelationKwargs,
    Devices,
    Ensembles,
    OutputKwargs,
    PathLike,
    PostProcessKwargs,
)
from janus_core.helpers.struct_io import input_structs, output_structs
from janus_core.helpers.utils import (
    build_file_dir,
    get_progress,
    none_to_dict,
    set_minimize_logging,
    write_table,
)
from janus_core.processing.correlator import Correlation
from janus_core.processing.post_process import compute_rdf, compute_vaf

units = create_units("2014")
DENS_FACT = (units.m / 1.0e2) ** 3 / units.mol


class MolecularDynamics(BaseCalculation):
    """
    Configure shared molecular dynamics simulation options.

    Parameters
    ----------
    struct
        ASE Atoms structure, or filepath to structure to simulate.
    arch
        MLIP architecture to use for simulation. Default is `None`.
    device
        Device to run MLIP model on. Default is "cpu".
    model
        Path to MLIP model or name of model. Default is `None`.
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
    struct
        Structure to simulate.
    ensemble
        Name for thermodynamic ensemble. Default is `None`.
    steps
        Number of steps in simulation. Default is 0.
    timestep
        Timestep for integrator, in fs. Default is 1.0.
    temp
        Temperature, in K. Default is 300.
    equil_steps
        Maximum number of steps at which to perform optimization and reset velocities.
        Default is 0.
    minimize
        Whether to minimize structure during equilibration. Default is False.
    minimize_every
        Frequency of minimizations. Default is -1, which disables minimization after
        beginning dynamics.
    minimize_kwargs
        Keyword arguments to pass to geometry optimizer. Default is {}.
    rescale_velocities
        Whether to rescale velocities. Default is False.
    remove_rot
        Whether to remove rotation. Default is False.
    rescale_every
        Frequency to rescale velocities. Default is 10.
    file_prefix
        Prefix for output filenames. Default is inferred from structure, ensemble,
        and temperature.
    restart
        Whether restarting dynamics. Default is False.
    restart_auto
        Whether to infer restart file name if restarting dynamics. Default is True.
    restart_stem
        Stem for restart file name. Default inferred from `file_prefix`.
    restart_every
        Frequency of steps to save restart info. Default is 1000.
    rotate_restart
        Whether to rotate restart files. Default is False.
    restarts_to_keep
        Restart files to keep if rotating. Default is 4.
    final_file
        File to save final configuration at each temperature of similation. Default
        inferred from `file_prefix`.
    stats_file
        File to save thermodynamical statistics. Default inferred from `file_prefix`.
    stats_every
        Frequency to output statistics. Default is 100.
    traj_file
        Trajectory file to save. Default inferred from `file_prefix`.
    traj_append
        Whether to append trajectory. Default is False.
    traj_start
        Step to start saving trajectory. Default is 0.
    traj_every
        Frequency of steps to save trajectory. Default is 100.
    temp_start
        Temperature to start heating, in K. Default is None, which disables heating.
    temp_end
        Maximum temperature for heating, in K. Default is None, which disables heating.
    temp_step
        Size of temperature steps when heating, in K. Default is None, which disables
        heating.
    temp_time
        Time between heating steps, in fs. Default is None, which disables heating.
    write_kwargs
        Keyword arguments to pass to `output_structs` when saving trajectory and final
        files. Default is {}.
    post_process_kwargs
        Keyword arguments to control post-processing operations. Default is `None`.
    correlation_kwargs
        Keyword arguments to control on-the-fly correlations. Default is `None`.
    plumed_input
        Path to a PLUMED input file. If provided, the ASE Plumed calculator will be
        used to wrap the MLIP calculator. Requires the `plumed` extra to be installed,
        and configured. Default is None.
    plumed_log
        Path for the PLUMED log file. Default is inferred from `file_prefix`.
    seed
        Random seed used by numpy.random and random functions, such as in Langevin.
        Default is `None`.
    enable_progress_bar
        Whether to show a progress bar. Default is False.
    update_progress_every
        Number of timesteps between progress bar updates. Default is steps / 100,
        rounded up.

    Attributes
    ----------
    dyn : Dynamics
        Dynamics object to run simulation.
    n_atoms : int
        Number of atoms in structure being simulated.
    restart_files : list[PathLike]
        List of files saved to restart dynamics.
    offset : int
        Number of previous steps if restarting simulation.
    created_final : bool
        Whether the final structure file has been created.
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
        ensemble: Ensembles | None = None,
        steps: int = 0,
        timestep: float = 1.0,
        temp: float = 300,
        equil_steps: int = 0,
        minimize: bool = False,
        minimize_every: int = -1,
        minimize_kwargs: dict[str, Any] | None = None,
        rescale_velocities: bool = False,
        remove_rot: bool = False,
        rescale_every: int = 10,
        file_prefix: PathLike | None = None,
        restart: bool = False,
        restart_auto: bool = True,
        restart_stem: PathLike | None = None,
        restart_every: int = 1000,
        rotate_restart: bool = False,
        restarts_to_keep: int = 4,
        final_file: PathLike | None = None,
        stats_file: PathLike | None = None,
        stats_every: int = 100,
        traj_file: PathLike | None = None,
        traj_append: bool = False,
        traj_start: int = 0,
        traj_every: int = 100,
        temp_start: float | None = None,
        temp_end: float | None = None,
        temp_step: float | None = None,
        temp_time: float | None = None,
        write_kwargs: OutputKwargs | None = None,
        post_process_kwargs: PostProcessKwargs | None = None,
        correlation_kwargs: list[CorrelationKwargs] | None = None,
        plumed_input: PathLike | None = None,
        plumed_log: PathLike | None = None,
        seed: int | None = None,
        enable_progress_bar: bool = False,
        update_progress_every: int | None = None,
    ) -> None:
        """
        Initialise molecular dynamics simulation configuration.

        Parameters
        ----------
        struct
            ASE Atoms structure, or filepath to structure to simulate.
        arch
            MLIP architecture to use for simulation. Default is `None`.
        device
            Device to run MLIP model on. Default is "cpu".
        model
            Path to MLIP model or name of model. Default is `None`.
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
        ensemble
            Name for thermodynamic ensemble. Default is `None`.
        steps
            Number of steps in simulation. Default is 0.
        timestep
            Timestep for integrator, in fs. Default is 1.0.
        temp
            Temperature, in K. Default is 300.
        equil_steps
            Maximum number of steps at which to perform optimization and reset
            velocities. Default is 0.
        minimize
            Whether to minimize structure during equilibration. Default is False.
        minimize_every
            Frequency of minimizations. Default is -1, which disables minimization
            after beginning dynamics.
        minimize_kwargs
            Keyword arguments to pass to geometry optimizer. Default is {}.
        rescale_velocities
            Whether to rescale velocities. Default is False.
        remove_rot
            Whether to remove rotation. Default is False.
        rescale_every
            Frequency to rescale velocities. Default is 10.
        file_prefix
            Prefix for output filenames. Default is inferred from structure, ensemble,
            and temperature.
        restart
            Whether restarting dynamics. Default is False.
        restart_auto
            Whether to infer restart file name if restarting dynamics. Default is True.
        restart_stem
            Stem for restart file name. Default inferred from `file_prefix`.
        restart_every
            Frequency of steps to save restart info. Default is 1000.
        rotate_restart
            Whether to rotate restart files. Default is False.
        restarts_to_keep
            Restart files to keep if rotating. Default is 4.
        final_file
            File to save final configuration at each temperature of similation. Default
            inferred from `file_prefix`.
        stats_file
            File to save thermodynamical statistics. Default inferred from
            `file_prefix`.
        stats_every
            Frequency to output statistics. Default is 100.
        traj_file
            Trajectory file to save. Default inferred from `file_prefix`.
        traj_append
            Whether to append trajectory. Default is False.
        traj_start
            Step to start saving trajectory. Default is 0.
        traj_every
            Frequency of steps to save trajectory. Default is 100.
        temp_start
            Temperature to start heating, in K. Default is None, which disables
            heating.
        temp_end
            Maximum temperature for heating, in K. Default is None, which disables
            heating.
        temp_step
            Size of temperature steps when heating, in K. Default is None, which
            disables heating.
        temp_time
            Time between heating steps, in fs. Default is None, which disables heating.
        write_kwargs
            Keyword arguments to pass to `output_structs` when saving trajectory and
            final files. Default is {}.
        post_process_kwargs
            Keyword arguments to control post-processing operations. Default is `None`.
        correlation_kwargs
            Keyword arguments to control on-the-fly correlations. Default is `None`.
        plumed_input
            Path to a PLUMED input file. If provided, the ASE Plumed calculator will be
            used to wrap the MLIP calculator. Requires the `plumed` extra to be
            installed, and configured. Default is None.
        plumed_log
            Path for the PLUMED log file. Default is inferred from `file_prefix`.
        seed
            Random seed used by numpy.random and random functions, such as in Langevin.
            Default is `None`.
        enable_progress_bar
            Whether to show a progress bar. Default is False.
        update_progress_every
            Number of timesteps between progress bar updates. Default is steps / 100,
            rounded up.
        """
        (
            read_kwargs,
            minimize_kwargs,
            write_kwargs,
            post_process_kwargs,
            correlation_kwargs,
        ) = none_to_dict(
            read_kwargs,
            minimize_kwargs,
            write_kwargs,
            post_process_kwargs,
            correlation_kwargs,
        )

        self.ensemble = ensemble
        self.steps = steps
        self.timestep = timestep * units.fs
        self.temp = temp
        self.equil_steps = equil_steps
        self.minimize = minimize
        self.minimize_every = minimize_every
        self.minimize_kwargs = minimize_kwargs
        self.rescale_velocities = rescale_velocities
        self.remove_rot = remove_rot
        self.rescale_every = rescale_every
        self.restart = restart
        self.restart_auto = restart_auto
        self.restart_stem = restart_stem
        self.restart_every = restart_every
        self.rotate_restart = rotate_restart
        self.restarts_to_keep = restarts_to_keep
        self.final_file = final_file
        self.stats_file = stats_file
        self.stats_every = stats_every
        self.traj_file = traj_file
        self.traj_append = traj_append
        self.traj_start = traj_start
        self.traj_every = traj_every
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.temp_step = temp_step
        self.temp_time = temp_time * units.fs if temp_time else None
        self.write_kwargs = write_kwargs
        self.post_process_kwargs = post_process_kwargs
        self.correlation_kwargs = correlation_kwargs
        self.plumed_input = plumed_input
        self.plumed_log = plumed_log
        self.seed = seed
        self.enable_progress_bar = enable_progress_bar
        self.update_progress_every = update_progress_every

        if "append" in self.write_kwargs:
            raise ValueError("`append` cannot be specified when writing files")

        # Check temperatures for heating differ
        if self.temp_start is not None and self.temp_start == self.temp_end:
            raise ValueError("Start and end temperatures must be different")

        # Warn if attempting to rescale/minimize during dynamics
        # but equil_steps is too low
        if rescale_velocities and equil_steps < rescale_every:
            warn(
                "Velocities and angular momentum will not be reset during dynamics",
                stacklevel=2,
            )
        if minimize and equil_steps < minimize_every:
            warn("Geometry will not be minimized during dynamics", stacklevel=2)

        # Warn if attempting to remove rotation without resetting velocities
        if remove_rot and not rescale_velocities:
            warn(
                "Rotation will not be removed unless `rescale_velocities` is True",
                stacklevel=2,
            )

        # Warn if mix of None and not None
        self.ramp_temp = (
            self.temp_start is not None
            and self.temp_end is not None
            and self.temp_step
            and self.temp_time
        )
        if (
            self.temp_start is not None
            or self.temp_end is not None
            or self.temp_step
            or self.temp_time
        ) and not self.ramp_temp:
            warn(
                "`temp_start`, `temp_end` and `temp_step` must all be specified for "
                "heating to run",
                stacklevel=2,
            )
        if self.ramp_temp:
            if self.ensemble in ("nve", "nph"):
                raise ValueError(
                    "Temperature ramp requested for ensemble with no thermostat"
                )

            # Validate start and end temperatures
            if self.temp_start < 0 or self.temp_end < 0:
                raise ValueError("Start and end temperatures must be positive")

            if self.temp_time < self.timestep:
                raise ValueError(
                    "Temperature ramp step time cannot be less than 1 timestep"
                )

        # Read last image by default
        read_kwargs.setdefault("index", -1)

        self.param_prefix = self._set_param_prefix(file_prefix)

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
            additional_prefix=self.ensemble,
            param_prefix=self.param_prefix,
        )

        if not self.struct.calc:
            raise ValueError("Please attach a calculator to `struct`.")

        self._config_plumed()

        # Set output file names
        self.final_file = self._build_filename(
            "final.extxyz", self.param_prefix, filename=self.final_file
        )
        self.stats_file = self._build_filename(
            "stats.dat", self.param_prefix, filename=self.stats_file
        )
        self.traj_file = self._build_filename(
            "traj.extxyz", self.param_prefix, filename=self.traj_file
        )

        # If not specified otherwise, save optimized structure consistently with others
        opt_file = self._build_filename("opt.extxyz", self.param_prefix, filename=None)

        # Set defaults
        default_columns = ["symbols", "positions", "momenta", "masses"]
        if not write_kwargs.get("invalidate_calc", False):
            default_columns.append("forces")
        if "arch" in self.struct.calc.parameters and write_kwargs.get("set_info", True):
            default_columns.append(f"{self.arch}_forces")

        self.write_kwargs.setdefault("columns", default_columns)

        # Use MD logger for geometry optimization
        set_minimize_logging(
            self.logger, self.minimize_kwargs, self.log_kwargs, track_carbon
        )

        if "write_kwargs" in self.minimize_kwargs:
            # Use _build_filename even if given filename to ensure directory exists
            self.minimize_kwargs["write_kwargs"]["filename"] = self._build_filename(
                "", filename=self.minimize_kwargs["write_kwargs"].get("filename")
            )

            # Assume if write_kwargs are specified that results should be written
            self.minimize_kwargs.setdefault("write_results", True)
        else:
            self.minimize_kwargs["write_kwargs"] = {"filename": opt_file}
            self.minimize_kwargs.setdefault("write_results", False)

        self.dyn: Langevin | VelocityVerlet | ASE_NPT
        self.n_atoms = len(self.struct)

        self.offset = 0
        if self.restart:
            self._prepare_restart()
        self.restart_files = []
        self.created_final_file = False

        if "masses" not in self.struct.arrays:
            self.struct.set_masses()

        if self.seed:
            np.random.seed(seed)
            random.seed(seed)

        if self.update_progress_every is None:
            self.update_progress_every = np.ceil(self.total_steps / 100)

        self._parse_correlations()

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
            "stats": self.stats_file,
            "trajectory": self.traj_file,
            "final_structure": self.final_file,
            "restarts": self.restart_files if self.restart_files else None,
            "correlations": (self._correlations_file if self._correlations else None),
            "minimized_initial_structure": (
                self.minimize_kwargs["write_kwargs"]["filename"]
                if self.minimize_kwargs["write_results"]
                else None
            ),
            "rdfs": (
                self._rdf_files
                if self.post_process_kwargs.get("rdf_compute", False)
                else None
            ),
            "vafs": (
                self._vaf_files
                if self.post_process_kwargs.get("vaf_compute", False)
                else None
            ),
            "plumed_log": self.plumed_log if self.plumed_input else None,
        }

    @property
    def ramp_temps(self) -> list[float]:
        """
        Get list of temperatures in heating/cooling ramp.

        Returns
        -------
        list[float]
            Temperatures in heating/cooling ramp.
        """
        ramp_sign = 1 if (self.temp_end - self.temp_start) > 0 else -1
        ramp_temps = [
            self.temp_start + ramp_sign * i * self.temp_step
            for i in range(self.heating_n_temps)
        ]

        if self.restart:
            ramp_steps_completed = self.offset // self.heating_steps_per_temp
            ramp_steps_completed = min(ramp_steps_completed, len(ramp_temps))
            if isclose(self.temp_start, 0.0):
                # T~0K steps do not run any MD, so are not included in the offset.
                # We have to account for that step manually.
                ramp_steps_completed += 1
            ramp_temps = ramp_temps[ramp_steps_completed:]

        return ramp_temps

    @property
    def heating_n_temps(self) -> int:
        """
        Get number of temperatures in heating/cooling ramp.

        Returns
        -------
        int
            Number of temperatures in heating/cooling ramp.
        """
        # Always include start temperature in ramp, and include end temperature
        # if separated by an integer number of temperature steps
        return int(1 + abs(self.temp_end - self.temp_start) // self.temp_step)

    @property
    def heating_steps_per_temp(self) -> int:
        """
        Get number of steps to be run at each temperature in heating/cooling ramp.

        Returns
        -------
        int
            Number of steps to be run at each temperature in heating/cooling ramp.
        """
        return int(self.temp_time // self.timestep)

    @property
    def total_steps(self) -> int:
        """
        Get combined temperature ramp and constant temperature MD steps.

        Returns
        -------
        int
           Total number of steps in simulation.
        """
        total_steps = self.steps

        if self.ramp_temp:
            total_steps += self.heating_n_temps * self.heating_steps_per_temp
            # Heating steps at 0 K are skipped
            if np.isclose(self.temp_start, 0.0):
                total_steps -= self.heating_steps_per_temp
            if np.isclose(self.temp_end, 0.0):
                total_steps -= self.heating_steps_per_temp

        return total_steps

    def _config_plumed(self):
        """Wrap calculator with Plumed if input is provided."""
        if self.plumed_input:
            if isinstance(self, NPT | NPH | NPT_MTK):
                raise NotImplementedError("Plumed calculator does not implement stress")

            try:
                from ase.calculators.plumed import Plumed as PlumedCalc
                from plumed import Plumed

                # Raise runtime error if PLUMED_KERNEL environment variable not set
                Plumed()
            except ImportError as err:
                raise ImportError(
                    "PLUMED Python package not found. Please install PLUMED with its "
                    "Python interface to use this feature."
                ) from err
            except RuntimeError as err:
                if any("PLUMED not available" in arg for arg in err.args):
                    raise RuntimeError(
                        "Please ensure PLUMED_KERNEL environment variable is set"
                    ) from err
                raise

            # Read input script from file
            plumed_input_path = Path(self.plumed_input)
            if not plumed_input_path.is_file():
                raise FileNotFoundError(
                    f"PLUMED input file not found or is not a file: {plumed_input_path}"
                )
            plumed_script = plumed_input_path.read_text(encoding="utf-8")

            self.plumed_log = self._build_filename(
                "plumed.log", self.param_prefix, filename=self.plumed_log
            )

            build_file_dir(self.plumed_log)

            self.struct.calc = PlumedCalc(
                calc=self.struct.calc,
                input=plumed_script.splitlines(),
                timestep=self.timestep,
                atoms=self.struct,
                kT=self.temp * units.kB,  # kT in ASE units (eV)
                log=str(self.plumed_log),
                restart=self.restart,
            )

            if self.logger:
                self.logger.info("Plumed calculator configured")

    def _init_progress_bar(self) -> Progress:
        """
        Initialise MD progress bar.

        Returns
        -------
        ProgressBar
            Object used for managing progress bars.
        """
        if not self.enable_progress_bar:
            return Progress(disable=True)
        progress_bar = get_progress()

        task_id = progress_bar.add_task(
            "Initialising MD simulation...",
            total=self.total_steps,
            completed=self.offset,
        )

        update_func = partial(self._update_progress_bar, progress_bar, task_id)
        self.dyn.attach(update_func, interval=self.update_progress_every)

        # Also ensure progress is updated at the end
        self.dyn.attach(update_func, interval=-self.total_steps)
        return progress_bar

    def _update_progress_bar(
        self,
        progress_bar: Progress,
        task_id: TaskID,
    ):
        """
        Update the progress bar.

        Parameters
        ----------
        progress_bar
            Progress bar object tracking all tasks.
        task_id
            Task ID tracking overall simulation progress.
        """
        current_step = self.dyn.nsteps + self.offset

        # 0th step, no progress yet.
        if current_step == 0:
            return

        progress_bar.update(
            task_id,
            completed=current_step,
            description=f"Simulating at {self.temp:.1f} K...",
        )
        progress_bar.refresh()

    def _set_info(self) -> None:
        """Set time in fs, current dynamics step, density, and temperature to info."""
        time = (self.offset * self.timestep + self.dyn.get_time()) / units.fs
        step = self.offset + self.dyn.nsteps
        self.dyn.atoms.info["time"] = time
        self.dyn.atoms.info["step"] = step
        try:
            density = (
                np.sum(self.dyn.atoms.get_masses())
                / self.dyn.atoms.get_volume()
                * DENS_FACT
            )
            self.dyn.atoms.info["density"] = density
        except ValueError:
            self.dyn.atoms.info["density"] = 0.0

        e_kin = self.dyn.atoms.get_kinetic_energy() / self.n_atoms
        current_temp = e_kin / (1.5 * units.kB)
        self.struct.info["temperature"] = current_temp

        if hasattr(self.dyn, "set_temperature") or isinstance(self, NVT_CSVR | NPT_MTK):
            self.struct.info["target_temperature"] = self.temp

    def _prepare_restart(self) -> None:
        """Prepare restart files, structure and offset."""
        # Check offset can be read from steps
        try:
            with open(self.stats_file, encoding="utf8") as stats_file:
                last_line = stats_file.readlines()[-1].split()
            try:
                self.offset = int(last_line[0])
            except (IndexError, ValueError) as e:
                raise ValueError("Unable to read restart file") from e
        except FileNotFoundError as e:
            raise FileNotFoundError("Unable to read restart file") from e

        if self.restart_auto:
            # Attempt to infer restart file
            restart_stem = self._restart_stem

            # Use restart_stem.name otherwise T300.0 etc. counts as extension
            poss_restarts = restart_stem.parent.glob(f"{restart_stem.name}*.extxyz")
            try:
                last_restart = max(poss_restarts, key=getmtime)

                # Read in last structure
                self.struct, self.struct_path = input_structs(
                    struct=last_restart,
                    read_kwargs=self.read_kwargs,
                    sequence_allowed=False,
                    arch=self.arch,
                    device=self.device,
                    model=self.model,
                    calc_kwargs=self.calc_kwargs,
                    logger=self.logger,
                )

                self._config_plumed()

                # Infer last dynamics step
                last_stem = last_restart.stem
                try:
                    # Remove restart_stem from filename
                    # Use restart_stem.name otherwise T300.0 etc. counts as extension
                    self.offset = int(last_stem.split("-")[-1])

                    # Check "-" not inlcuded in offset
                    assert self.offset > 0

                except (ValueError, AssertionError) as e:
                    raise ValueError(
                        f"Unable to infer final dynamics step from {last_restart}"
                    ) from e

                if self.logger:
                    self.logger.info("Auto restart successful")

            except IndexError:
                if self.logger:
                    self.logger.info(
                        "Auto restart failed with stem: %s. Using `struct`",
                        restart_stem,
                    )

        # Check files exist to append
        if not self.stats_file.exists() or not self.traj_file.exists():
            raise ValueError(
                "Statistics and trajectory files must already exist to restart "
                "simulation"
            )

        # Positions and momenta already set to match last config
        # Plumed calculator istep must be set too
        if self.plumed_input:
            self.struct.calc.istep = self.offset

    def _rotate_restart_files(self) -> None:
        """Rotate restart files."""
        if len(self.restart_files) > self.restarts_to_keep:
            path = Path(self.restart_files.pop(0))
            path.unlink(missing_ok=True)

    def _set_velocity_distribution(self) -> None:
        """
        Set velocities to current target temperature.

        Sets Maxwell-Boltzmann velocity distribution, as well as removing
        centre-of-mass momentum, and (optionally) total angular momentum.
        """
        atoms = self.struct

        MaxwellBoltzmannDistribution(atoms, temperature_K=self.temp)
        Stationary(atoms)
        if self.logger:
            self.logger.info("Velocities reset at step %s", self.dyn.nsteps)
        if self.remove_rot:
            ZeroRotation(atoms)
            if self.logger:
                self.logger.info("Rotation reset at step %s", self.dyn.nsteps)

    def _reset_velocities(self) -> None:
        """Reset velocities and (optionally) rotation of system while equilibrating."""
        if self.dyn.nsteps < self.equil_steps:
            self._set_velocity_distribution()

    def _optimize_structure(self) -> None:
        """Perform geometry optimization."""
        if self.dyn.nsteps == 0 or self.dyn.nsteps < self.equil_steps:
            if self.logger:
                self.logger.info("Minimizing at step %s", self.dyn.nsteps)
            optimizer = GeomOpt(self.struct, **self.minimize_kwargs)
            optimizer.run()

    def _set_param_prefix(self, file_prefix: PathLike | None = None) -> str:
        """
        Set ensemble parameters for output files.

        Parameters
        ----------
        file_prefix
            Prefix for output filenames on class init. If not None, param_prefix = "".

        Returns
        -------
        str
           Formatted ensemble parameters, including temp ramp range and/or and MD temp.
        """
        if file_prefix is not None:
            return ""

        temperature_prefix = ""
        if self.temp_start is not None and self.temp_end is not None:
            temperature_prefix += f"-T{self.temp_start}-T{self.temp_end}"

        if self.steps > 0:
            temperature_prefix += f"-T{self.temp}"

        return temperature_prefix.lstrip("-")

    @property
    def _restart_stem(self) -> str:
        """
        Stem for restart files.

        Restart files will be named {restart_stem}-{step}.extxyz. If file_prefix is
        specified, restart_stem will be of the form {file_prefix}-{param_prefix}-res.

        Returns
        -------
        str
           Stem for restart files.
        """
        if self.restart_stem is not None:
            return Path(self.restart_stem)

        # param_prefix is empty if file_prefix was None on init
        return Path(
            "-".join(filter(None, (str(self.file_prefix), self.param_prefix, "res")))
        )

    @property
    def _restart_file(self) -> str:
        """
        Restart file name.

        Returns
        -------
        str
           File name for restart files.
        """
        step = self.offset + self.dyn.nsteps
        return self._build_filename(
            f"{step}.extxyz", prefix_override=self._restart_stem
        )

    @property
    def _correlations_file(self) -> str:
        """
        Define correlations file name.

        Returns
        -------
        str
           File name for correlations files.
        """
        return self._build_filename("cor.dat", self.param_prefix)

    @property
    def _rdf_files(self) -> tuple[Path]:
        """
        Get RDF filenames.

        Returns
        -------
        str
           Filenames for RDF file.
        """
        base_name = self.post_process_kwargs.get("rdf_output_file", None)
        rdf_args = {
            name: self.post_process_kwargs.get(key, default)
            for name, (key, default) in (
                ("elements", ("rdf_elements", None)),
                ("by_elements", ("rdf_by_elements", False)),
            )
        }

        if rdf_args["by_elements"]:
            elements = (
                tuple(sorted(set(self.struct.get_chemical_symbols())))
                if rdf_args["elements"] is None
                else rdf_args["elements"]
            )

            out_paths = tuple(
                self._build_filename(
                    "rdf.dat",
                    self.param_prefix,
                    "_".join(element),
                    prefix_override=base_name,
                )
                for element in combinations_with_replacement(elements, 2)
            )

        else:
            out_paths = (
                self._build_filename(
                    "rdf.dat", self.param_prefix, prefix_override=base_name
                ),
            )

        return out_paths

    @property
    def _vaf_files(self) -> str:
        """
        Define VAF filenames.

        Returns
        -------
        str
           Filenames for VAF files.
        """
        file_names = self.post_process_kwargs.get("vaf_output_files", None)
        if not isinstance(file_names, Sequence):
            file_names = (file_names,)

        return tuple(
            self._build_filename("vaf.dat", self.param_prefix, filename=file_name)
            for file_name in file_names
        )

    def _parse_correlations(self) -> None:
        """Parse correlation kwargs into Correlations."""
        if self.correlation_kwargs:
            self._correlations = [Correlation(**cor) for cor in self.correlation_kwargs]
        else:
            self._correlations = ()

    def _attach_correlations(self) -> None:
        """Attach all correlations to self.dyn."""
        for i, _ in enumerate(self._correlations):
            self.dyn.attach(
                partial(lambda i: self._correlations[i].update(self.dyn.atoms), i),
                self._correlations[i].update_frequency,
            )

    def _write_correlations(self) -> None:
        """Write out the correlations."""
        if self._correlations:
            build_file_dir(self._correlations_file)
            with open(self._correlations_file, "w", encoding="utf-8") as out_file:
                data = {}
                for cor in self._correlations:
                    value, lags = cor.get()
                    data[str(cor)] = {"value": value.tolist(), "lags": lags.tolist()}
                yaml.dump(data, out_file, default_flow_style=None)

    @property
    def info_unit_keys(self) -> tuple[str]:
        """
        Get Atoms.info keys to save units for.

        Returns
        -------
        tuple[str]
            Keys of Atoms.info.
        """
        return (
            "energy",
            "forces",
            "stress",
            "time",
            "real_time",
            "temperature",
            "pressure",
            "density",
            "momenta",
        )

    def get_stats(self) -> dict[str, float]:
        """
        Get thermodynamical statistics to be written to file.

        Returns
        -------
        dict[str, float]
            Thermodynamical statistics to be written out.
        """
        e_pot = self.dyn.atoms.get_potential_energy() / self.n_atoms
        e_kin = self.dyn.atoms.get_kinetic_energy() / self.n_atoms

        self._set_info()

        time_now = datetime.datetime.now()
        real_time = time_now - self.dyn.atoms.info["real_time"]
        self.dyn.atoms.info["real_time"] = time_now

        try:
            volume = self.dyn.atoms.get_volume()
            pressure = (
                -np.trace(
                    self.dyn.atoms.get_stress(include_ideal_gas=True, voigt=False)
                )
                / 3
                / units.GPa
            )
            pressure_tensor = (
                -self.dyn.atoms.get_stress(include_ideal_gas=True, voigt=True)
                / units.GPa
            )
        except (ValueError, NotImplementedError):
            # ASE's Plumed wrapper does not implement get_stress().
            # This does not affect the simulation, only statistics logging
            volume = 0.0
            pressure = 0.0
            pressure_tensor = np.zeros(6)

        return {
            "Step": self.dyn.atoms.info["step"],
            "Real_Time": real_time.total_seconds(),
            "Time": self.dyn.atoms.info["time"],
            "Epot/N": e_pot,
            "EKin/N": e_kin,
            "T": self.dyn.atoms.info["temperature"],
            "ETot/N": e_pot + e_kin,
            "Density": self.dyn.atoms.info["density"],
            "Volume": volume,
            "P": pressure,
            "Pxx": pressure_tensor[0],
            "Pyy": pressure_tensor[1],
            "Pzz": pressure_tensor[2],
            "Pyz": pressure_tensor[3],
            "Pxz": pressure_tensor[4],
            "Pxy": pressure_tensor[5],
        }

    @property
    def stats_units(self) -> dict[str, str]:
        """
        Get units of returned statistics.

        Returns
        -------
        dict[str, str]
            Units attached to statistical properties.
        """
        return {
            "Step": None,
            "Real_Time": JANUS_UNITS["real_time"],
            "Time": JANUS_UNITS["time"],
            "Epot/N": JANUS_UNITS["energy"],
            "EKin/N": JANUS_UNITS["energy"],
            "T": JANUS_UNITS["temperature"],
            "ETot/N": JANUS_UNITS["energy"],
            "Density": JANUS_UNITS["density"],
            "Volume": JANUS_UNITS["volume"],
            "P": JANUS_UNITS["pressure"],
            "Pxx": JANUS_UNITS["pressure"],
            "Pyy": JANUS_UNITS["pressure"],
            "Pzz": JANUS_UNITS["pressure"],
            "Pyz": JANUS_UNITS["pressure"],
            "Pxz": JANUS_UNITS["pressure"],
            "Pxy": JANUS_UNITS["pressure"],
        }

    @property
    def default_formats(self) -> dict[str, str]:
        """
        Default format of returned statistics.

        Returns
        -------
        dict[str, str]
            Default formats attached to statistical properties.
        """
        return {
            "Step": "10d",
            "Real_Time": ".3f",
            "Time": "13.2f",
            "Epot/N": ".8e",
            "EKin/N": ".8e",
            "T": ".3f",
            "ETot/N": ".8e",
            "Density": ".3f",
            "Volume": ".8e",
            "P": ".8e",
            "Pxx": ".8e",
            "Pyy": ".8e",
            "Pzz": ".8e",
            "Pyz": ".8e",
            "Pxz": ".8e",
            "Pxy": ".8e",
        }

    def _write_header(self) -> None:
        """Write header for stats file."""
        build_file_dir(self.stats_file)
        with open(self.stats_file, "w", encoding="utf-8") as stats_file:
            write_table(
                "ascii",
                file=stats_file,
                units=self.stats_units,
                **dict.fromkeys(self.stats_units, ()),
            )

    def _write_stats_file(self) -> None:
        """Write molecular dynamics statistics."""
        stats = self.get_stats()

        # Do not print step 0 for restarts
        if self.restart and self.dyn.nsteps == 0:
            return

        with open(self.stats_file, "a", encoding="utf8") as stats_file:
            write_table(
                "ascii",
                file=stats_file,
                units=self.stats_units,
                formats=self.default_formats,
                print_header=False,
                **stats,
            )

    def _write_traj(self) -> None:
        """Write current structure to trajectory file."""
        # Do not save step 0 for restarts
        if self.restart and self.dyn.nsteps == 0:
            return

        if self.dyn.nsteps >= self.traj_start:
            # Append if restarting or already started writing
            append = self.restart or (
                self.dyn.nsteps > self.traj_start + self.traj_start % self.traj_every
            )

            self._set_info()
            write_kwargs = self.write_kwargs
            write_kwargs["filename"] = self.traj_file
            write_kwargs["append"] = append

            output_structs(
                images=self.struct,
                struct_path=self.struct_path,
                write_results=True,
                write_kwargs=write_kwargs,
                config_type="md",
            )

    def _write_final_state(self) -> None:
        """Write the final system state."""
        # Append if final file has been created
        append = self.created_final_file

        self._set_info()
        write_kwargs = self.write_kwargs
        write_kwargs["filename"] = self.final_file
        write_kwargs["append"] = append

        output_structs(
            images=self.struct,
            struct_path=self.struct_path,
            write_results=True,
            write_kwargs=write_kwargs,
            config_type="md",
        )

    def _post_process(self) -> None:
        """Compute properties after MD run."""
        # Nothing to do
        if not any(
            self.post_process_kwargs.get(kwarg, None)
            for kwarg in ("rdf_compute", "vaf_compute")
        ):
            warn(
                "Post-processing arguments present, but no computation requested. "
                "Please set either 'rdf_compute' or 'vaf_compute' "
                "to do post-processing.",
                stacklevel=2,
            )

        data = read(self.traj_file, index=":")

        ana = Analysis(data)

        if self.post_process_kwargs.get("rdf_compute", False):
            rdf_args = {
                name: self.post_process_kwargs.get(key, default)
                for name, (key, default) in (
                    ("rmax", ("rdf_rmax", 2.5)),
                    ("nbins", ("rdf_nbins", 50)),
                    ("elements", ("rdf_elements", None)),
                    ("by_elements", ("rdf_by_elements", False)),
                )
            }
            slice_ = (
                self.post_process_kwargs.get("rdf_start", len(data) - 1),
                self.post_process_kwargs.get("rdf_stop", len(data)),
                self.post_process_kwargs.get("rdf_step", 1),
            )
            rdf_args["index"] = slice_

            compute_rdf(data, ana, filenames=self._rdf_files, **rdf_args)

        if self.post_process_kwargs.get("vaf_compute", False):
            use_vel = self.post_process_kwargs.get("vaf_velocities", False)
            fft = self.post_process_kwargs.get("vaf_fft", False)

            slice_ = (
                self.post_process_kwargs.get("vaf_start", 0),
                self.post_process_kwargs.get("vaf_stop", None),
                self.post_process_kwargs.get("vaf_step", 1),
            )

            compute_vaf(
                data,
                self._vaf_files,
                use_velocities=use_vel,
                fft=fft,
                index=slice_,
                atoms_filter=self.post_process_kwargs.get("vaf_atoms", None),
            )

    def _write_restart(self) -> None:
        """Write restart file and (optionally) rotate files saved."""
        step = self.offset + self.dyn.nsteps
        if step > 0:
            write_kwargs = self.write_kwargs
            write_kwargs["filename"] = self._restart_file
            self._set_info()

            output_structs(
                images=self.struct,
                struct_path=self.struct_path,
                write_results=True,
                write_kwargs=write_kwargs,
                config_type="md",
            )
            self.restart_files.append(self._restart_file)
            if self.rotate_restart:
                self._rotate_restart_files()

    def _set_target_temperature(self, temperature: float):
        """
        Set the target temperature of the thermostat.

        Parameters
        ----------
        temperature
            New target temperature, in K.
        """
        if hasattr(self.dyn, "set_temperature"):
            self.dyn.set_temperature(temperature_K=temperature)
        elif isinstance(self, NVT_CSVR):
            self.dyn.temp = temperature * units.kB
            self.dyn.target_kinetic_energy = 0.5 * self.dyn.temp * self.dyn.ndof
        elif isinstance(self, NPT_MTK):
            kt = temperature * units.kB
            self.dyn._kT = kt
            self.dyn._thermostat._kT = kt
            self.dyn._barostat._kT = kt
        else:
            raise ValueError("Temperature set for ensemble with no thermostat.")

    def run(self) -> None:
        """Run molecular dynamics simulation and/or temperature ramp."""
        self._set_info_units(self.info_unit_keys)

        if not self.restart:
            if self.minimize:
                self._optimize_structure()
            if self.rescale_velocities:
                self._reset_velocities()

        if self.offset == 0:
            self._write_header()

        self.dyn.attach(self._write_stats_file, interval=self.stats_every)
        self.dyn.attach(self._write_traj, interval=self.traj_every)
        self.dyn.attach(self._write_restart, interval=self.restart_every)

        self._attach_correlations()

        if self.rescale_velocities:
            self.dyn.attach(self._reset_velocities, interval=self.rescale_every)

        if self.minimize and self.minimize_every > 0:
            self.dyn.attach(self._optimize_structure, interval=self.minimize_every)

        # Note current time
        self.struct.info["real_time"] = datetime.datetime.now()
        with self._init_progress_bar():
            self._run_dynamics()

        if self.post_process_kwargs:
            self._post_process()

        self._write_correlations()

    def _run_dynamics(self) -> None:
        """Run dynamics and/or temperature ramp."""
        # Store temperature for final MD
        md_temp = self.temp
        if self.ramp_temp:
            self.temp = self.temp_start

        # Set velocities to match current temperature
        if not self.restart:
            self._set_velocity_distribution()

        # Run temperature ramp
        if self.ramp_temp:
            if self.logger and not np.isclose(self.temp_time % self.timestep, 0.0):
                rounded_temp_step = (
                    self.heating_steps_per_temp * self.timestep / units.fs
                )
                self.logger.info(
                    "Temperature ramp step time rounded to nearest timestep "
                    f"({rounded_temp_step:.5} fs)"
                )

            if self.ramp_temps:
                if self.logger:
                    self.logger.info(
                        "Beginning temperature ramp at %sK", self.ramp_temps[0]
                    )
                if self.tracker:
                    self.tracker.start_task("Temperature ramp")

            first_step = True
            for temp in self.ramp_temps:
                self.temp = temp
                steps = self.heating_steps_per_temp
                if first_step:
                    first_step = False
                    steps -= self.offset % self.heating_steps_per_temp
                self._set_velocity_distribution()
                if isclose(temp, 0.0):
                    # Calculate forces and energies to be output
                    self.struct.get_potential_energy()
                    self.struct.get_forces()
                    self._write_final_state()
                    self.created_final_file = True
                    continue
                self._set_target_temperature(temp)
                self.dyn.run(steps)
                self._write_final_state()
                self.created_final_file = True

            if self.ramp_temps:
                if self.logger:
                    self.logger.info(
                        "Temperature ramp complete at %sK", self.ramp_temps[-1]
                    )
                if self.tracker:
                    emissions = self.tracker.stop_task().emissions
                    self.struct.info["emissions"] = emissions

        md_offset = self.offset
        if self.restart and self.ramp_temp:
            # Take the ramp time off the offset for MD.
            # If restarting during the ramp, MD has no offset.
            md_offset = max(
                0, md_offset - self.heating_steps_per_temp * self.heating_n_temps
            )

        # Run MD
        if self.steps > 0:
            if self.logger:
                self.logger.info("Starting molecular dynamics simulation")
            if self.tracker:
                self.tracker.start_task("Molecular dynamics")
            self.temp = md_temp
            if self.ramp_temp:
                self._set_velocity_distribution()
                self._set_target_temperature(self.temp)
            self.dyn.run(self.steps - md_offset)
            self._write_final_state()
            self.created_final_file = True
            if self.logger:
                self.logger.info("Molecular dynamics simulation complete")
            if self.tracker:
                emissions = self.tracker.stop_task().emissions
                self.struct.info["emissions"] = emissions
                self.tracker.stop()


class NPT(MolecularDynamics):
    """
    Configure NPT dynamics.

    Parameters
    ----------
    *args
        Additional arguments.
    thermostat_time
        Thermostat time, in fs. Default is 50.0.
    barostat_time
        Barostat time, in fs. Default is 75.0.
    bulk_modulus
        Bulk modulus, in GPa. Default is 2.0.
    pressure
        Pressure, in GPa. Default is 0.0.
    ensemble
        Name for thermodynamic ensemble. Default is "npt".
    file_prefix
        Prefix for output filenames. Default is inferred from structure, ensemble,
        temperature, and pressure.
    ensemble_kwargs
        Keyword arguments to pass to ensemble initialization. Default is {}.
    **kwargs
        Additional keyword arguments.
    """

    def __init__(
        self,
        *args,
        thermostat_time: float = 50.0,
        barostat_time: float = 75.0,
        bulk_modulus: float = 2.0,
        pressure: float = 0.0,
        ensemble: Ensembles = "npt",
        file_prefix: PathLike | None = None,
        ensemble_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """
        Initialise dynamics for NPT simulation.

        Parameters
        ----------
        *args
            Additional arguments.
        thermostat_time
            Thermostat time, in fs. Default is 50.0.
        barostat_time
            Barostat time, in fs. Default is 75.0.
        bulk_modulus
            Bulk modulus, in GPa. Default is 2.0.
        pressure
            Pressure, in GPa. Default is 0.0.
        ensemble
            Name for thermodynamic ensemble. Default is "npt".
        file_prefix
            Prefix for output filenames. Default is inferred from structure, ensemble,
            temperature, and pressure.
        ensemble_kwargs
            Keyword arguments to pass to ensemble initialization. Default is {}.
        **kwargs
            Additional keyword arguments.
        """
        self.pressure = pressure
        super().__init__(*args, ensemble=ensemble, file_prefix=file_prefix, **kwargs)

        (ensemble_kwargs,) = none_to_dict(ensemble_kwargs)
        self.ttime = thermostat_time * units.fs

        if barostat_time:
            pfactor = barostat_time**2 * bulk_modulus
            if self.logger:
                self.logger.info("NPT pfactor=%s GPa fs^2", pfactor)

            # convert the pfactor to ASE internal units
            pfactor *= units.fs**2 * units.GPa
        else:
            pfactor = None
        self.dyn = ASE_NPT(
            self.struct,
            timestep=self.timestep,
            temperature_K=self.temp,
            ttime=self.ttime,
            pfactor=pfactor,
            append_trajectory=self.traj_append,
            externalstress=self.pressure * units.GPa,
            **ensemble_kwargs,
        )

    def _set_param_prefix(self, file_prefix: PathLike | None = None) -> str:
        """
        Set ensemble parameters for output files.

        Parameters
        ----------
        file_prefix
            Prefix for output filenames on class init. If not None, param_prefix = "".

        Returns
        -------
        str
           Formatted ensemble parameters, including pressure and temperature(s).
        """
        if file_prefix is not None:
            return ""

        return f"{super()._set_param_prefix(file_prefix)}-p{self.pressure}"

    @property
    def info_unit_keys(self) -> tuple[str]:
        """
        Get Atoms.info keys to save units for.

        Returns
        -------
        tuple[str]
            Keys of Atoms.info.
        """
        return super().info_unit_keys + ("target_temperature",)

    def get_stats(self) -> dict[str, float]:
        """
        Get thermodynamical statistics to be written to file.

        Returns
        -------
        dict[str, float]
            Thermodynamical statistics to be written out.
        """
        stats = super().get_stats()
        stats |= {"Target_P": self.pressure, "Target_T": self.temp}
        return stats

    @property
    def stats_units(self) -> dict[str, str]:
        """
        Get units of returned statistics.

        Returns
        -------
        dict[str, str]
            Units attached to statistical properties.
        """
        return super().stats_units | {
            "Target_P": JANUS_UNITS["pressure"],
            "Target_T": JANUS_UNITS["temperature"],
        }

    @property
    def default_formats(self) -> dict[str, str]:
        """
        Default format of returned statistics.

        Returns
        -------
        dict[str, str]
            Default formats attached to statistical properties.
        """
        return super().default_formats | {"Target_P": ".5f", "Target_T": ".5f"}


class NVT(MolecularDynamics):
    """
    Configure NVT simulation.

    Parameters
    ----------
    *args
        Additional arguments.
    friction
        Friction coefficient in fs^-1. Default is 0.005.
    ensemble
        Name for thermodynamic ensemble. Default is "nvt".
    ensemble_kwargs
        Keyword arguments to pass to ensemble initialization. Default is {}.
    **kwargs
        Additional keyword arguments.
    """

    def __init__(
        self,
        *args,
        friction: float = 0.005,
        ensemble: Ensembles = "nvt",
        ensemble_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """
        Initialise dynamics for NVT simulation.

        Parameters
        ----------
        *args
            Additional arguments.
        friction
            Friction coefficient in fs^-1. Default is 0.005.
        ensemble
            Name for thermodynamic ensemble. Default is "nvt".
        ensemble_kwargs
            Keyword arguments to pass to ensemble initialization. Default is {}.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(*args, ensemble=ensemble, **kwargs)

        (ensemble_kwargs,) = none_to_dict(ensemble_kwargs)
        self.dyn = Langevin(
            self.struct,
            timestep=self.timestep,
            temperature_K=self.temp,
            friction=friction / units.fs,
            append_trajectory=self.traj_append,
            **ensemble_kwargs,
        )

    @property
    def info_unit_keys(self) -> tuple[str]:
        """
        Get Atoms.info keys to save units for.

        Returns
        -------
        tuple[str]
            Keys of Atoms.info.
        """
        return super().info_unit_keys + ("target_temperature",)

    def get_stats(self) -> dict[str, float]:
        """
        Get thermodynamical statistics to be written to file.

        Returns
        -------
        dict[str, float]
            Thermodynamical statistics to be written out.
        """
        stats = MolecularDynamics.get_stats(self)
        stats |= {"Target_T": self.temp}
        return stats

    @property
    def stats_units(self) -> dict[str, str]:
        """
        Get units of returned statistics.

        Returns
        -------
        dict[str, str]
            Units attached to statistical properties.
        """
        return super().stats_units | {"Target_T": JANUS_UNITS["temperature"]}

    @property
    def default_formats(self) -> dict[str, str]:
        """
        Default format of returned statistics.

        Returns
        -------
        dict[str, str]
            Default formats attached to statistical properties.
        """
        return super().default_formats | {"Target_T": ".5f"}


class NVE(MolecularDynamics):
    """
    Configure NVE simulation.

    Parameters
    ----------
    *args
        Additional arguments.
    ensemble
        Name for thermodynamic ensemble. Default is "nve".
    ensemble_kwargs
        Keyword arguments to pass to ensemble initialization. Default is {}.
    **kwargs
        Additional keyword arguments.
    """

    def __init__(
        self,
        *args,
        ensemble: Ensembles = "nve",
        ensemble_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """
        Initialise dynamics for NVE simulation.

        Parameters
        ----------
        *args
            Additional arguments.
        ensemble
            Name for thermodynamic ensemble. Default is "nve".
        ensemble_kwargs
            Keyword arguments to pass to ensemble initialization. Default is {}.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(*args, ensemble=ensemble, **kwargs)
        (ensemble_kwargs,) = none_to_dict(ensemble_kwargs)

        self.dyn = VelocityVerlet(
            self.struct,
            timestep=self.timestep,
            append_trajectory=self.traj_append,
            **ensemble_kwargs,
        )


class NVT_NH(MolecularDynamics):  # noqa: N801 (invalid-class-name)
    """
    Configure NVT Nosé-Hoover simulation.

    Parameters
    ----------
    *args
        Additional arguments.
    thermostat_time
        Thermostat time, in fs. Default is 50.0.
    ensemble
        Name for thermodynamic ensemble. Default is "nvt-nh".
    file_prefix
        Prefix for output filenames. Default is inferred from structure, ensemble,
        temperature, and pressure.
    ensemble_kwargs
        Keyword arguments to pass to ensemble initialization. Default is {}.
    **kwargs
        Additional keyword arguments.
    """

    def __init__(
        self,
        *args,
        thermostat_time: float = 50.0,
        ensemble: Ensembles = "nvt-nh",
        file_prefix: PathLike | None = None,
        ensemble_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """
        Initialise dynamics for NVT Nosé-Hoover simulation.

        Parameters
        ----------
        *args
            Additional arguments.
        thermostat_time
            Thermostat time, in fs. Default is 50.0.
        ensemble
            Name for thermodynamic ensemble. Default is "nvt-nh".
        file_prefix
            Prefix for output filenames. Default is inferred from structure, ensemble,
            temperature, and pressure.
        ensemble_kwargs
            Keyword arguments to pass to ensemble initialization. Default is {}.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(*args, ensemble=ensemble, file_prefix=file_prefix, **kwargs)

        (ensemble_kwargs,) = none_to_dict(ensemble_kwargs)
        self.ttime = thermostat_time * units.fs

        self.dyn = ASE_NPT(
            self.struct,
            timestep=self.timestep,
            temperature_K=self.temp,
            ttime=self.ttime,
            append_trajectory=self.traj_append,
            **ensemble_kwargs,
        )

    @property
    def info_unit_keys(self) -> tuple[str]:
        """
        Get Atoms.info keys to save units for.

        Returns
        -------
        tuple[str]
            Keys of Atoms.info.
        """
        return super().info_unit_keys + ("target_temperature",)

    def get_stats(self) -> dict[str, float]:
        """
        Get thermodynamical statistics to be written to file.

        Returns
        -------
        dict[str, float]
            Thermodynamical statistics to be written out.
        """
        stats = super().get_stats()
        stats |= {"Target_T": self.temp}
        return stats

    @property
    def stats_units(self) -> dict[str, str]:
        """
        Get units of returned statistics.

        Returns
        -------
        dict[str, str]
            Units attached to statistical properties.
        """
        return super().stats_units | {"Target_T": JANUS_UNITS["temperature"]}

    @property
    def default_formats(self) -> dict[str, str]:
        """
        Default format of returned statistics.

        Returns
        -------
        dict[str, str]
            Default formats attached to statistical properties.
        """
        return super().default_formats | {"Target_T": ".5f"}


class NVT_CSVR(NVT):  # noqa: N801 (invalid-class-name)
    """
    Configure NVT simulation using CSVR thermostat proposed by Bussi et al.

    Parameters
    ----------
    *args
        Additional arguments.
    taut
        Time constant for CSVR thermostat coupling, in fs. Default is 100.0.
    ensemble
        Name for thermodynamic ensemble. Default is "nvt-csvr".
    ensemble_kwargs
        Keyword arguments to pass to ensemble initialization. Default is {}.
    **kwargs
        Additional keyword arguments.
    """

    def __init__(
        self,
        *args,
        taut: float = 100.0,
        ensemble: Ensembles = "nvt-csvr",
        ensemble_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """
        Initialise dynamics for NVT simulation using CSVR thermostat.

        Parameters
        ----------
        *args
            Additional arguments.
        taut
            Time constant for CSVR thermostat coupling, in fs. Defaylt is 100.0.
        ensemble
            Name for thermodynamic ensemble. Default is "nvt-csvr".
        ensemble_kwargs
            Keyword arguments to pass to ensemble initialization. Default is {}.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(*args, ensemble=ensemble, **kwargs)

        (ensemble_kwargs,) = none_to_dict(ensemble_kwargs)

        # Velocity distribution must be non-zero before dynamics is set
        if np.isclose(self.struct.get_kinetic_energy(), 0.0, rtol=0, atol=1e-12):
            if self.logger:
                self.logger.warning("Velocities modified during MD initialisation")
            self._set_velocity_distribution()

        self.dyn = Bussi(
            self.struct,
            timestep=self.timestep,
            temperature_K=self.temp,
            taut=taut * units.fs,
            **ensemble_kwargs,
        )


class NPH(MolecularDynamics):
    """
    Configure NPH simulation.

    Parameters
    ----------
    *args
        Additional arguments.
    barostat_time
        Barostat time, in fs. Default is 75.0.
    bulk_modulus
        Bulk modulus, in GPa. Default is 2.0.
    pressure
        Pressure, in GPa. Default is 0.0.
    ensemble
        Name for thermodynamic ensemble. Default is "nph".
    file_prefix
        Prefix for output filenames. Default is inferred from structure, ensemble,
        temperature, and pressure.
    ensemble_kwargs
        Keyword arguments to pass to ensemble initialization. Default is {}.
    **kwargs
        Additional keyword arguments.
    """

    def __init__(
        self,
        *args,
        barostat_time: float = 75.0,
        bulk_modulus: float = 2.0,
        pressure: float = 0.0,
        ensemble: Ensembles = "nph",
        file_prefix: PathLike | None = None,
        ensemble_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """
        Initialise dynamics for NPH simulation.

        Parameters
        ----------
        *args
            Additional arguments.
        barostat_time
            Barostat time, in fs. Default is 75.0.
        bulk_modulus
            Bulk modulus, in GPa. Default is 2.0.
        pressure
            Pressure, in GPa. Default is 0.0.
        ensemble
            Name for thermodynamic ensemble. Default is "nph".
        file_prefix
            Prefix for output filenames. Default is inferred from structure, ensemble,
            temperature, and pressure.
        ensemble_kwargs
            Keyword arguments to pass to ensemble initialization. Default is {}.
        **kwargs
            Additional keyword arguments.
        """
        self.pressure = pressure
        super().__init__(*args, ensemble=ensemble, file_prefix=file_prefix, **kwargs)

        (ensemble_kwargs,) = none_to_dict(ensemble_kwargs)

        pfactor = barostat_time**2 * bulk_modulus
        if self.logger:
            self.logger.info("NPT pfactor=%s GPa fs^2", pfactor)

        # convert the pfactor to ASE internal units
        pfactor *= units.fs**2 * units.GPa

        self.dyn = ASE_NPT(
            self.struct,
            timestep=self.timestep,
            temperature_K=self.temp,
            ttime=None,
            pfactor=pfactor,
            append_trajectory=self.traj_append,
            externalstress=self.pressure * units.GPa,
            **ensemble_kwargs,
        )

    def _set_param_prefix(self, file_prefix: PathLike | None = None) -> str:
        """
        Set ensemble parameters for output files.

        Parameters
        ----------
        file_prefix
            Prefix for output filenames on class init. If not None, param_prefix = "".

        Returns
        -------
        str
           Formatted ensemble parameters, including pressure and temperature(s).
        """
        if file_prefix is not None:
            return ""

        return f"{super()._set_param_prefix(file_prefix)}-p{self.pressure}"

    def get_stats(self) -> dict[str, float]:
        """
        Get thermodynamical statistics to be written to file.

        Returns
        -------
        dict[str, float]
            Thermodynamical statistics to be written out.
        """
        stats = super().get_stats()
        stats |= {"Target_P": self.pressure}
        return stats

    @property
    def stats_units(self) -> dict[str, str]:
        """
        Get units of returned statistics.

        Returns
        -------
        dict[str, str]
            Units attached to statistical properties.
        """
        return super().stats_units | {
            "Target_P": JANUS_UNITS["pressure"],
        }

    @property
    def default_formats(self) -> dict[str, str]:
        """
        Default format of returned statistics.

        Returns
        -------
        dict[str, str]
            Default formats attached to statistical properties.
        """
        return super().default_formats | {"Target_P": ".5f"}


class NPT_MTK(MolecularDynamics):  # noqa: N801 (invalid-class-name)
    """
    Configure NPT simulation using isotropic (MTK) thermostat and barostat chains.

    Parameters
    ----------
    *args
        Additional arguments.
    pressure
        Pressure, in GPa. Default is 0.0.
    thermostat_time
        Thermostat time, in fs. Default is 100 times `timestep`.
    barostat_time
        Barostat time, in fs. Default is 1000 times `timestep`.
    thermostat_chain
        Length of thermostat chain. Default is 3.
    barostat_chain
        Length of barostat chain. Default is 3.
    thermostat_substeps
        The number of sub-steps in thermostat integration. Default is 1.
    barostat_substeps
        The number of sub-steps in barostat integration. Default is 1.
    ensemble
        Name for thermodynamic ensemble. Default is "npt-mtk".
    ensemble_kwargs
        Keyword arguments to pass to ensemble initialization. Default is {}.
    **kwargs
        Additional keyword arguments.
    """

    def __init__(
        self,
        *args,
        pressure: float = 0.0,
        thermostat_time: float = 100.0,
        barostat_time: float = 1000.0,
        thermostat_chain: int = 3,
        barostat_chain: int = 3,
        thermostat_substeps: int = 1,
        barostat_substeps: int = 1,
        ensemble: Ensembles = "npt-mtk",
        ensemble_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """
        Initialise dynamics for isotropic NPT simulation.

        Parameters
        ----------
        *args
            Additional arguments.
        pressure
            Pressure, in GPa. Default is 0.0.
        thermostat_time
            Thermostat time, in fs. Recommended value is 100 times `timestep`.
            Default is 100 fs.
        barostat_time
            Barostat time, in fs. Recommended value is 1000 times `timestep`.
            Default is 1000 fs.
        thermostat_chain
            Length of thermostat chain. Default is 3.
        barostat_chain
            Length of barostat chain. Default is 3.
        thermostat_substeps
            The number of sub-steps in thermostat integration. Default is 1.
        barostat_substeps
            The number of sub-steps in barostat integration. Default is 1.
        ensemble
            Name for thermodynamic ensemble. Default is "npt-mtk".
        ensemble_kwargs
            Keyword arguments to pass to ensemble initialization. Default is {}.
        **kwargs
            Additional keyword arguments.
        """
        try:
            from ase.md.nose_hoover_chain import (
                IsotropicMTKNPT as ASE_NPT_MTK,  # noqa: N814 (camelcase-imported-as-constant)
            )
        except ImportError as e:
            raise NotImplementedError(
                "Please download the latest ASE commits to use this module"
            ) from e

        self.pressure = pressure

        super().__init__(*args, ensemble=ensemble, **kwargs)

        (ensemble_kwargs,) = none_to_dict(ensemble_kwargs)

        self.dyn = ASE_NPT_MTK(
            self.struct,
            timestep=self.timestep,
            temperature_K=self.temp,
            pressure_au=self.pressure * units.GPa,
            tdamp=thermostat_time * units.fs,
            pdamp=barostat_time * units.fs,
            tchain=thermostat_chain,
            pchain=barostat_chain,
            tloop=thermostat_substeps,
            ploop=barostat_substeps,
            **ensemble_kwargs,
        )

    def _set_param_prefix(self, file_prefix: PathLike | None = None) -> str:
        """
        Set ensemble parameters for output files.

        Parameters
        ----------
        file_prefix
            Prefix for output filenames on class init. If not None, param_prefix = "".

        Returns
        -------
        str
           Formatted ensemble parameters, including pressure and temperature(s).
        """
        if file_prefix is not None:
            return ""

        pressure = f"-p{self.pressure}"
        return f"{super()._set_param_prefix(file_prefix)}{pressure}"

    @property
    def info_unit_keys(self) -> tuple[str]:
        """
        Get Atoms.info keys to save units for.

        Returns
        -------
        tuple[str]
            Keys of Atoms.info.
        """
        return super().info_unit_keys + ("target_temperature",)

    def get_stats(self) -> dict[str, float]:
        """
        Get thermodynamical statistics to be written to file.

        Returns
        -------
        dict[str, float]
            Thermodynamical statistics to be written out.
        """
        stats = MolecularDynamics.get_stats(self)
        stats |= {"Target_P": self.pressure, "Target_T": self.temp}
        return stats

    @property
    def stats_units(self) -> dict[str, str]:
        """
        Get units of returned statistics.

        Returns
        -------
        dict[str, str]
            Units attached to statistical properties.
        """
        return super().stats_units | {
            "Target_P": JANUS_UNITS["pressure"],
            "Target_T": JANUS_UNITS["temperature"],
        }

    @property
    def default_formats(self) -> dict[str, str]:
        """
        Default format of returned statistics.

        Returns
        -------
        dict[str, str]
            Default formats attached to statistical properties.
        """
        return super().default_formats | {"Target_P": ".5f", "Target_T": ".5f"}
