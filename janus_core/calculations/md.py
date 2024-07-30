# pylint: disable=too-many-lines
"""Run molecular dynamics simulations."""

import datetime
from functools import partial
from itertools import combinations_with_replacement
from math import isclose
from pathlib import Path
import random
from typing import Any, Optional, Union
from warnings import warn

from ase import Atoms, units
from ase.geometry.analysis import Analysis
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.npt import NPT as ASE_NPT
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet
import numpy as np
import yaml

from janus_core.calculations.geom_opt import GeomOpt
from janus_core.helpers.correlator import Correlation
from janus_core.helpers.janus_types import (
    CorrelationKwargs,
    Ensembles,
    OutputKwargs,
    PathLike,
    PostProcessKwargs,
)
from janus_core.helpers.log import config_logger, config_tracker
from janus_core.helpers.post_process import compute_rdf, compute_vaf
from janus_core.helpers.utils import FileNameMixin, output_structs

DENS_FACT = (units.m / 1.0e2) ** 3 / units.mol


class MolecularDynamics(FileNameMixin):  # pylint: disable=too-many-instance-attributes
    """
    Configure shared molecular dynamics simulation options.

    Parameters
    ----------
    struct : Atoms
        Structure to simulate.
    struct_name : str
        Name of structure to simulate. Default is inferred from filepath or
        chemical formula.
    ensemble : Ensembles
        Name for thermodynamic ensemble. Default is None.
    steps : int
        Number of steps in simulation. Default is 0.
    timestep : float
        Timestep for integrator, in fs. Default is 1.0.
    temp : float
        Temperature, in K. Default is 300.
    equil_steps : int
        Maximum number of steps at which to perform optimization and reset velocities.
        Default is 0.
    minimize : bool
        Whether to minimize structure during equilibration. Default is False.
    minimize_every : int
        Frequency of minimizations. Default is -1, which disables minimization after
        beginning dynamics.
    minimize_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to geometry optimizer. Default is {}.
    rescale_velocities : bool
        Whether to rescale velocities. Default is False.
    remove_rot : bool
        Whether to remove rotation. Default is False.
    rescale_every : int
        Frequency to rescale velocities. Default is 10.
    file_prefix : Optional[PathLike]
        Prefix for output filenames. Default is inferred from structure, ensemble,
        and temperature.
    restart : bool
        Whether restarting dynamics. Default is False.
    restart_stem : str
        Stem for restart file name. Default inferred from `file_prefix`.
    restart_every : int
        Frequency of steps to save restart info. Default is 1000.
    rotate_restart : bool
        Whether to rotate restart files. Default is False.
    restarts_to_keep : int
        Restart files to keep if rotating. Default is 4.
    final_file : Optional[PathLike]
        File to save final configuration at each temperature of similation. Default
        inferred from `file_prefix`.
    stats_file : Optional[PathLike]
        File to save thermodynamical statistics. Default inferred from `file_prefix`.
    stats_every : int
        Frequency to output statistics. Default is 100.
    traj_file : Optional[PathLike]
        Trajectory file to save. Default inferred from `file_prefix`.
    traj_append : bool
        Whether to append trajectory. Default is False.
    traj_start : int
        Step to start saving trajectory. Default is 0.
    traj_every : int
        Frequency of steps to save trajectory. Default is 100.
    temp_start : Optional[float]
        Temperature to start heating, in K. Default is None, which disables heating.
    temp_end : Optional[float]
        Maximum temperature for heating, in K. Default is None, which disables heating.
    temp_step : Optional[float]
        Size of temperature steps when heating, in K. Default is None, which disables
        heating.
    temp_time : Optional[float]
        Time between heating steps, in fs. Default is None, which disables heating.
    write_kwargs : Optional[OutputKwargs],
        Keyword arguments to pass to `output_structs` when saving trajectory and final
        files. Default is {}.
    post_process_kwargs : Optional[PostProcessKwargs]
        Keyword arguments to control post-processing operations.
    correlation_kwargs : Optional[CorrelationKwargs]
        Keyword arguments to control on-the-fly correlations.
    log_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to `config_logger`. Default is None.
    tracker_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to `config_tracker`. Default is None.
    seed : Optional[int]
        Random seed used by numpy.random and random functions, such as in Langevin.
        Default is None.

    Attributes
    ----------
    logger : Optional[logging.Logger]
        Logger if log file has been specified.
    tracker : Optional[OfflineEmissionsTracker]
        Tracker if logging is enabled.
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

    Methods
    -------
    run()
        Run molecular dynamics simulation and/or heating ramp.
    get_stats()
        Get thermodynamical statistics to be written to file.
    get_stats_header()
        Get header string for molecular dynamics statistics.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
        self,
        struct: Atoms,
        struct_name: Optional[str] = None,
        ensemble: Optional[Ensembles] = None,
        steps: int = 0,
        timestep: float = 1.0,
        temp: float = 300,
        equil_steps: int = 0,
        minimize: bool = False,
        minimize_every: int = -1,
        minimize_kwargs: Optional[dict[str, Any]] = None,
        rescale_velocities: bool = False,
        remove_rot: bool = False,
        rescale_every: int = 10,
        file_prefix: Optional[PathLike] = None,
        restart: bool = False,
        restart_stem: Optional[PathLike] = None,
        restart_every: int = 1000,
        rotate_restart: bool = False,
        restarts_to_keep: int = 4,
        final_file: Optional[PathLike] = None,
        stats_file: Optional[PathLike] = None,
        stats_every: int = 100,
        traj_file: Optional[PathLike] = None,
        traj_append: bool = False,
        traj_start: int = 0,
        traj_every: int = 100,
        temp_start: Optional[float] = None,
        temp_end: Optional[float] = None,
        temp_step: Optional[float] = None,
        temp_time: Optional[float] = None,
        write_kwargs: Optional[OutputKwargs] = None,
        post_process_kwargs: Optional[PostProcessKwargs] = None,
        correlation_kwargs: Optional[list[CorrelationKwargs]] = None,
        log_kwargs: Optional[dict[str, Any]] = None,
        tracker_kwargs: Optional[dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialise molecular dynamics simulation configuration.

        Parameters
        ----------
        struct : Atoms
            Structure to simulate.
        struct_name : Optional[str]
            Name of structure to simulate. Default is inferred from filepath or
            chemical formula.
        ensemble : Ensembles
            Name for thermodynamic ensemble. Default is None.
        steps : int
            Number of steps in simulation. Default is 0.
        timestep : float
            Timestep for integrator, in fs. Default is 1.0.
        temp : float
            Temperature, in K. Default is 300.
        equil_steps : int
            Maximum number of steps at which to perform optimization and reset
            velocities. Default is 0.
        minimize : bool
            Whether to minimize structure during equilibration. Default is False.
        minimize_every : int
            Frequency of minimizations. Default is -1, which disables minimization
            after beginning dynamics.
        minimize_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to geometry optimizer. Default is {}.
        rescale_velocities : bool
            Whether to rescale velocities. Default is False.
        remove_rot : bool
            Whether to remove rotation. Default is False.
        rescale_every : int
            Frequency to rescale velocities. Default is 10.
        file_prefix : Optional[PathLike]
            Prefix for output filenames. Default is inferred from structure, ensemble,
            and temperature.
        restart : bool
            Whether restarting dynamics. Default is False.
        restart_stem : str
            Stem for restart file name. Default inferred from `file_prefix`.
        restart_every : int
            Frequency of steps to save restart info. Default is 1000.
        rotate_restart : bool
            Whether to rotate restart files. Default is False.
        restarts_to_keep : int
            Restart files to keep if rotating. Default is 4.
        final_file : Optional[PathLike]
            File to save final configuration at each temperature of similation. Default
            inferred from `file_prefix`.
        stats_file : Optional[PathLike]
            File to save thermodynamical statistics. Default inferred from
            `file_prefix`.
        stats_every : int
            Frequency to output statistics. Default is 100.
        traj_file : Optional[PathLike]
            Trajectory file to save. Default inferred from `file_prefix`.
        traj_append : bool
            Whether to append trajectory. Default is False.
        traj_start : int
            Step to start saving trajectory. Default is 0.
        traj_every : int
            Frequency of steps to save trajectory. Default is 100.
        temp_start : Optional[float]
            Temperature to start heating, in K. Default is None, which disables
            heating.
        temp_end : Optional[float]
            Maximum temperature for heating, in K. Default is None, which disables
            heating.
        temp_step : Optional[float]
            Size of temperature steps when heating, in K. Default is None, which
            disables heating.
        temp_time : Optional[float]
            Time between heating steps, in fs. Default is None, which disables heating.
        write_kwargs : Optional[OutputKwargs],
            Keyword arguments to pass to `output_structs` when saving trajectory and
            final files. Default is {}.
        post_process_kwargs : Optional[PostProcessKwargs]
            Keyword arguments to control post-processing operations.
        correlation_kwargs : Optional[list[CorrelationKwargs]]
            Keyword arguments to control on-the-fly correlations.
        log_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to `config_logger`. Default is None.
        tracker_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to `config_tracker`. Default is None.
        seed : Optional[int]
            Random seed used by numpy.random and random functions, such as in Langevin.
            Default is None.
        """
        self.struct = struct
        self.timestep = timestep * units.fs
        self.steps = steps
        self.temp = temp
        self.equil_steps = equil_steps
        self.minimize = minimize
        self.minimize_every = minimize_every
        self.rescale_velocities = rescale_velocities
        self.remove_rot = remove_rot
        self.rescale_every = rescale_every
        self.restart = restart
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
        self.write_kwargs = write_kwargs if write_kwargs is not None else {}
        self.post_process_kwargs = (
            post_process_kwargs if post_process_kwargs is not None else {}
        )
        self.correlation_kwargs = (
            correlation_kwargs if correlation_kwargs is not None else {}
        )
        self.log_kwargs = log_kwargs
        self.ensemble = ensemble
        self.seed = seed

        FileNameMixin.__init__(self, struct, struct_name, file_prefix, ensemble)

        self.write_kwargs.setdefault(
            "columns", ["symbols", "positions", "momenta", "masses"]
        )
        if "append" in self.write_kwargs:
            raise ValueError("`append` cannot be specified when writing files")

        self.log_kwargs = (
            log_kwargs if log_kwargs else {}
        )  # pylint: disable=duplicate-code
        self.tracker_kwargs = (
            tracker_kwargs if tracker_kwargs else {}
        )  # pylint: disable=duplicate-code
        if self.log_kwargs and "filename" not in self.log_kwargs:
            raise ValueError("'filename' must be included in `log_kwargs`")

        self.log_kwargs.setdefault("name", __name__)
        self.logger = config_logger(**self.log_kwargs)
        self.tracker = config_tracker(self.logger, **self.tracker_kwargs)

        # Warn if attempting to rescale/minimize during dynamics
        # but equil_steps is too low
        if rescale_velocities and equil_steps < rescale_every:
            warn("Velocities and angular momentum will not be reset during dynamics")
        if minimize and equil_steps < minimize_every:
            warn("Geometry will not be minimized during dynamics")

        # Warn if attempting to remove rotation without resetting velocities
        if remove_rot and not rescale_velocities:
            warn("Rotation will not be removed unless `rescale_velocities` is True")

        # Check temperatures for heating differ
        if self.temp_start is not None and self.temp_start == self.temp_end:
            raise ValueError("Start and end temperatures must be different")

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
                "heating to run"
            )

        if self.ramp_temp and (self.temp_start < 0 or self.temp_end < 0):
            raise ValueError("Start and end temperatures must be positive")

        self.minimize_kwargs = minimize_kwargs if minimize_kwargs else {}
        self.restart_files = []
        self.dyn: Union[Langevin, VelocityVerlet, ASE_NPT]
        self.n_atoms = len(self.struct)

        self.final_file = self._build_filename(
            "final.extxyz",
            self._parameter_prefix if file_prefix is None else "",
            filename=self.final_file,
        )
        self.stats_file = self._build_filename(
            "stats.dat",
            self._parameter_prefix if file_prefix is None else "",
            filename=self.stats_file,
        )
        self.traj_file = self._build_filename(
            "traj.extxyz",
            self._parameter_prefix if file_prefix is None else "",
            filename=self.traj_file,
        )

        self.offset = 0
        self.created_final_file = False

        if "masses" not in self.struct.arrays.keys():
            self.struct.set_masses()

        if self.seed:
            np.random.seed(seed)
            random.seed(seed)

        self._parse_correlations()

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
        if self.dyn.nsteps >= 0:
            atoms = self.dyn.atoms

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
                self.minimize_kwargs["log_kwargs"] = {
                    "filename": self.log_kwargs["filename"],
                    "name": self.logger.name,
                    "filemode": "a",
                }
                self.logger.info("Minimizing at step %s", self.dyn.nsteps)
            optimizer = GeomOpt(self.struct, **self.minimize_kwargs)
            optimizer.run()

    @property
    def _parameter_prefix(self) -> str:
        """
        Ensemble parameters for output files.

        Returns
        -------
        str
           Formatted temperature range if heating and target temp if running md.
        """

        temperature_prefix = ""
        if self.temp_start is not None and self.temp_end is not None:
            temperature_prefix += f"-T{self.temp_start}-T{self.temp_end}"

        if self.steps > 0:
            temperature_prefix += f"-T{self.temp}"

        return temperature_prefix.lstrip("-")

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
            f"res-{step}.extxyz", f"T{self.temp}", prefix_override=self.restart_stem
        )

    def _parse_correlations(self):
        """Parse correlation kwargs into Correlations."""
        if self.correlation_kwargs:
            self._correlations = [Correlation(**cor) for cor in self.correlation_kwargs]
        else:
            self._correlations = ()

    def _attach_correlations(self):
        """Attach all correlations to self.dyn."""
        for i, _ in enumerate(self._correlations):
            self.dyn.attach(
                partial(lambda i: self._correlations[i].update(self.dyn.atoms), i),
                self._correlations[i].update_frequency,
            )

    def _write_correlations(self):
        """Write out the correlations."""
        if self._correlations:
            param_pref = self._parameter_prefix if self.file_prefix is None else ""
            with open(
                self._build_filename("cor.dat", param_pref), "w", encoding="utf-8"
            ) as out_file:
                data = {}
                for cor in self._correlations:
                    value, lags = cor.get()
                    data[str(cor)] = {"value": value.tolist(), "lags": lags.tolist()}
                yaml.dump(data, out_file, default_flow_style=None)

    @staticmethod
    def get_stats_header() -> str:
        """
        Get header string for molecular dynamics statistics.

        Returns
        -------
        str
            Header for molecular dynamics statistics.
        """
        stats_header = (
            "# Step | Real Time [s] | Time [fs] | Epot/N [eV] | Ekin/N [eV] | "
            "T [K] | Etot/N [eV] | Density [g/cm^3] | Volume [A^3] | P [GPa] | "
            "Pxx [GPa] | Pyy [GPa] | Pzz [GPa] | Pyz [GPa] | Pxz [GPa] | Pxy [GPa]"
        )

        return stats_header

    def get_stats(self) -> str:
        """
        Get thermodynamical statistics to be written to file.

        Returns
        -------
        str
            Thermodynamical statistics to be written out.
        """
        e_pot = self.dyn.atoms.get_potential_energy() / self.n_atoms
        e_kin = self.dyn.atoms.get_kinetic_energy() / self.n_atoms
        current_temp = e_kin / (1.5 * units.kB)

        time = (self.offset * self.timestep + self.dyn.get_time()) / units.fs
        step = self.offset + self.dyn.nsteps
        self.dyn.atoms.info["time_fs"] = time
        self.dyn.atoms.info["step"] = step

        time_now = datetime.datetime.now()
        real_time = time_now - self.dyn.atoms.info["real_time"]
        self.dyn.atoms.info["real_time"] = time_now

        try:
            density = (
                np.sum(self.dyn.atoms.get_masses())
                / self.dyn.atoms.get_volume()
                * DENS_FACT
            )
            self.dyn.atoms.info["density"] = density
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
        except ValueError:
            volume = 0.0
            pressure = 0.0
            density = 0.0
            pressure_tensor = np.zeros(6)

        stats = (
            f"{step:10d} {real_time.total_seconds():.3f} {time:13.2f} {e_pot:.8e} "
            f"{e_kin:.8e} {current_temp:.3f} {e_pot + e_kin:.8e} {density:.3f} "
            f"{volume:.8e} {pressure:.8e} {pressure_tensor[0]:.8e} "
            f"{pressure_tensor[1]:.8e} {pressure_tensor[2]:.8e} "
            f"{pressure_tensor[3]:.8e} {pressure_tensor[4]:.8e} "
            f"{pressure_tensor[5]:.8e}"
        )

        return stats

    def _write_stats_file(self) -> None:
        """Write molecular dynamics statistics."""
        stats = self.get_stats()

        # we do not want to print step 0 in restarts
        if self.restart and self.dyn.nsteps == 0:
            return

        with open(self.stats_file, "a", encoding="utf8") as stats_file:
            print(stats, file=stats_file)

    def _write_traj(self) -> None:
        """Write current structure to trajectory file."""
        if self.dyn.nsteps >= self.traj_start:
            # Append if restarting or already started writing
            append = self.restart or (
                self.dyn.nsteps > self.traj_start + self.traj_start % self.traj_every
            )

            write_kwargs = self.write_kwargs
            write_kwargs["filename"] = self.traj_file
            write_kwargs["append"] = append
            output_structs(
                images=self.struct,
                write_results=True,
                write_kwargs=write_kwargs,
            )

    def _write_final_state(self) -> None:
        """Write the final system state."""
        self.struct.info["temperature"] = self.temp
        if isinstance(self, NPT) and not isinstance(self, NVT_NH):
            self.struct.info["pressure"] = self.pressure

        # Append if final file has been created
        append = self.created_final_file

        write_kwargs = self.write_kwargs
        write_kwargs["filename"] = self.final_file
        write_kwargs["append"] = append
        output_structs(
            images=self.struct,
            write_results=True,
            write_kwargs=write_kwargs,
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
                "to do post-processing."
            )

        data = read(self.traj_file, index=":")

        ana = Analysis(data)

        param_pref = self._parameter_prefix if self.file_prefix is None else ""

        if self.post_process_kwargs.get("rdf_compute", False):
            base_name = self.post_process_kwargs.get("rdf_output_file", None)
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

            if rdf_args["by_elements"]:
                elements = (
                    tuple(sorted(set(data[0].get_chemical_symbols())))
                    if rdf_args["elements"] is None
                    else rdf_args["elements"]
                )

                out_paths = [
                    self._build_filename(
                        "rdf.dat",
                        param_pref,
                        "_".join(element),
                        prefix_override=base_name,
                    )
                    for element in combinations_with_replacement(elements, 2)
                ]

            else:
                out_paths = [
                    self._build_filename(
                        "rdf.dat", param_pref, prefix_override=base_name
                    )
                ]

            compute_rdf(data, ana, filename=out_paths, **rdf_args)

        if self.post_process_kwargs.get("vaf_compute", False):

            file_name = self.post_process_kwargs.get("vaf_output_file", None)
            use_vel = self.post_process_kwargs.get("vaf_velocities", False)
            fft = self.post_process_kwargs.get("vaf_fft", False)

            out_path = self._build_filename("vaf.dat", param_pref, filename=file_name)
            slice_ = (
                self.post_process_kwargs.get("vaf_start", 0),
                self.post_process_kwargs.get("vaf_stop", None),
                self.post_process_kwargs.get("vaf_step", 1),
            )

            compute_vaf(
                data,
                out_path,
                use_velocities=use_vel,
                fft=fft,
                index=slice_,
                filter_atoms=self.post_process_kwargs.get("vaf_atoms", None),
            )

    def _write_restart(self) -> None:
        """Write restart file and (optionally) rotate files saved."""
        step = self.offset + self.dyn.nsteps
        if step > 0:
            write_kwargs = self.write_kwargs
            write_kwargs["filename"] = self._restart_file
            output_structs(
                images=self.struct,
                write_results=True,
                write_kwargs=write_kwargs,
            )
            if self.rotate_restart:
                self.restart_files.append(self._restart_file)
                self._rotate_restart_files()

    def run(self) -> None:
        """Run molecular dynamics simulation and/or temperature ramp."""
        # Check if restarting simulation
        if self.restart:
            try:
                with open(self.stats_file, encoding="utf8") as stats_file:
                    last_line = stats_file.readlines()[-1].split()
                try:
                    self.offset = int(last_line[0])
                except (IndexError, ValueError) as e:
                    raise ValueError("Unable to read restart file") from e
            except FileNotFoundError as e:
                raise FileNotFoundError("Unable to read restart file") from e

        else:
            if self.minimize:
                self._optimize_structure()
            if self.rescale_velocities:
                self._reset_velocities()

            stats_header = self.get_stats_header()
            with open(self.stats_file, "w", encoding="utf8") as stats_file:
                print(stats_header, file=stats_file)

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
        self._set_velocity_distribution()

        # Run temperature ramp
        if self.ramp_temp:
            heating_steps = int(self.temp_time // self.timestep)

            # Always include start temperature in ramp, and include end temperature
            # if separated by an integer number of temperature steps
            n_temps = int(1 + abs(self.temp_end - self.temp_start) // self.temp_step)

            # Add or subtract temperatures
            ramp_sign = 1 if (self.temp_end - self.temp_start) > 0 else -1
            temps = [
                self.temp_start + ramp_sign * i * self.temp_step for i in range(n_temps)
            ]

            if self.logger:
                self.logger.info("Beginning temperature ramp at %sK", temps[0])
                self.tracker.start_task("Temperature ramp")

            for temp in temps:
                self.temp = temp
                self._set_velocity_distribution()
                if isclose(temp, 0.0):
                    self._write_final_state()
                    self.created_final_file = True
                    continue
                if not isinstance(self, NVE):
                    self.dyn.set_temperature(temperature_K=self.temp)
                self.dyn.run(heating_steps)
                self._write_final_state()
                self.created_final_file = True

            if self.logger:
                self.logger.info("Temperature ramp complete at %sK", temps[-1])
                self.tracker.stop_task()

        # Run MD
        if self.steps > 0:
            if self.logger:
                self.logger.info("Starting molecular dynamics simulation")
                self.tracker.start_task("Molecular dynamics")
            self.temp = md_temp
            if self.ramp_temp:
                self._set_velocity_distribution()
                if not isinstance(self, NVE):
                    self.dyn.set_temperature(temperature_K=self.temp)
            self.dyn.run(self.steps)
            self._write_final_state()
            self.created_final_file = True
            if self.logger:
                self.tracker.stop_task()
                self.tracker.stop()
                self.logger.info("Molecular dynamics simulation complete")


class NPT(MolecularDynamics):
    """
    Configure NPT dynamics.

    Parameters
    ----------
    *args
        Additional arguments.
    thermostat_time : float
        Thermostat time, in fs. Default is 50.0.
    barostat_time : float
        Barostat time, in fs. Default is 75.0.
    bulk_modulus : float
        Bulk modulus, in GPa. Default is 2.0.
    pressure : float
        Pressure, in GPa. Default is 0.0.
    ensemble : Ensembles
        Name for thermodynamic ensemble. Default is "npt".
    file_prefix : Optional[PathLike]
        Prefix for output filenames. Default is inferred from structure, ensemble,
        temperature, and pressure.
    ensemble_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ensemble initialization. Default is {}.
    **kwargs
        Additional keyword arguments.

    Attributes
    ----------
    dyn : Dynamics
        Configured NPT dynamics.
    """

    def __init__(
        self,
        *args,
        thermostat_time: float = 50.0,
        barostat_time: float = 75.0,
        bulk_modulus: float = 2.0,
        pressure: float = 0.0,
        ensemble: Ensembles = "npt",
        file_prefix: Optional[PathLike] = None,
        ensemble_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialise dynamics for NPT simulation.

        Parameters
        ----------
        *args
            Additional arguments.
        thermostat_time : float
            Thermostat time, in fs. Default is 50.0.
        barostat_time : float
            Barostat time, in fs. Default is 75.0.
        bulk_modulus : float
            Bulk modulus, in GPa. Default is 2.0.
        pressure : float
            Pressure, in GPa. Default is 0.0.
        ensemble : Ensembles
            Name for thermodynamic ensemble. Default is "npt".
        file_prefix : Optional[PathLike]
            Prefix for output filenames. Default is inferred from structure, ensemble,
            temperature, and pressure.
        ensemble_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to ensemble initialization. Default is {}.
        **kwargs
            Additional keyword arguments.
        """
        self.pressure = pressure
        super().__init__(ensemble=ensemble, file_prefix=file_prefix, *args, **kwargs)

        ensemble_kwargs = ensemble_kwargs if ensemble_kwargs else {}
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

    @property
    def _parameter_prefix(self) -> str:
        """
        Ensemble parameters for output files.

        Returns
        -------
        str
           Formatted temperature range if heating and target temp if running md.
        """

        pressure = f"-p{self.pressure}" if not isinstance(self, NVT_NH) else ""
        return f"{super()._parameter_prefix}{pressure}"

    @property
    def _restart_file(self) -> str:
        """
        Restart file name.

        Returns
        -------
        str
           File name for restart file, includes pressure.
        """
        step = self.offset + self.dyn.nsteps
        pressure = f"p{self.pressure}" if not isinstance(self, NVT_NH) else ""
        return self._build_filename(
            f"res-{step}.extxyz",
            f"T{self.temp}",
            pressure,
            prefix_override=self.restart_stem,
        )

    def get_stats(self) -> str:
        """
        Get thermodynamical statistics to be written to file.

        Returns
        -------
        str
            Thermodynamical statistics to be written out.
        """
        stats = MolecularDynamics.get_stats(self)
        return stats + f" {self.pressure} {self.temp}"

    @staticmethod
    def get_stats_header() -> str:
        """
        Get header string for molecular dynamics statistics.

        Returns
        -------
        str
            Header for molecular dynamics statistics.
        """
        stats_header = MolecularDynamics.get_stats_header()
        return stats_header + " | Target P [GPa] | Target T [K]"


class NVT(MolecularDynamics):
    """
    Configure NVT simulation.

    Parameters
    ----------
    *args
        Additional arguments.
    friction : float
        Friction coefficient in fs^-1. Default is 0.005.
    ensemble : Ensembles
        Name for thermodynamic ensemble. Default is "nvt".
    ensemble_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ensemble initialization. Default is {}.
    **kwargs
        Additional keyword arguments.

    Attributes
    ----------
    dyn : Dynamics
        Configured NVT dynamics.
    """

    def __init__(
        self,
        *args,
        friction: float = 0.005,
        ensemble: Ensembles = "nvt",
        ensemble_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialise dynamics for NVT simulation.

        Parameters
        ----------
        *args
            Additional arguments.
        friction : float
            Friction coefficient in fs^-1. Default is 0.005.
        ensemble : Ensembles
            Name for thermodynamic ensemble. Default is "nvt".
        ensemble_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to ensemble initialization. Default is {}.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(ensemble=ensemble, *args, **kwargs)

        ensemble_kwargs = ensemble_kwargs if ensemble_kwargs else {}
        self.dyn = Langevin(
            self.struct,
            timestep=self.timestep,
            temperature_K=self.temp,
            friction=friction / units.fs,
            append_trajectory=self.traj_append,
            **ensemble_kwargs,
        )

    def get_stats(self) -> str:
        """
        Get thermodynamical statistics to be written to file.

        Returns
        -------
        str
            Thermodynamical statistics to be written out.
        """
        stats = MolecularDynamics.get_stats(self)
        return stats + f" {self.temp}"

    @staticmethod
    def get_stats_header() -> str:
        """
        Get header string for molecular dynamics statistics.

        Returns
        -------
        str
            Header for molecular dynamics statistics.
        """
        stats_header = MolecularDynamics.get_stats_header()
        return stats_header + " | Target T [K]"


class NVE(MolecularDynamics):
    """
    Configure NVE simulation.

    Parameters
    ----------
    *args
        Additional arguments.
    ensemble : Ensembles
        Name for thermodynamic ensemble. Default is "nve".
    ensemble_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ensemble initialization. Default is {}.
    **kwargs
        Additional keyword arguments.

    Attributes
    ----------
    dyn : Dynamics
        Configured NVE dynamics.
    """

    def __init__(
        self,
        *args,
        ensemble: Ensembles = "nve",
        ensemble_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialise dynamics for NVE simulation.

        Parameters
        ----------
        *args
            Additional arguments.
        ensemble : Ensembles
            Name for thermodynamic ensemble. Default is "nve".
        ensemble_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to ensemble initialization. Default is {}.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(ensemble=ensemble, *args, **kwargs)
        ensemble_kwargs = ensemble_kwargs if ensemble_kwargs else {}

        self.dyn = VelocityVerlet(
            self.struct,
            timestep=self.timestep,
            append_trajectory=self.traj_append,
            **ensemble_kwargs,
        )


class NVT_NH(NPT):  # pylint: disable=invalid-name
    """
    Configure NVT Nosé-Hoover simulation.

    Parameters
    ----------
    *args
        Additional arguments.
    thermostat_time : float
        Thermostat time, in fs. Default is 50.0.
    ensemble : Ensembles
        Name for thermodynamic ensemble. Default is "nvt-nh".
    ensemble_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ensemble initialization. Default is {}.
    **kwargs
        Additional keyword arguments.
    """

    def __init__(
        self,
        *args,
        thermostat_time: float = 50.0,
        ensemble: Ensembles = "nvt-nh",
        ensemble_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialise dynamics for NVT simulation.

        Parameters
        ----------
        *args
            Additional arguments.
        thermostat_time : float
            Thermostat time, in fs. Default is 50.0.
        ensemble : Ensembles
            Name for thermodynamic ensemble. Default is "nvt-nh".
        ensemble_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to ensemble initialization. Default is {}.
        **kwargs
            Additional keyword arguments.
        """
        ensemble_kwargs = ensemble_kwargs if ensemble_kwargs else {}
        super().__init__(
            ensemble=ensemble,
            thermostat_time=thermostat_time,
            barostat_time=None,
            ensemble_kwargs=ensemble_kwargs,
            *args,
            **kwargs,
        )

    def get_stats(self) -> str:
        """
        Get thermodynamical statistics to be written to file.

        Returns
        -------
        str
            Thermodynamical statistics to be written out.
        """
        stats = MolecularDynamics.get_stats(self)
        return stats + f" {self.temp}"

    @staticmethod
    def get_stats_header() -> str:
        """
        Get header string for molecular dynamics statistics.

        Returns
        -------
        str
            Header for molecular dynamics statistics.
        """
        stats_header = MolecularDynamics.get_stats_header()
        return stats_header + " | Target T [K]"


class NPH(NPT):
    """
    Configure NPH simulation.

    Parameters
    ----------
    *args
        Additional arguments.
    thermostat_time : float
        Thermostat time, in fs. Default is 50.0.
    bulk_modulus : float
        Bulk modulus, in GPa. Default is 2.0.
    pressure : float
        Pressure, in GPa. Default is 0.0.
    ensemble : Ensembles
        Name for thermodynamic ensemble. Default is "nph".
    file_prefix : Optional[PathLike]
        Prefix for output filenames. Default is inferred from structure, ensemble,
        temperature, and pressure.
    ensemble_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to ensemble initialization. Default is {}.
    **kwargs
        Additional keyword arguments.

    Attributes
    ----------
    dyn : Dynamics
        Configured NVE dynamics.
    """

    def __init__(
        self,
        *args,
        thermostat_time: float = 50.0,
        bulk_modulus: float = 2.0,
        pressure: float = 0.0,
        ensemble: Ensembles = "nph",
        file_prefix: Optional[PathLike] = None,
        ensemble_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialise dynamics for NPH simulation.

        Parameters
        ----------
        *args
            Additional arguments.
        thermostat_time : float
            Thermostat time, in fs. Default is 50.0.
        bulk_modulus : float
            Bulk modulus, in GPa. Default is 2.0.
        pressure : float
            Pressure, in GPa. Default is 0.0.
        ensemble : Ensembles
            Name for thermodynamic ensemble. Default is "nph".
        file_prefix : Optional[PathLike]
            Prefix for output filenames. Default is inferred from structure, ensemble,
            temperature, and pressure.
        ensemble_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to ensemble initialization. Default is {}.
        **kwargs
            Additional keyword arguments.
        """
        ensemble_kwargs = ensemble_kwargs if ensemble_kwargs else {}
        super().__init__(
            *args,
            thermostat_time=thermostat_time,
            barostat_time=None,
            bulk_modulus=bulk_modulus,
            pressure=pressure,
            ensemble=ensemble,
            file_prefix=file_prefix,
            ensemble_kwargs=ensemble_kwargs,
            **kwargs,
        )
