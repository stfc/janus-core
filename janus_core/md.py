"""Run molecular dynamics simulations."""

import datetime
from pathlib import Path
import random
from typing import Any, Optional
from warnings import warn

from ase import Atoms, units
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.npt import NPT as ASE_NPT
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet
import numpy as np

from janus_core.geom_opt import optimize
from janus_core.janus_types import Ensembles, PathLike
from janus_core.log import config_logger

DENS_FACT = (units.m / 1.0e2) ** 3 / units.mol


class MolecularDynamics:  # pylint: disable=too-many-instance-attributes
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
    log_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to log config. Default is None.
    seed : Optional[int]
        Random seed used by numpy.random and random functions, such as in Langevin.
        Default is None.

    Attributes
    ----------
    logger : logging.Logger
        Logger if log file has been specified.
    dyn : Dynamics
        Dynamics object to run simulation.
    n_atoms : int
        Number of atoms in structure being simulated.
    restart_files : list[PathLike]
        List of files saved to restart dynamics.
    offset : int
        Number of previous steps if restarting simulation.

    Methods
    -------
    optimize_structure()
        Perform geometry optimization.
    reset_velocities()
        Reset velocities and (optionally) rotation of system.
    rotate_restart_files()
        Rotate restart files.
    run()
        Run molecular dynamics simulation.
    write_md_log()
        Write molecular dynamics log.
    write_traj()
        Write current structure to trajectory file.
    write_restart()
        Write restart file and (optionally) rotate files saved.
    get_log_stats()
        Get thermodynamical statistics to be written to molecular dynamics log.
    get_log_header()
        Get header string for molecular dynamics log.
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
        stats_file: Optional[PathLike] = None,
        stats_every: int = 100,
        traj_file: Optional[PathLike] = None,
        traj_append: bool = False,
        traj_start: int = 0,
        traj_every: int = 100,
        log_kwargs: Optional[dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialise molecular dynamics simulation configuration.

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
        log_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to log config. Default is None.
        seed : Optional[int]
            Random seed used by numpy.random and random functions, such as in Langevin.
            Default is None.
        """
        self.struct = struct
        self.struct_name = struct_name
        self.timestep = timestep * units.fs
        self.steps = steps
        self.temp = temp
        self.equil_steps = equil_steps
        self.minimize = minimize
        self.minimize_every = minimize_every
        self.rescale_velocities = rescale_velocities
        self.remove_rot = remove_rot
        self.rescale_every = rescale_every
        self.file_prefix = file_prefix
        self.restart = restart
        self.restart_stem = restart_stem
        self.restart_every = restart_every
        self.rotate_restart = rotate_restart
        self.restarts_to_keep = restarts_to_keep
        self.stats_file = stats_file
        self.stats_every = stats_every
        self.traj_file = traj_file
        self.traj_append = traj_append
        self.traj_start = traj_start
        self.traj_every = traj_every
        self.log_kwargs = log_kwargs
        self.ensemble = ensemble
        self.seed = seed

        self.log_kwargs = (
            log_kwargs if log_kwargs else {}
        )  # pylint: disable=duplicate-code
        if self.log_kwargs and "filename" not in self.log_kwargs:
            raise ValueError("'filename' must be included in `log_kwargs`")

        self.log_kwargs.setdefault("name", __name__)
        self.logger = config_logger(**self.log_kwargs)

        # Warn if attempting to rescale/minimize during dynamics
        # but equil_steps is too low
        if rescale_velocities and equil_steps < rescale_every:
            warn("Velocities and angular momentum will not be reset during dynamics")
        if minimize and equil_steps < minimize_every:
            warn("Geometry will not be minimized during dynamics")

        # Warn if attempting to remove rotation without resetting velocities
        if remove_rot and not rescale_velocities:
            warn("Rotation will not be removed unless `rescale_velocities` is True")

        self.minimize_kwargs = minimize_kwargs if minimize_kwargs else {}
        self.restart_files = []
        self.dyn = None
        self.n_atoms = len(self.struct)

        # Infer names for structure and if not specified
        if not self.struct_name:
            self.struct_name = self.struct.get_chemical_formula()

        self.configure_filenames()

        self.offset = 0

        if "masses" not in self.struct.arrays.keys():
            self.struct.set_masses()

        if self.seed:
            np.random.seed(seed)
            random.seed(seed)

    def rotate_restart_files(self) -> None:
        """Rotate restart files."""
        if len(self.restart_files) > self.restarts_to_keep:
            path = Path(self.restart_files.pop(0))
            path.unlink(missing_ok=True)

    def reset_velocities(self) -> None:
        """Reset velocities and (optionally) rotation of system."""
        if self.dyn.nsteps < self.equil_steps:
            MaxwellBoltzmannDistribution(self.struct, temperature_K=self.temp)
            Stationary(self.struct)
            if self.logger:
                self.logger.info("Velocities reset at step %s", self.dyn.nsteps)
            if self.remove_rot:
                ZeroRotation(self.struct)
                if self.logger:
                    self.logger.info("Rotation reset at step %s", self.dyn.nsteps)

    def optimize_structure(self) -> None:
        """Perform geometry optimization."""
        if self.dyn.nsteps < self.equil_steps:
            if self.logger:
                self.minimize_kwargs["log_kwargs"] = {
                    "filename": self.log_kwargs["filename"],
                    "name": self.logger.name,
                    "filemode": "a",
                }
                self.logger.info("Minimizing at step %s", self.dyn.nsteps)
            optimize(self.struct, **self.minimize_kwargs)

    def configure_filenames(self) -> None:
        """Setup filenames for output files."""
        if not self.file_prefix:
            self.file_prefix = f"{self.struct_name}-{self.ensemble}-T{self.temp}"

        if not self.stats_file:
            self.stats_file = f"{self.file_prefix}-stats.dat"

        if not self.traj_file:
            self.traj_file = f"{self.file_prefix}-traj.xyz"

        if not self.restart_stem:
            self.restart_stem = f"{self.file_prefix}-res"

    @staticmethod
    def get_log_header() -> str:
        """
        Get header string for molecular dynamics log.

        Returns
        -------
        str
            Header for molecular dynamics log.
        """
        log_header = (
            "Step | Real Time [s] | Time [fs] | Epot/N [eV] | Ekin/N [eV] | "
            "T [K] | Etot/N [eV] | Density [g/cm^3] | Volume [A^3] | P [bar] | "
            "Pxx [bar] | Pyy [bar] | Pzz [bar] | Pyz [bar] | Pxz [bar] | Pxy [bar]"
        )

        return log_header

    def get_log_stats(self) -> str:
        """
        Get thermodynamical statistics to be written to molecular dynamics log.

        Returns
        -------
        str
            Thermodynamical statistics to be written out.
        """
        e_pot = self.dyn.atoms.get_potential_energy() / self.n_atoms
        e_kin = self.dyn.atoms.get_kinetic_energy() / self.n_atoms
        current_temp = e_kin / (1.5 * units.kB)

        time = self.offset * self.timestep + self.dyn.get_time() / units.fs
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
                / units.bar
            )
            pressure_tensor = (
                -self.dyn.atoms.get_stress(include_ideal_gas=True, voigt=True)
                / units.bar
            )
        except ValueError:
            volume = 0.0
            pressure = 0.0
            density = 0.0
            pressure_tensor = np.zeros(6)

        log_stats = (
            f"{step:10d} {real_time.total_seconds():.3f} {time:13.2f} {e_pot:.3e} "
            f"{e_kin:.3e} {current_temp:.3f} {e_pot + e_kin:.3e} {density:.3f} "
            f"{volume:.3e} {pressure:.3e} {pressure_tensor[0]:.3e} "
            f"{pressure_tensor[1]:.3e} {pressure_tensor[2]:.3e} "
            f"{pressure_tensor[3]:.3e} {pressure_tensor[4]:.3e} "
            f"{pressure_tensor[5]:.3e}"
        )

        return log_stats

    def write_md_log(self) -> None:
        """Write molecular dynamics log."""
        log_stats = self.get_log_stats()
        with open(self.stats_file, "a", encoding="utf8") as md_log:
            print(log_stats, file=md_log)

    def write_traj(self) -> None:
        """Write current structure to trajectory file."""
        if self.dyn.nsteps >= self.traj_start:
            # Append if restarting or already started writing
            append = self.restart or (
                self.dyn.nsteps > self.traj_start + self.traj_start % self.traj_every
            )

            self.dyn.atoms.write(
                self.traj_file,
                write_info=True,
                columns=["symbols", "positions", "momenta", "masses"],
                append=append,
            )

    def write_restart(self) -> None:
        """Write restart file and (optionally) rotate files saved."""
        step = self.offset + self.dyn.nsteps
        if step > 0:
            restart_file = f"{self.restart_stem}-{step}.xyz"
            write(
                restart_file,
                self.struct,
                write_info=True,
                columns=["symbols", "positions", "momenta", "masses"],
            )
            if self.rotate_restart:
                self.restart_files.append(restart_file)
                self.rotate_restart_files()

    def run(self) -> None:
        """Run molecular dynamics simulation."""
        if self.logger:
            self.logger.info("Starting molecular dynamics simulation")

        self.struct.info["real_time"] = datetime.datetime.now()

        if self.restart:
            try:
                with open(self.stats_file, encoding="utf8") as md_log:
                    last_line = md_log.readlines()[-1].split()
                try:
                    self.offset = int(last_line[0])
                except (IndexError, ValueError) as e:
                    raise ValueError("Unable to read restart file") from e
            except FileNotFoundError as e:
                raise FileNotFoundError("Unable to read restart file") from e

        else:
            if self.minimize:
                optimize(self.struct, **self.minimize_kwargs)
            if self.rescale_velocities:
                self.reset_velocities()

            log_header = self.get_log_header()
            with open(self.stats_file, "w", encoding="utf8") as md_log:
                print(log_header, file=md_log)

        self.dyn.attach(self.write_md_log, interval=self.stats_every)
        self.dyn.attach(self.write_traj, interval=self.traj_every)
        self.dyn.attach(self.write_restart, interval=self.restart_every)

        if self.rescale_velocities:
            self.dyn.attach(self.reset_velocities, interval=self.rescale_every)

        if self.minimize and self.minimize_every > 0:
            self.dyn.attach(self.optimize_structure, interval=self.minimize_every)

        self.dyn.run(self.steps)

        if self.logger:
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
        Pressure, in bar. Default is 0.0.
    ensemble : Ensembles
        Name for thermodynamic ensemble. Default is "npt".
    file_prefix : Optional[PathLike]
        Prefix for output filenames. Default is inferred from structure, ensemble,
        temperature, and pressure.
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
            Pressure, in bar. Default is 0.0.
        ensemble : Ensembles
            Name for thermodynamic ensemble. Default is "npt".
        file_prefix : Optional[PathLike]
            Prefix for output filenames. Default is inferred from structure, ensemble,
            temperature, and pressure.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(ensemble=ensemble, file_prefix=file_prefix, *args, **kwargs)

        self.pressure = pressure
        self.ttime = thermostat_time * units.fs
        scaled_bulk_modulus = bulk_modulus * units.GPa
        if barostat_time:
            pfactor = (barostat_time * units.fs) ** 2 * scaled_bulk_modulus
        else:
            pfactor = None

        # Reconfigure filenames to include pressure if `file_prefix` not specified
        # Requires super().__init__ first to determine `self.struct_name`
        if not file_prefix and not isinstance(self, NVT_NH):
            self.file_prefix = (
                f"{self.struct_name}-{self.ensemble}-T{self.temp}-p{self.pressure}"
            )
            if not kwargs.get("stats_file"):
                self.stats_file = ""
            if not kwargs.get("traj_file"):
                self.traj_file = ""
            if not kwargs.get("restart_stem"):
                self.restart_stem = ""
            self.configure_filenames()

        self.dyn = ASE_NPT(
            self.struct,
            timestep=self.timestep,
            temperature_K=self.temp,
            ttime=self.ttime,
            pfactor=pfactor,
            append_trajectory=self.traj_append,
            externalstress=self.pressure * units.bar,
        )

    def get_log_stats(self) -> str:
        """
        Get thermodynamical statistics to be written to molecular dynamics log.

        Returns
        -------
        str
            Thermodynamical statistics to be written out.
        """
        log_stats = MolecularDynamics.get_log_stats(self)
        return log_stats + f" {self.pressure} {self.temp}"

    @staticmethod
    def get_log_header() -> str:
        """
        Get header string for molecular dynamics log.

        Returns
        -------
        str
            Header for molecular dynamics log.
        """
        log_header = MolecularDynamics.get_log_header()
        return log_header + " | Target P [bar] | Target T [K]"


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
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(ensemble=ensemble, *args, **kwargs)

        self.dyn = Langevin(
            self.struct,
            timestep=self.timestep,
            temperature_K=self.temp,
            friction=friction / units.fs,
            append_trajectory=self.traj_append,
        )

    def get_log_stats(self) -> str:
        """
        Get thermodynamical statistics to be written to molecular dynamics log.

        Returns
        -------
        str
            Thermodynamical statistics to be written out.
        """
        log_stats = MolecularDynamics.get_log_stats(self)
        return log_stats + f" {self.temp}"

    @staticmethod
    def get_log_header() -> str:
        """
        Get header string for molecular dynamics log.

        Returns
        -------
        str
            Header for molecular dynamics log.
        """
        log_header = MolecularDynamics.get_log_header()
        return log_header + " | Target T [K]"


class NVE(MolecularDynamics):
    """
    Configure NVE simulation.

    Parameters
    ----------
    *args
        Additional arguments.
    ensemble : Ensembles
        Name for thermodynamic ensemble. Default is "nve".
    **kwargs
        Additional keyword arguments.

    Attributes
    ----------
    dyn : Dynamics
        Configured NVE dynamics.
    """

    def __init__(self, *args, ensemble: Ensembles = "nve", **kwargs) -> None:
        """
        Initialise dynamics for NVE simulation.

        Parameters
        ----------
        *args
            Additional arguments.
        ensemble : Ensembles
            Name for thermodynamic ensemble. Default is "nve".
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(ensemble=ensemble, *args, **kwargs)
        self.dyn = VelocityVerlet(
            self.struct,
            timestep=self.timestep,
            append_trajectory=self.traj_append,
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
    **kwargs
        Additional keyword arguments.
    """

    def __init__(
        self,
        *args,
        thermostat_time: float = 50.0,
        ensemble: Ensembles = "nvt-nh",
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
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(
            ensemble=ensemble,
            thermostat_time=thermostat_time,
            barostat_time=None,
            *args,
            **kwargs,
        )

    def get_log_stats(self) -> str:
        """
        Get thermodynamical statistics to be written to molecular dynamics log.

        Returns
        -------
        str
            Thermodynamical statistics to be written out.
        """
        log_stats = MolecularDynamics.get_log_stats(self)
        return log_stats + f" {self.temp}"

    @staticmethod
    def get_log_header() -> str:
        """
        Get header string for molecular dynamics log.

        Returns
        -------
        str
            Header for molecular dynamics log.
        """
        log_header = MolecularDynamics.get_log_header()
        return log_header + " | Target T [K]"


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
        Pressure, in bar. Default is 0.0.
    ensemble : Ensembles
        Name for thermodynamic ensemble. Default is "nph".
    file_prefix : Optional[PathLike]
        Prefix for output filenames. Default is inferred from structure, ensemble,
        temperature, and pressure.
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
            Pressure, in bar. Default is 0.0.
        ensemble : Ensembles
            Name for thermodynamic ensemble. Default is "nph".
        file_prefix : Optional[PathLike]
            Prefix for output filenames. Default is inferred from structure, ensemble,
            temperature, and pressure.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(
            *args,
            thermostat_time=thermostat_time,
            barostat_time=None,
            bulk_modulus=bulk_modulus,
            pressure=pressure,
            ensemble=ensemble,
            file_prefix=file_prefix,
            **kwargs,
        )
