"""Run molecular dynamics simulations."""

import datetime as clock
from pathlib import Path
from typing import Any, Optional

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
        timestep : float
            Timestep for integrator, in fs. Default is 1.0.
        steps : int
            Number of steps in simulation. Default is 0.
        temp : float
            Temperature, in K. Default is 300.
        minimize : bool
            Minimize structure during equilibration. Default is False.
        minimize_every : int
            Interval between minimizations. Default is -1, which disables further
            minimization.
        equil_steps : int
            Number of equilibration steps. Default is 0.
        rescale_velocities : bool
            Whether to rescale velocities. Default is False.
        remove_rot : bool
            Whether to remove rotation. Default is False.
        rescale_every : int
            Frequency to rescale velocities. Default is 10.
        restart : bool
            Whether restarting dynamics. Default is False.
        restart_stem : str
            Stem for restart file name. Default inferred from struct_name and ensemble.
        restart_every : int
            Frequency of steps to save restart info. Default is 1000.
        rotate_restart : bool
            Whether to rotate restart files. Default is False.
        restarts_to_keep : int
            Restart files to keep. Default is 4.
        md_file : Optional[PathLike]
            MD file to save. Default inferred from struct_name, ensemble and
            temperature.
        traj_file : Optional[PathLike]
            Trajectory file to save. Default inferred from struct_name, ensemble and
            temperature.
        traj_append : bool
            Whether to append trajectory. Default is False.
        traj_start : int
            Step to start saving trajectory. Default is 0.
        traj_every : int
            Frequency of steps to save trajectory. Default is 100.
        log_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to log config. Default is None.
        output_every : int
            Frequency to output MD logs to. Default is 100.
        ensemble : Ensembles
            Name for thermodynamic ensemble. Default is None.

        Attributes
        ----------
        logger : logging.Logger
            Logger if log file has been specified.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
        self,
        struct: Atoms,
        struct_name: Optional[str] = None,
        timestep: float = 1.0,
        steps: int = 0,
        temp: float = 300,
        minimize: bool = False,
        minimize_every: int = -1,
        equil_steps: int = 0,
        rescale_velocities: bool = False,
        remove_rot: bool = False,
        rescale_every: int = 10,
        restart: bool = False,
        restart_stem: Optional[PathLike] = None,
        restart_every: int = 1000,
        rotate_restart: bool = False,
        restarts_to_keep: int = 4,
        md_file: Optional[PathLike] = None,
        traj_file: Optional[PathLike] = None,
        traj_append: bool = False,
        traj_start: int = 0,
        traj_every: int = 100,
        log_kwargs: Optional[dict[str, Any]] = None,
        output_every: int = 100,
        ensemble: Optional[Ensembles] = None,
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
        timestep : float
            Timestep for integrator, in fs. Default is 1.0.
        steps : int
            Number of steps in simulation. Default is 0.
        temp : float
            Temperature, in K. Default is 300.
        minimize : bool
            Minimize structure during equilibration. Default is False.
        minimize_every : int
            Interval between minimizations. Default is -1, which disables further
            minimization.
        equil_steps : int
            Number of equilibration steps. Default is 0.
        rescale_velocities : bool
            Whether to rescale velocities. Default is False.
        remove_rot : bool
            Whether to remove rotation. Default is False.
        rescale_every : int
            Frequency to rescale velocities. Default is 10.
        restart : bool
            Whether restarting dynamics. Default is False.
        restart_stem : str
            Stem for restart file name. Default inferred from struct_name and ensemble.
        restart_every : int
            Frequency of steps to save restart info. Default is 1000.
        rotate_restart : bool
            Whether to rotate restart files. Default is False.
        restarts_to_keep : int
            Restart files to keep. Default is 4.
        md_file : Optional[PathLike]
            MD file to save.  Default inferred from struct_name, ensemble and
            temperature.
        traj_file : Optional[PathLike]
            Trajectory file to save. Default inferred from struct_name, ensemble and
            temperature.
        traj_append : bool
            Whether to append trajectory. Default is False.
        traj_start : int
            Step to start saving trajectory. Default is 0.
        traj_every : int
            Frequency of steps to save trajectory. Default is 100.
        log_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to log config. Default is None.
        output_every : int
            Frequency to output MD logs to. Default is 100.
        ensemble : Ensembles
            Name for thermodynamic ensemble. Default is None.
        """
        self.struct = struct
        self.struct_name = struct_name
        self.timestep = timestep * units.fs
        self.steps = steps
        self.temp = temp
        self.minimize = minimize
        self.minimize_every = minimize_every
        self.equil_steps = equil_steps
        self.rescale_velocities = rescale_velocities
        self.remove_rot = remove_rot
        self.rescale_every = rescale_every
        self.restart = restart
        self.restart_stem = restart_stem
        self.rotate_restart = rotate_restart
        self.restarts_to_keep = restarts_to_keep
        self.md_file = md_file
        self.traj_file = traj_file
        self.traj_append = traj_append
        self.traj_start = traj_start
        self.traj_every = traj_every
        self.restart_every = restart_every
        self.output_every = output_every
        self.ensemble = ensemble

        log_kwargs = log_kwargs if log_kwargs else {}  # pylint: disable=duplicate-code
        if log_kwargs and "filename" not in log_kwargs:
            raise ValueError("'filename' must be included in `log_kwargs`")

        log_kwargs.setdefault("name", __name__)
        self.logger = config_logger(**log_kwargs)

        self.restart_files = []
        self.dyn = None
        self.pressure = None
        self.n_atoms = len(self.struct)

        if not self.struct_name and isinstance(self.struct, PathLike):
            self.struct_name = Path(self.struct).stem
        else:
            self.struct_name = self.struct.get_chemical_formula()

        if not self.md_file:
            self.md_file = f"{self.struct_name}-{self.ensemble}-{self.temp}-md.log"

        if not self.traj_file:
            self.traj_file = f"{self.struct_name}-{self.ensemble}-{self.temp}-traj.xyz"

        if not self.restart_stem:
            self.restart_stem = f"res-{self.struct_name}-{self.ensemble}"

        self.offset = 0

        if "masses" not in self.struct.arrays.keys():
            self.struct.set_masses()

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
            if self.remove_rot:
                ZeroRotation(self.struct)

    def optimize_structure(self) -> None:
        """Perform geometry optimization."""
        if self.dyn.nsteps < self.equil_steps:
            optimize(self.struct)

    def write_frame(self) -> None:
        """Write frame."""
        if self.dyn.nsteps > self.traj_start and self.dyn.nsteps % self.traj_every == 0:
            self.dyn.atoms.write(
                self.traj_file,
                write_info=True,
                columns=["symbols", "positions", "momenta", "masses"],
                append=True,
            )
        e_pot = self.dyn.atoms.get_potential_energy() / self.n_atoms
        e_kin = self.dyn.atoms.get_kinetic_energy() / self.n_atoms
        c_T = e_kin / (1.5 * units.kB)  # pylint: disable=invalid-name

        time = self.offset * self.timestep + self.dyn.get_time() / units.fs
        step = self.offset + self.dyn.nsteps
        self.dyn.atoms.info["time_fs"] = time
        self.dyn.atoms.info["step"] = step

        time_now = clock.datetime.now()
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
        print_stat = (
            f"{step:10d} {real_time.total_seconds():.3f} {time:13.2f} {e_pot:.3e} "
            f"{e_kin:.3e} {c_T:.3f} {e_pot + e_kin:.3e} {density:.3f} "
            f"{volume:.3e} {pressure:.3e} {pressure_tensor[0]:.3e} "
            f"{pressure_tensor[1]:.3e} {pressure_tensor[2]:.3e} "
            f"{pressure_tensor[3]:.3e} {pressure_tensor[4]:.3e} "
            f"{pressure_tensor[5]:.3e}"
        )

        if self.ensemble == "npt":
            print_stat += f"{self.pressure} {self.temp}"
        if self.ensemble in ("nvt", "nvt-nh"):
            print_stat += f"{self.temp}"

        with open(self.md_file, "a", encoding="utf8") as md_log:
            print(print_stat, file=md_log)

        if step % self.restart_every == 0:
            restart_file = f"{self.restart_stem}-{self.temp:.2f}K-{step}.xyz"
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
        self.struct.info["real_time"] = clock.datetime.now()

        if self.restart:
            with open(self.md_file, encoding="utf8") as md_log:
                last_line = md_log.readlines()[-1].split()
            try:
                self.offset = int(last_line[0])
            except IndexError:
                self.offset = 0

        else:
            if self.minimize:
                optimize(self.struct)
            self.reset_velocities()

            print_h = (
                "    Step  | real time[s] |     Time [fs] |   Epot/N [eV] | "
                "Ekin/N [eV] |  T [K] | Etot/N [eV] | Density [g/cm^3] |Volume [A^3] "
                "| Pressure [bar] |   Pxx [bar] |   Pyy [bar] |   Pzz[bar] |   "
                "Pyz[bar] |   Pxz[bar] |   Pxy[bar]"
            )
            if self.ensemble in ("nvt", "nvt-nh"):
                print_h += " |Target T [K]"
            if self.ensemble == "npt":
                print_h += " |Target Pressure[bar] |  T [K]"

            with open(self.md_file, "w", encoding="utf8") as md_log:
                print(print_h, file=md_log)

        self.dyn.attach(self.write_frame, interval=self.output_every)

        if self.rescale_velocities:
            self.dyn.attach(self.reset_velocities, interval=self.rescale_every)

        if self.minimize and self.minimize_every > 0:
            self.dyn.attach(self.optimize_structure, interval=self.minimize_every)

        self.dyn.run(self.steps)


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
        **kwargs
            Additional keyword arguments.
        """
        self.ensemble = ensemble
        super().__init__(ensemble=self.ensemble, *args, **kwargs)

        self.pressure = pressure
        self.ttime = thermostat_time * units.fs
        scaled_bulk_modulus = bulk_modulus * units.GPa
        if barostat_time:
            pfactor = (barostat_time * units.fs) ** 2 * scaled_bulk_modulus
        else:
            pfactor = None

        self.dyn = ASE_NPT(
            self.struct,
            timestep=self.timestep,
            temperature_K=self.temp,
            ttime=self.ttime,
            pfactor=pfactor,
            append_trajectory=self.traj_append,
            externalstress=self.pressure * units.bar,
        )


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
        self.ensemble = ensemble
        super().__init__(ensemble=self.ensemble, *args, **kwargs)

        self.dyn = Langevin(
            self.struct,
            timestep=self.timestep,
            temperature_K=self.temp,
            friction=friction / units.fs,
            append_trajectory=self.traj_append,
        )


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
        self.ensemble = ensemble
        super().__init__(ensemble=self.ensemble, *args, **kwargs)
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
        self.ensemble = ensemble
        self.ttime = thermostat_time
        super().__init__(
            ensemble=self.ensemble,
            thermostat_time=self.ttime,
            barostat_time=None,
            *args,
            **kwargs,
        )


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
        **kwargs
            Additional keyword arguments.
        """
        self.ensemble = ensemble
        super().__init__(
            *args,
            thermostat_time=thermostat_time,
            barostat_time=None,
            bulk_modulus=bulk_modulus,
            pressure=pressure,
            ensemble=self.ensemble,
            **kwargs,
        )