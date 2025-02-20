"""Prepare and run Nudged Elastic Band method."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import copy
from typing import Any

from ase import Atoms
import ase.mep
from ase.mep import NEB as ASE_NEB
from ase.mep import NEBTools
from ase.mep.neb import NEBOptimizer
from matplotlib.figure import Figure
from numpy.linalg import LinAlgError
from pymatgen.io.ase import AseAtomsAdaptor

from janus_core.calculations.base import BaseCalculation
from janus_core.calculations.geom_opt import GeomOpt
from janus_core.helpers.janus_types import (
    Architectures,
    ASEReadArgs,
    Devices,
    Interpolators,
    OutputKwargs,
    PathLike,
)
from janus_core.helpers.struct_io import input_structs, output_structs
from janus_core.helpers.utils import none_to_dict, set_minimize_logging


class NEB(BaseCalculation):
    """
    Prepare and run Nudged Elastic Band method.

    Parameters
    ----------
    init_struct
        Initial ASE Atoms structure for Nudged Elastic Band method. Required if
        `init_struct_path` is None. Default is None.
    init_struct_path
        Path of initial structure for Nudged Elastic Band method. Required if
        `init_struct` is None. Default is None.
    final_struct
        Final ASE Atoms structure for Nudged Elastic Band method. Required if
        `final_struct_path` is None. Any attached calculators will be replaced with a
        copy matching the one attached to init_struct. Default is None.
    final_struct_path
        Path of final structure for Nudged Elastic Band method. Required if
        `final_struct` is None. Default is None.
    band_structs
        Band of ASE Atoms images to optimize, skipping interpolation between the initial
        and final structures. sets `interpolator` to None.
    band_path
        Path of band of images to optimize, skipping interpolation between the initial
        and final structures. sets `interpolator` to None.
    arch
        MLIP architecture to use for Nudged Elastic Band method. Default is "mace_mp".
    device
        Device to run MLIP model on. Default is "cpu".
    model_path
        Path to MLIP model. Default is `None`.
    read_kwargs
        Keyword arguments to pass to ase.io.read. By default, read_kwargs["index"]
        is -1 if using `init_struct` and `final_struct`, or ":" for `band_structs`.
    calc_kwargs
        Keyword arguments to pass to the selected calculator. Default is {}.
    set_calc
        Whether to set (new) calculators for structures. Default is None.
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
    neb_class
        Nudged Elastic Band class to use. Default is ase.mep.NEB.
    neb_kwargs
        Keyword arguments to pass to neb_class. Default is {}.
    n_images
        Number of images to use in NEB. Default is 15.
    write_results
        Whether to write out results from NEB. Default is True.
    write_band
        Whether to write out all band images after optimization. Default is False.
    write_kwargs
        Keyword arguments to pass to ase.io.write when writing images.
    interpolator
        Choice of interpolation strategy. Default is "ase" if using `init_struct` and
        `final_struct`, or None if using `band_structs`.
    interpolator_kwargs
        Keyword arguments to pass to interpolator. Default is
        {"method": "idpp"} for "ase" interpolator, or
        {"interpolate_lattices": False, "autosort_tol", 0.5} for "pymatgen".
    optimizer
        Optimizer to apply to NEB object. Default is NEBOptimizer.
    fmax
        Maximum force for NEB optimizer. Default is 0.1.
    steps
        Maximum number of steps to optimize NEB. Default is 100.
    optimizer_kwargs
        Keyword arguments to pass to optimizer. Deault is {}.
    plot_band
        Whether to plot and save NEB band. Default is False.
    minimize
        Whether to perform geometry optimisation on initial and final structures.
        Default is False.
    minimize_kwargs
        Keyword arguments to pass to geometry optimizer. Default is {}.
    file_prefix
        Prefix for output filenames. Default is inferred from the intial structure
        name, or chemical formula of the intial structure.
    """

    def __init__(
        self,
        init_struct: Atoms | None = None,
        init_struct_path: PathLike | None = None,
        final_struct: Atoms | None = None,
        final_struct_path: PathLike | None = None,
        band_structs: Sequence[Atoms] | None = None,
        band_path: PathLike | None = None,
        arch: Architectures = "mace_mp",
        device: Devices = "cpu",
        model_path: PathLike | None = None,
        read_kwargs: ASEReadArgs | None = None,
        calc_kwargs: dict[str, Any] | None = None,
        set_calc: bool | None = None,
        attach_logger: bool | None = None,
        log_kwargs: dict[str, Any] | None = None,
        track_carbon: bool | None = None,
        tracker_kwargs: dict[str, Any] | None = None,
        neb_class: Callable | str = ASE_NEB,
        neb_kwargs: dict[str, Any] | None = None,
        n_images: int = 15,
        write_results: bool = True,
        write_band: bool = False,
        write_kwargs: OutputKwargs | None = None,
        interpolator: Interpolators | None = "ase",
        interpolator_kwargs: dict[str, Any] | None = None,
        optimizer: Callable | str = NEBOptimizer,
        fmax: float = 0.1,
        steps: int = 100,
        optimizer_kwargs: dict[str, Any] | None = None,
        plot_band: bool = False,
        minimize: bool = False,
        minimize_kwargs: dict[str, Any] | None = None,
        file_prefix: PathLike | None = None,
    ) -> None:
        """
        Initialise NEB class.

        Parameters
        ----------
        init_struct
            Initial ASE Atoms structure for Nudged Elastic Band method. Required if
            `init_struct_path` is None. Default is None.
        init_struct_path
            Path of initial structure for Nudged Elastic Band method. Required if
            `init_struct` is None. Default is None.
        final_struct
            Final ASE Atoms structure for Nudged Elastic Band method. Required if
            `final_struct_path` is None. Default is None. Any attached calculators will
            be replaced with a copy matching the one attached to init_struct. Default
            is None.
        final_struct_path
            Path of final structure for Nudged Elastic Band method. Required if
            `final_struct` is None. Default is None.
        band_structs
            Band of ASE Atoms images to optimize, skipping interpolation between the
            initial and final structures. sets `interpolator` to None.
        band_path
            Path of band of images to optimize, skipping interpolation between the
            initial and final structures. sets `interpolator` to None.
        arch
            MLIP architecture to use for Nudged Elastic Band method. Default is
            "mace_mp".
        device
            Device to run MLIP model on. Default is "cpu".
        model_path
            Path to MLIP model. Default is `None`.
        read_kwargs
            Keyword arguments to pass to ase.io.read. By default, read_kwargs["index"]
            is -1 if using `init_struct` and `final_struct`, or ":" for `band_structs`.
        calc_kwargs
            Keyword arguments to pass to the selected calculator. Default is {}.
        set_calc
            Whether to set (new) calculators for structures. Default is None.
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
        neb_class
            Nudged Elastic Band class to use. Default is ase.mep.NEB.
        neb_kwargs
            Keyword arguments to pass to neb_class. Default is {}.
        n_images
            Number of images to use in NEB. Default is 15.
        write_results
            Whether to write out results from NEB. Default is Trueß.
        write_band
            Whether to write out all band images after optimization. Default is False.
        write_kwargs
            Keyword arguments to pass to ase.io.write when writing images.
        interpolator
            Choice of interpolation strategy. Default is "ase" if using `init_struct`
            and `final_struct`, or None if using `band_structs`.
        interpolator_kwargs
            Keyword arguments to pass to interpolator. Default is
            {"method": "idpp"} for "ase" interpolator, or
            {"interpolate_lattices": False, "autosort_tol", 0.5} for "pymatgen".
        optimizer
            Optimizer to apply to NEB object. Default is NEBOptimizer.
        fmax
            Maximum force for NEB optimizer. Default is 0.1.
        steps
            Maximum number of steps to optimize NEB. Default is 100.
        optimizer_kwargs
            Keyword arguments to pass to optimizer. Deault is {}.
        plot_band
            Whether to plot and save NEB band. Default is False.
        minimize
            Whether to perform geometry optimisation on initial and final structures.
            Default is False.
        minimize_kwargs
            Keyword arguments to pass to geometry optimizer. Default is {}.
        file_prefix
            Prefix for output filenames. Default is inferred from the intial structure
            name, or chemical formula of the intial structure.
        """
        (
            read_kwargs,
            write_kwargs,
            neb_kwargs,
            interpolator_kwargs,
            optimizer_kwargs,
            minimize_kwargs,
        ) = none_to_dict(
            read_kwargs,
            write_kwargs,
            neb_kwargs,
            interpolator_kwargs,
            optimizer_kwargs,
            minimize_kwargs,
        )

        self.neb_class = neb_class
        self.n_images = n_images
        self.write_results = write_results
        self.write_band = write_band
        self.write_kwargs = write_kwargs
        self.neb_kwargs = neb_kwargs
        self.interpolator = interpolator
        self.interpolator_kwargs = interpolator_kwargs
        self.optimizer = optimizer
        self.fmax = fmax
        self.steps = steps
        self.optimizer_kwargs = optimizer_kwargs
        self.plot_band = plot_band
        self.minimize = minimize
        self.minimize_kwargs = minimize_kwargs

        # Validate parameters
        if self.n_images <= 0 or not isinstance(self.n_images, int):
            raise ValueError("`n_images` must be an integer greater than 0.")

        # Identify whether interpolating
        if band_structs or band_path:
            self.interpolator = None
            if init_struct or init_struct_path or final_struct or final_struct_path:
                raise ValueError(
                    "Band cannot be specified in combination with an initial or final "
                    "structure"
                )

            if minimize:
                raise ValueError("Cannot minimize band structures.")

            # Pass band strutures to base class init
            init_struct = band_structs
            init_struct_path = band_path

            # Read all image by default for band
            read_kwargs.setdefault("index", ":")
        else:
            if self.interpolator is None:
                raise ValueError(
                    "An interpolator must be specified when using an initial and final "
                    "structure"
                )
            # Read last image by default for init_struct
            read_kwargs.setdefault("index", -1)

        # Initialise structures and logging
        super().__init__(
            calc_name=__name__,
            struct=init_struct,
            struct_path=init_struct_path,
            arch=arch,
            device=device,
            model_path=model_path,
            read_kwargs=read_kwargs,
            sequence_allowed=True,
            calc_kwargs=calc_kwargs,
            set_calc=set_calc,
            attach_logger=attach_logger,
            log_kwargs=log_kwargs,
            track_carbon=track_carbon,
            tracker_kwargs=tracker_kwargs,
            file_prefix=file_prefix,
        )

        if self.interpolator:
            if not isinstance(self.struct, Atoms):
                raise ValueError("`init_struct` must be a single structure.")
            if not self.struct.calc:
                raise ValueError("Please attach a calculator to `init_struct`.")

            # Use initial structure (path) for default file paths etc.
            self.init_struct = self.struct
            self.init_struct_path = self.struct_path

            self.final_struct = input_structs(
                struct=final_struct,
                struct_path=final_struct_path,
                read_kwargs=read_kwargs,
                sequence_allowed=False,
                set_calc=False,
            )
            self.final_struct_path = final_struct_path
            self.final_struct.calc = copy(self.struct.calc)
        else:
            if not isinstance(self.struct, Sequence):
                raise ValueError("`images` must include multiple structures.")
            self.images = self.struct

        # Set default interpolation kwargs
        if self.interpolator == "ase":
            interpolator_kwargs.setdefault("method", "idpp")
        if self.interpolator == "pymatgen":
            interpolator_kwargs.setdefault("interpolate_lattices", False)
            interpolator_kwargs.setdefault("autosort_tol", 0.5)

        # Set output file defaults
        self.results_file = self._build_filename("neb-results.dat").absolute()
        self.plot_file = self._build_filename("neb-plot.svg").absolute()

        self.write_kwargs["filename"] = self._build_filename(
            "neb-band.extxyz"
        ).absolute()

        if self.minimize:
            set_minimize_logging(
                self.logger, self.minimize_kwargs, self.log_kwargs, track_carbon
            )

            # Variable cell in periodic directions is not implemented yet for NEB
            self.minimize_kwargs.setdefault("filter_func", None)

            # Write out file by default
            self.minimize_kwargs.setdefault("write_results", True)

            # Set minimized file paths
            self.init_struct_min_path = self._build_filename("init-opt.extxyz")
            self.final_struct_min_path = self._build_filename(
                "final-opt.extxyz",
                prefix_override=self._get_default_prefix(
                    file_prefix, final_struct, final_struct_path
                ),
            )
            if "write_kwargs" in self.minimize_kwargs:
                if "filename" in self.minimize_kwargs["write_kwargs"]:
                    raise ValueError(
                        "Filenames for minimized structures cannot currently be set"
                    )
                self.minimize_kwargs["write_kwargs"]["filename"] = (
                    self.init_struct_min_path
                )
            else:
                self.minimize_kwargs["write_kwargs"] = {
                    "filename": self.init_struct_min_path
                }

        # Set NEB method
        self._set_neb()

    def _set_neb(self) -> None:
        """Set NEB method and optimizer."""
        # Set NEB method
        if isinstance(self.neb_class, str):
            try:
                self.neb_class = getattr(ase.mep, self.neb_class)
            except AttributeError as e:
                raise AttributeError(f"No such class: {self.neb_class}") from e
        if self.logger:
            self.logger.info("Using NEB class: %s", self.neb_class.__name__)

        # Set NEB optimizer
        if isinstance(self.optimizer, str):
            try:
                self.optimizer = getattr(ase.mep.neb, self.optimizer)
            except AttributeError:
                try:
                    self.optimizer = getattr(ase.optimize, self.optimizer)
                except AttributeError as e:
                    raise AttributeError(f"No such class: {self.optimizer}") from e
        if self.logger:
            self.logger.info("Using optimizer: %s", self.optimizer.__name__)

    def plot(self) -> Figure | None:
        """
        Plot NEB band and save figure.

        Returns
        -------
        Figure | None
            Plotted NEB band.
        """
        if not hasattr(self.neb, "nebtools"):
            self.run_nebtools()

        if self.plot_band:
            fig = self.nebtools.plot_band()
            fig.savefig(self.plot_file)
        else:
            fig = None

        return fig

    def interpolate(self) -> None:
        """Interpolate images to create initial band."""
        match self.interpolator:
            case "ase":
                # Create band of images and attach calculators
                if self.logger:
                    self.logger.info("Using ASE interpolator")
                self.images = [self.init_struct]
                self.images += [self.init_struct.copy() for i in range(self.n_images)]
                for image in self.images[1:]:
                    image.calc = copy(self.init_struct.calc)
                self.images += [self.final_struct]

                self.neb = self.neb_class(self.images, **self.neb_kwargs)
                self.neb.interpolate(**self.interpolator_kwargs)

            case "pymatgen":
                # Create band of images and attach calculators
                if self.logger:
                    self.logger.info("Using pymatgen interpolator")
                try:
                    py_start_struct = AseAtomsAdaptor.get_structure(self.init_struct)
                    py_final_struct = AseAtomsAdaptor.get_structure(self.final_struct)
                except LinAlgError as e:
                    raise ValueError(
                        "Unable to convert to pymatgen structure. Please use the "
                        "'ase' interpolator instead"
                    ) from e
                py_images = py_start_struct.interpolate(
                    py_final_struct,
                    nimages=self.n_images + 1,
                    **self.interpolator_kwargs,
                )
                self.images = [image.to_ase_atoms() for image in py_images]
                for image in self.images:
                    image.calc = copy(self.init_struct.calc)

                self.neb = self.neb_class(self.images, **self.neb_kwargs)

            case None:
                # Band already created
                if self.logger:
                    self.logger.info("Skipping interpolation")
                self.neb = self.neb_class(self.images, **self.neb_kwargs)
                pass
            case _:
                raise ValueError("Invalid interpolator selected")

    def optimize(self):
        """Run NEB optimization."""
        if not hasattr(self, "neb"):
            self.interpolate()

        optimizer = self.optimizer(self.neb, **self.optimizer_kwargs)
        optimizer.run(fmax=self.fmax, steps=self.steps)
        if self.logger:
            self.logger.info("Optimization steps: %s", optimizer.nsteps)

        # Optionally write band images to file
        output_structs(
            images=self.images,
            struct_path=self.struct_path,
            write_results=self.write_band,
            write_kwargs=self.write_kwargs,
        )

    def run_nebtools(self):
        """Run NEBTools analysis."""
        self.nebtools = NEBTools(self.images[1:-1])
        barrier, delta_E = self.nebtools.get_barrier()  # noqa: N806
        max_force = self.nebtools.get_fmax()
        self.results = {
            "barrier": barrier,
            "delta_E": delta_E,
            "max_force": max_force,
        }

        if self.write_results:
            with open(self.results_file, "w", encoding="utf8") as out:
                print("#Barrier [eV] | delta E [eV] | Max force [eV/Å] ", file=out)
                print(*self.results.values(), file=out)

    def run(self) -> dict[str, float]:
        """
        Run Nudged Elastic Band method.

        Returns
        -------
        dict[str, float]
            Dictionary of calculated results.
        """
        if self.logger:
            self.logger.info("Starting Nudged Elastic Band method")
        if self.tracker:
            self.tracker.start_task("NEB")

        self._set_info_units()

        if self.minimize:
            GeomOpt(self.init_struct, **self.minimize_kwargs).run()

            # Change filename to be written
            self.minimize_kwargs["write_kwargs"]["filename"] = (
                self.final_struct_min_path
            )
            GeomOpt(self.final_struct, **self.minimize_kwargs).run()

        self.interpolate()
        self.optimize()
        self.run_nebtools()
        self.plot()

        if self.logger:
            self.logger.info("Nudged Elastic Band method complete")
        if self.tracker:
            emissions = self.tracker.stop_task().emissions
            if isinstance(self.struct, Sequence):
                for image in self.struct:
                    image.info["emissions"] = emissions
            else:
                self.struct.info["emissions"] = emissions
            self.tracker.stop()

        return self.results
