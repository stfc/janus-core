"""Prepare and run Nudged Elastic Band method."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import copy
from typing import Any

from ase import Atoms
import ase.mep
from ase.mep import NEB as ASE_NEB
from ase.mep import DyNEB, NEBTools
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
        and final structures. Requires interpolator to be None.
    band_path
        Path of band of images to optimize, skipping interpolation between the initial
        and final structures. Requires interpolator to be None.
    arch
        MLIP architecture to use for Nudged Elastic Band method. Default is "mace_mp".
    device
        Device to run MLIP model on. Default is "cpu".
    model_path
        Path to MLIP model. Default is `None`.
    read_kwargs
        Keyword arguments to pass to ase.io.read. By default,
        read_kwargs["index"] is -1.
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
    neb_method
        Nudged Elastic Band method to use. Default is ase.mep.NEB.
    n_images
        Number of images to use in NEB. Default is 15.
    write_results
        Whether to write out results from NEB. Default is True.
    write_images
        Whether to write out all band images after optimization. Default is False.
    write_kwargs
        Keyword arguments to pass to ase.io.write when writing images.
    neb_kwargs
        Keyword arguments to pass to neb_method. Defaults are
        {"k": 0.1, "climb": True, "method": "string"} for NEB,
        {"fmax": 0.1, "dynamic_relaxation": True, "climb": True, "scale_fmax": 1.2} for
        DynNEB, else {}.
    interpolator
        Choice of interpolation strategy. Default is "ase".
    interpolation_kwargs
        Keyword arguments to pass to interpolator. Default is
        {"method": "idpp"}.
    neb_optimizer
        Optimizer to apply to NEB object. Default is NEBOptimizer.
    fmax
        Maximum force for NEB optimizer. Default is 0.1.
    steps
        Maximum number of steps to optimize NEB. Default is 100.
    optimizer_kwargs
        Keyword arguments to pass to neb_optimizer. Deault is {}.
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

    Methods
    -------
    run()
        Run Nudged Elastic Band method.
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
        neb_method: Callable | str = ASE_NEB,
        n_images: int = 15,
        write_results: bool = True,
        write_images: bool = False,
        write_kwargs: OutputKwargs | None = None,
        neb_kwargs: dict[str, Any] | None = None,
        interpolator: Interpolators | None = "ase",
        interpolation_kwargs: dict[str, Any] | None = None,
        neb_optimizer: Callable | str = NEBOptimizer,
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
            initial and final structures. Requires interpolator to be None.
        band_path
            Path of band of images to optimize, skipping interpolation between the
            initial and final structures. Requires interpolator to be None.
        arch
            MLIP architecture to use for Nudged Elastic Band method. Default is
            "mace_mp".
        device
            Device to run MLIP model on. Default is "cpu".
        model_path
            Path to MLIP model. Default is `None`.
        read_kwargs
            Keyword arguments to pass to ase.io.read. By default,
            read_kwargs["index"] is -1.
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
        neb_method
            Nudged Elastic Band method to use. Default is ase.mep.NEB.
        n_images
            Number of images to use in NEB. Default is 15.
        write_results
            Whether to write out results from NEB. Default is Trueß.
        write_images
            Whether to write out all band images after optimization. Default is False.
        write_kwargs
            Keyword arguments to pass to ase.io.write when writing images.
        neb_kwargs
            Keyword arguments to pass to neb_method. Defaults are
            {"k": 0.1, "climb": True, "method": "string"} for NEB,
            {"fmax": 0.1, "dynamic_relaxation": True, "climb": True, "scale_fmax": 1.2}
            for DynNEB, else {}.
        interpolator
            Choice of interpolation strategy. Default is "ase".
        interpolation_kwargs
            Keyword arguments to pass to interpolator. Default is
            {"method": "idpp"}.
        neb_optimizer
            Optimizer to apply to NEB object. Default is NEBOptimizer.
        fmax
            Maximum force for NEB optimizer. Default is 0.1.
        steps
            Maximum number of steps to optimize NEB. Default is 100.
        optimizer_kwargs
            Keyword arguments to pass to neb_optimizer. Deault is {}.
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
            interpolation_kwargs,
            optimizer_kwargs,
            minimize_kwargs,
        ) = none_to_dict(
            read_kwargs,
            write_kwargs,
            neb_kwargs,
            interpolation_kwargs,
            optimizer_kwargs,
            minimize_kwargs,
        )

        self.neb_method = neb_method
        self.n_images = n_images
        self.write_results = write_results
        self.write_images = write_images
        self.write_kwargs = write_kwargs
        self.neb_kwargs = neb_kwargs
        self.interpolator = interpolator
        self.interpolation_kwargs = interpolation_kwargs
        self.neb_optimizer = neb_optimizer
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
            interpolating = False
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
            interpolating = True
            if interpolator is None:
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

        if interpolating:
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
            interpolation_kwargs.setdefault("method", "idpp")
        if self.interpolator == "pymatgen":
            interpolation_kwargs.setdefault("interpolate_lattices", False)
            interpolation_kwargs.setdefault("autosort_tol", 0.5)

        # Set output file defaults
        self.results_file = self._build_filename("neb-results.dat").absolute()
        self.plot_file = self._build_filename("neb-plot.svg").absolute()

        self.write_kwargs["filename"] = self._build_filename(
            "neb-images.extxyz"
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
        """Set NEB method, kwargs and optimizer."""
        # Set NEB method
        if isinstance(self.neb_method, str):
            try:
                self.neb_method = getattr(ase.mep, self.neb_method)
            except AttributeError as e:
                raise AttributeError(f"No such method: {self.neb_method}") from e

        # Set NEB optimizer
        if isinstance(self.neb_optimizer, str):
            try:
                self.neb_optimizer = getattr(ase.mep.neb, self.neb_optimizer)
            except AttributeError as e:
                raise AttributeError(f"No such method: {self.neb_optimizer}") from e

        # Set default neb_kwargs
        if isinstance(self.neb_method, ASE_NEB):
            neb_defaults = {"k": 0.1, "climb": True, "method": "string"}
        elif isinstance(self.neb_method, DyNEB):
            neb_defaults = {
                "fmax": 0.1,
                "dynamic_relaxation": True,
                "climb": True,
                "scale_fmax": 1.2,
            }
        else:
            neb_defaults = {}
        self.neb_kwargs = neb_defaults | self.neb_kwargs

    def plot(self) -> Figure | None:
        """
        Plot NEB band and save figure.

        Returns
        -------
        Figure | None
            Plotted NEB band.
        """
        if self.plot_band:
            fig = self.nebtools.plot_band()
            fig.savefig(self.plot_file)
        else:
            fig = None

        return fig

    def set_interpolator(self) -> None:
        """Interpolate images to create initial band."""
        match self.interpolator:
            case "ase":
                # Create band of images and attach calculators
                self.images = [self.init_struct]
                self.images += [self.init_struct.copy() for i in range(self.n_images)]
                for image in self.images[1:]:
                    image.calc = copy(self.init_struct.calc)
                self.images += [self.final_struct]

                self.neb = self.neb_method(self.images, **self.neb_kwargs)
                self.neb.interpolate(**self.interpolation_kwargs)

            case "pymatgen":
                # Create band of images and attach calculators
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
                    **self.interpolation_kwargs,
                )
                self.images = [image.to_ase_atoms() for image in py_images]
                for image in self.images:
                    image.calc = copy(self.init_struct.calc)

                self.neb = self.neb_method(self.images, **self.neb_kwargs)

            case None:
                # Band already created
                self.neb = self.neb_method(self.images, **self.neb_kwargs)
                pass
            case _:
                raise ValueError("Invalid interpolator selected")

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

        self.set_interpolator()

        optimizer = self.neb_optimizer(self.neb, **self.optimizer_kwargs)
        optimizer.run(fmax=self.fmax, steps=self.steps)

        # Optionally write band images to file
        output_structs(
            images=self.images,
            struct_path=self.struct_path,
            write_results=self.write_images,
            write_kwargs=self.write_kwargs,
        )

        self.nebtools = NEBTools(self.images[1:-1])
        barrier, delta_E = self.nebtools.get_barrier()  # noqa: N806
        max_force = self.nebtools.get_fmax()
        self.results = {
            "barrier": barrier,
            "delta_E": delta_E,
            "max_force": max_force,
        }

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

        if self.write_results:
            with open(self.results_file, "w", encoding="utf8") as out:
                print("#Barrier [eV] | delta E [eV] | Max force [eV/Å] ", file=out)
                print(*self.results.values(), file=out)

        return self.results
