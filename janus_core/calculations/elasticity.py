"""Elasticity Tensor."""

from __future__ import annotations

from collections.abc import Sequence
from copy import copy
from typing import Any

from ase import Atoms
from ase.units import GPa
from pymatgen.analysis.elasticity import ElasticTensor
from pymatgen.analysis.elasticity.strain import (
    DeformedStructureSet,
)
from pymatgen.analysis.elasticity.stress import Stress
from pymatgen.io.ase import AseAtomsAdaptor

from janus_core.calculations.base import BaseCalculation
from janus_core.calculations.geom_opt import GeomOpt
from janus_core.helpers.janus_types import (
    Architectures,
    ASEReadArgs,
    Devices,
    OutputKwargs,
    PathLike,
)
from janus_core.helpers.struct_io import output_structs
from janus_core.helpers.utils import build_file_dir, none_to_dict, set_minimize_logging


class Elasticity(BaseCalculation):
    """
    Calculate the elasticity tensor.

    Parameters
    ----------
    struct
        ASE Atoms structure(s), or filepath to structure(s) to simulate.
    arch
        MLIP architecture to use for calculations. Default is `None`.
    device
        Device to run model on. Default is "cpu".
    model
        MLIP model label, path to model, or loaded model. Default is `None`.
    model_path
        Deprecated. Please use `model`.
    read_kwargs
        Keyword arguments to pass to ase.io.read. Default is {}.
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
        Prefix for output filenames. Default is `None`.
    minimize
        Whether to optimize geometry for the initial structure.
        Default is True.
    minimize_all
        Whether to optimize geometry for all generated structures.
        Default is False.
    minimize_kwargs
        Keyword arguments to pass to optimize. Default is `None`.
    write_results
        Whether to write out the elasticity tensor. Default is True.
    write_structures
        Whether to write out all generated structures. Deault is False.
    write_kwargs
        Keyword arguments to pass to ase.io.write to save generated structures.
        Default is {}.
    write_voigt
        Whether to write out in Voigt notation, Default is True.
    shear_strains
        The shear strains to build the DeformedStructureSet.
        Default is (-0.06, -0.03, 0.03, 0.06).
    normal_strains
        The normal strains to build the DeformedStructureSet.
        Default is (-0.01, -0.005, 0.005, 0.01).

    Attributes
    ----------
    results: ElasticityResults
        Dictionary containing the elasticity tensor and the derived
        bulk and shear moduli.
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
        minimize: bool = True,
        minimize_all: bool = False,
        minimize_kwargs: dict[str, Any] | None = None,
        write_results: bool = True,
        write_structures: bool = False,
        write_kwargs: OutputKwargs | None = None,
        write_voigt: bool = True,
        shear_strains: Sequence[float] = (-0.06, -0.03, 0.03, 0.06),
        normal_strains: Sequence[float] = (-0.01, -0.005, 0.005, 0.01),
    ) -> None:
        """
        Initialise class.

        Parameters
        ----------
        struct
            ASE Atoms structure(s), or filepath to structure(s) to simulate.
        arch
            MLIP architecture to use for calculations. Default is `None`.
        device
            Device to run model on. Default is "cpu".
        model
            MLIP model label, path to model, or loaded model. Default is `None`.
        model_path
            Deprecated. Please use `model`.
        read_kwargs
            Keyword arguments to pass to ase.io.read. Default is {}.
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
            Prefix for output filenames. Default is `None`.
        minimize
            Whether to optimize geometry for the initial structure.
            Default is True.
        minimize_all
            Whether to optimize geometry for all generated structures.
            Default is False.
        minimize_kwargs
            Keyword arguments to pass to optimize. Default is `None`.
        write_results
            Whether to write out the elasticity tensor. Default is True.
        write_structures
            Whether to write out all generated structures. Deault is False.
        write_kwargs
            Keyword arguments to pass to ase.io.write to save generated structures.
            Default is {}.
        write_voigt
            Whether to write out in Voigt notation, Default is True.
        shear_strains
            The shear strains to build the DeformedStructureSet.
            Default is (-0.06, -0.03, 0.03, 0.06).
        normal_strains
            The normal strains to build the DeformedStructureSet.
            Default is (-0.01, -0.005, 0.005, 0.01).
        """
        read_kwargs, minimize_kwargs, write_kwargs = none_to_dict(
            read_kwargs, minimize_kwargs, write_kwargs
        )

        self.minimize = minimize
        self.minimize_all = minimize_all
        self.minimize_kwargs = minimize_kwargs
        self.write_results = write_results
        self.write_structures = write_structures
        self.write_kwargs = write_kwargs
        self.write_voigt = write_voigt
        self.normal_strains = normal_strains
        self.shear_strains = shear_strains

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

        self.elasticity_file = self._build_filename("elastic_tensor.dat")

        self.results = {}

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
            "elasticity": self.elasticity_file if self.write_results else None,
        }

    def run(self) -> ElasticTensor:
        """
        Calculate the elasticity tensor.

        Returns
        -------
        ElasticityResults
            Dictionary containing the ElasticTensor and derived values of
            the shear and bulk moduli.
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
                config_type="elasticity",
            )

        self._calculate_elasticity()

        if self.write_results:
            build_file_dir(self.elasticity_file)
            with open(self.elasticity_file, "w", encoding="utf8") as out:
                print(
                    "# Bulk modulus (Reuss) [GPa] |"
                    " Bulk modulus (Voigt) [GPa] |"
                    " Bulk modulus (VRH) [GPa] |"
                    " Shear modulus (Reuss) [GPa] |"
                    " Shear modulus (Voigt) [GPa] |"
                    " Shear modulus (VRH) [GPa] |"
                    " Young's modulus [GPa] |"
                    " Universal anisotropy |"
                    " Homogeneous Poisson ratio |"
                    " Elastic constants (row-major) [GPa]",
                    file=out,
                )
                values = [
                    self.elastic_tensor.property_dict[prop]
                    for prop in (
                        "k_reuss",
                        "k_voigt",
                        "k_vrh",
                        "g_reuss",
                        "g_voigt",
                        "g_vrh",
                        "y_mod",
                        "universal_anisotropy",
                        "homogeneous_poisson",
                    )
                ]

                vals = (
                    self.elastic_tensor.voigt
                    if self.write_voigt
                    else self.elastic_tensor
                )
                for cijkl in vals.flatten():
                    values.append(cijkl)
                print(" ".join(map(str, values)), file=out)
        return self.elastic_tensor

    def _calculate_elasticity(self) -> None:
        """Generate deformed structures and calculate the elasticity."""
        if self.logger:
            self.logger.info("Starting structure calculations")
        if self.tracker:
            self.tracker.start_task("Calculate configurations")

        self.deformed_structure_set = DeformedStructureSet(
            AseAtomsAdaptor.get_structure(self.struct),
            norm_strains=self.normal_strains,
            shear_strains=self.shear_strains,
        )

        self.stresses = []
        self.strains = [
            i.green_lagrange_strain for i in self.deformed_structure_set.deformations
        ]
        self.deformed_structures = [
            AseAtomsAdaptor.get_atoms(struct) for struct in self.deformed_structure_set
        ]
        for i, deformed_structure in enumerate(self.deformed_structures):
            deformed_structure.calc = copy(self.struct.calc)
            progress = f"{i} / {len(self.deformed_structures)}"
            if self.minimize_all:
                if self.logger:
                    self.logger.info(
                        "Calculating stress for deformed structure " + progress
                    )
                optimizer = GeomOpt(deformed_structure, **self.minimize_kwargs)
                optimizer.run()

            if self.logger:
                self.logger.info(
                    "Calculating stress for deformed structure " + progress,
                )

            # Always append first original structure
            self.write_kwargs["append"] = True
            # Write structures, but no need to set info c_struct is not used elsewhere
            output_structs(
                images=deformed_structure,
                struct_path=self.struct_path,
                write_results=self.write_structures,
                set_info=False,
                write_kwargs=self.write_kwargs,
                config_type="elasticity",
            )

            self.stresses.append(deformed_structure.get_stress(voigt=False) / GPa)

        if self.logger:
            self.logger.info("Calculating stress for initial structure")

        self.eq_stress = self.struct.get_stress(voigt=False) / GPa

        self.elastic_tensor = ElasticTensor.from_independent_strains(
            self.strains, [Stress(s) for s in self.stresses], self.eq_stress
        )

        if self.logger:
            self.logger.info("Structure calculations complete.")
        if self.tracker:
            emissions = self.tracker.stop_task().emissions
            self.struct.info["emissions"] = emissions
