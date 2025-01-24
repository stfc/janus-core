"""Prepare structures for MLIP calculations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ase import Atoms

from janus_core.helpers.janus_types import (
    Architectures,
    ASEReadArgs,
    Devices,
    MaybeSequence,
    PathLike,
)
from janus_core.helpers.log import config_logger, config_tracker
from janus_core.helpers.struct_io import input_structs
from janus_core.helpers.utils import FileNameMixin, none_to_dict, set_log_tracker

UNITS = {
    "energy": "eV",
    "forces": "ev/Ang",
    "stress": "ev/Ang^3",
    "hessian": "ev/Ang^2",
    "time": "fs",
    "real_time": "s",
    "temperature": "K",
    "pressure": "GPa",
    "momenta": "(eV*u)^0.5",
    "density": "g/cm^3",
    "volume": "Ang^3",
}


class BaseCalculation(FileNameMixin):
    """
    Prepare structures for MLIP calculations.

    Parameters
    ----------
    calc_name
        Name of calculation being run, used for name of logger. Default is "base".
    struct
        ASE Atoms structure(s) to simulate. Required if `struct_path` is None.
        Default is None.
    struct_path
        Path of structure to simulate. Required if `struct` is None.
        Default is None.
    arch
        MLIP architecture to use for calculations. Default is "mace_mp".
    device
        Device to run model on. Default is "cpu".
    model_path
        Path to MLIP model. Default is `None`.
    read_kwargs
        Keyword arguments to pass to ase.io.read. Default is {}.
    sequence_allowed
        Whether a sequence of Atoms objects is allowed. Default is True.
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
    file_prefix
        Prefix for output filenames. Default is None.
    additional_prefix
        Component to add to default file_prefix (joined by hyphens). Default is None.
    param_prefix
        Additional parameters to add to default file_prefix. Default is None.

    Attributes
    ----------
    logger : logging.Logger | None
        Logger if log file has been specified.
    tracker : OfflineEmissionsTracker | None
        Tracker if logging is enabled.
    """

    def __init__(
        self,
        *,
        calc_name: str = "base",
        struct: MaybeSequence[Atoms] | None = None,
        struct_path: PathLike | None = None,
        arch: Architectures = "mace_mp",
        device: Devices = "cpu",
        model_path: PathLike | None = None,
        read_kwargs: ASEReadArgs | None = None,
        sequence_allowed: bool = True,
        calc_kwargs: dict[str, Any] | None = None,
        set_calc: bool | None = None,
        attach_logger: bool | None = None,
        log_kwargs: dict[str, Any] | None = None,
        track_carbon: bool | None = None,
        tracker_kwargs: dict[str, Any] | None = None,
        file_prefix: PathLike | None = None,
        additional_prefix: str | None = None,
        param_prefix: str | None = None,
    ) -> None:
        """
        Read the structure being simulated and attach an MLIP calculator.

        Parameters
        ----------
        calc_name
            Name of calculation being run, used for name of logger. Default is "base".
        struct
            ASE Atoms structure(s) to simulate. Required if `struct_path` is None.
            Default is None.
        struct_path
            Path of structure to simulate. Required if `struct` is None. Default is
            None.
        arch
            MLIP architecture to use for calculations. Default is "mace_mp".
        device
            Device to run MLIP model on. Default is "cpu".
        model_path
            Path to MLIP model. Default is `None`.
        read_kwargs
            Keyword arguments to pass to ase.io.read. Default is {}.
        sequence_allowed
            Whether a sequence of Atoms objects is allowed. Default is True.
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
        file_prefix
            Prefix for output filenames. Default is None.
        additional_prefix
            Component to add to default file_prefix (joined by hyphens). Default is
            None.
        param_prefix
            Additional parameters to add to default file_prefix. Default is None.
        """
        read_kwargs, calc_kwargs, log_kwargs, tracker_kwargs = none_to_dict(
            read_kwargs, calc_kwargs, log_kwargs, tracker_kwargs
        )

        self.struct = struct
        self.struct_path = struct_path
        self.arch = arch
        self.device = device
        self.model_path = model_path
        self.read_kwargs = read_kwargs
        self.calc_kwargs = calc_kwargs
        self.log_kwargs = log_kwargs
        self.tracker_kwargs = tracker_kwargs

        if not self.model_path and "model_path" in self.calc_kwargs:
            raise ValueError("`model_path` must be passed explicitly")

        attach_logger, self.track_carbon = set_log_tracker(
            attach_logger, log_kwargs, track_carbon
        )

        # Read structures and/or attach calculators
        # Note: logger not set up so yet so not passed here
        self.struct = input_structs(
            struct=self.struct,
            struct_path=self.struct_path,
            read_kwargs=self.read_kwargs,
            sequence_allowed=sequence_allowed,
            arch=self.arch,
            device=self.device,
            model_path=self.model_path,
            calc_kwargs=self.calc_kwargs,
            set_calc=set_calc,
        )

        # Set architecture to match calculator architecture
        if isinstance(self.struct, Sequence):
            if all(
                image.calc and "arch" in image.calc.parameters for image in self.struct
            ):
                self.arch = self.struct[0].calc.parameters["arch"]
        else:
            if self.struct.calc and "arch" in self.struct.calc.parameters:
                self.arch = self.struct.calc.parameters["arch"]

        FileNameMixin.__init__(
            self,
            self.struct,
            self.struct_path,
            file_prefix,
            additional_prefix,
        )

        # Configure logging
        # Extract command from module
        # e.g janus_core.calculations.single_point -> singlepoint
        log_suffix = f"{calc_name.split('.')[-1].replace('_', '')}-log.yml"
        if attach_logger:
            # Use _build_filename even if given filename to ensure directory exists
            self.log_kwargs.setdefault("filename", None)
            self.log_kwargs["filename"] = self._build_filename(
                log_suffix,
                param_prefix if file_prefix is None else "",
                filename=self.log_kwargs["filename"],
            ).absolute()

        self.log_kwargs.setdefault("name", calc_name)
        self.logger = config_logger(**self.log_kwargs)
        self.tracker = config_tracker(
            self.logger, self.track_carbon, **self.tracker_kwargs
        )

    def _set_info_units(
        self, keys: Sequence[str] = ("energy", "forces", "stress")
    ) -> None:
        """
        Save units to structure info.

        Parameters
        ----------
        keys
            Keys for which to add units to structure info. Default is
            ("energy", "forces", "stress").
        """
        if isinstance(self.struct, Sequence):
            for image in self.struct:
                image.info["units"] = {key: UNITS[key] for key in keys}
        else:
            self.struct.info["units"] = {key: UNITS[key] for key in keys}
