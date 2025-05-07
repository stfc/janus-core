"""Prepare structures for MLIP calculations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from warnings import warn

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
    "target_temperature": "K",
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
    struct
        ASE Atoms structure(s), or filepath to structure(s) to simulate.
    calc_name
        Name of calculation being run, used for name of logger. Default is "base".
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
    sequence_allowed
        Whether a sequence of Atoms objects is allowed. Default is True.
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
    additional_prefix
        Component to add to default file_prefix (joined by hyphens). Default is `None`.
    param_prefix
        Additional parameters to add to default file_prefix. Default is `None`.

    Attributes
    ----------
    logger : logging.Logger | None
        Logger if log file has been specified.
    tracker : OfflineEmissionsTracker | None
        Tracker if logging is enabled.
    """

    def __init__(
        self,
        struct: MaybeSequence[Atoms] | PathLike,
        *,
        calc_name: str = "base",
        arch: Architectures | None = None,
        device: Devices = "cpu",
        model: PathLike | None = None,
        model_path: PathLike | None = None,
        read_kwargs: ASEReadArgs | None = None,
        sequence_allowed: bool = True,
        calc_kwargs: dict[str, Any] | None = None,
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
        struct
            ASE Atoms structure(s), or filepath to structure(s) to simulate.
        calc_name
            Name of calculation being run, used for name of logger. Default is "base".
        arch
            MLIP architecture to use for calculations. Default is `None`.
        device
            Device to run MLIP model on. Default is "cpu".
        model
            MLIP model label, path to model, or loaded model. Default is `None`.
        model_path
            Deprecated. Please use `model`.
        read_kwargs
            Keyword arguments to pass to ase.io.read. Default is {}.
        sequence_allowed
            Whether a sequence of Atoms objects is allowed. Default is True.
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
        additional_prefix
            Component to add to default file_prefix (joined by hyphens). Default is
            None.
        param_prefix
            Additional parameters to add to default file_prefix. Default is `None`.
        """
        read_kwargs, calc_kwargs, log_kwargs, tracker_kwargs = none_to_dict(
            read_kwargs, calc_kwargs, log_kwargs, tracker_kwargs
        )

        self.arch = arch
        self.device = device
        self.model = model
        self.read_kwargs = read_kwargs
        self.calc_kwargs = calc_kwargs
        self.log_kwargs = log_kwargs
        self.tracker_kwargs = tracker_kwargs

        attach_logger, self.track_carbon = set_log_tracker(
            attach_logger, log_kwargs, track_carbon
        )

        # Set model from deprecated model_path (warn later, after logging is set up)
        if model_path:
            # `model`` is a new parameter, so there is no reason to be using both
            if model:
                raise ValueError(
                    "`model` has replaced `model_path`. Please only use `model`"
                )
            self.model = model_path

        # Disallow `model_path` in kwargs
        if not self.model and "model_path" in self.calc_kwargs:
            raise ValueError("`model` must be passed explicitly")

        # Disallow `model` in kwargs if `model` is used
        # Raise warning after logging is set up
        raise_model_warning = False
        if "model" in self.calc_kwargs:
            if model:
                raise ValueError("`model must be passed explicitly")
            self.model = self.calc_kwargs.pop("model")
            raise_model_warning = True

        # Read structures and/or attach calculators
        # Note: logger not set up so yet so not passed here
        self.struct, self.struct_path = input_structs(
            struct=struct,
            read_kwargs=self.read_kwargs,
            sequence_allowed=sequence_allowed,
            arch=self.arch,
            device=self.device,
            model=self.model,
            calc_kwargs=self.calc_kwargs,
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
            )

        self.log_kwargs.setdefault("name", calc_name)
        self.logger = config_logger(**self.log_kwargs)
        self.tracker = config_tracker(
            self.logger, self.track_carbon, **self.tracker_kwargs
        )

        # Warn now that logging is set up
        if model_path:
            warn(
                "`model_path` has been deprecated. Please use `model`.",
                FutureWarning,
                stacklevel=2,
            )

        if raise_model_warning:
            warn(
                "Please pass `model` explicitly.",
                FutureWarning,
                stacklevel=2,
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
