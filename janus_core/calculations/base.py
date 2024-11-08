"""Prepare structures for MLIP calculations."""

from __future__ import annotations

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
from janus_core.helpers.utils import FileNameMixin, none_to_dict


class BaseCalculation(FileNameMixin):
    """
    Prepare structures for MLIP calculations.

    Parameters
    ----------
    calc_name : str
        Name of calculation being run, used for name of logger. Default is "base".
    struct : MaybeSequence[Atoms] | None
        ASE Atoms structure(s) to simulate. Required if `struct_path` is None.
        Default is None.
    struct_path : PathLike | None
        Path of structure to simulate. Required if `struct` is None.
        Default is None.
    arch : Architectures
        MLIP architecture to use for calculations. Default is "mace_mp".
    device : Devices
        Device to run model on. Default is "cpu".
    model_path : PathLike | None
        Path to MLIP model. Default is `None`.
    read_kwargs : ASEReadArgs
        Keyword arguments to pass to ase.io.read. Default is {}.
    sequence_allowed : bool
        Whether a sequence of Atoms objects is allowed. Default is True.
    calc_kwargs : dict[str, Any] | None
        Keyword arguments to pass to the selected calculator. Default is {}.
    set_calc : bool | None
        Whether to set (new) calculators for structures. Default is None.
    attach_logger : bool
        Whether to attach a logger. Default is False.
    log_kwargs : dict[str, Any] | None
            Keyword arguments to pass to `config_logger`. Default is {}.
    track_carbon : bool
        Whether to track carbon emissions of calculation. Default is True.
    tracker_kwargs : dict[str, Any] | None
            Keyword arguments to pass to `config_tracker`. Default is {}.
    file_prefix : PathLike | None
        Prefix for output filenames. Default is None.
    additional_prefix : str | None
        Component to add to default file_prefix (joined by hyphens). Default is None.
    param_prefix : str | None
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
        attach_logger: bool = False,
        log_kwargs: dict[str, Any] | None = None,
        track_carbon: bool = True,
        tracker_kwargs: dict[str, Any] | None = None,
        file_prefix: PathLike | None = None,
        additional_prefix: str | None = None,
        param_prefix: str | None = None,
    ) -> None:
        """
        Read the structure being simulated and attach an MLIP calculator.

        Parameters
        ----------
        calc_name : str
            Name of calculation being run, used for name of logger. Default is "base".
        struct : MaybeSequence[Atoms] | None
            ASE Atoms structure(s) to simulate. Required if `struct_path` is None.
            Default is None.
        struct_path : PathLike | None
            Path of structure to simulate. Required if `struct` is None. Default is
            None.
        arch : Architectures
            MLIP architecture to use for calculations. Default is "mace_mp".
        device : Devices
            Device to run MLIP model on. Default is "cpu".
        model_path : PathLike | None
            Path to MLIP model. Default is `None`.
        read_kwargs : ASEReadArgs | None
            Keyword arguments to pass to ase.io.read. Default is {}.
        sequence_allowed : bool
            Whether a sequence of Atoms objects is allowed. Default is True.
        calc_kwargs : dict[str, Any] | None
            Keyword arguments to pass to the selected calculator. Default is {}.
        set_calc : bool | None
            Whether to set (new) calculators for structures. Default is None.
        attach_logger : bool
            Whether to attach a logger. Default is False.
        log_kwargs : dict[str, Any] | None
            Keyword arguments to pass to `config_logger`. Default is {}.
        track_carbon : bool
            Whether to track carbon emissions of calculation. Default is True.
        tracker_kwargs : dict[str, Any] | None
            Keyword arguments to pass to `config_tracker`. Default is {}.
        file_prefix : PathLike | None
            Prefix for output filenames. Default is None.
        additional_prefix : str | None
            Component to add to default file_prefix (joined by hyphens). Default is
            None.
        param_prefix : str | None
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
        self.track_carbon = track_carbon
        self.tracker_kwargs = tracker_kwargs

        if not self.model_path and "model_path" in self.calc_kwargs:
            raise ValueError("`model_path` must be passed explicitly")

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
