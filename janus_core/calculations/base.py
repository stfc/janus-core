"""Prepare structures for MLIP calculations."""

from typing import Any, Optional

from ase import Atoms

from janus_core.helpers.janus_types import (
    Architectures,
    ASEReadArgs,
    Devices,
    MaybeSequence,
    PathLike,
)
from janus_core.helpers.log import config_logger, config_tracker
from janus_core.helpers.utils import FileNameMixin, input_structs, none_to_dict


class BaseCalculation(FileNameMixin):
    """
    Prepare structures for MLIP calculations.

    Parameters
    ----------
    struct : Optional[MaybeSequence[Atoms]]
        ASE Atoms structure(s) to simulate. Required if `struct_path` is None.
        Default is None.
    struct_path : Optional[PathLike]
        Path of structure to simulate. Required if `struct` is None.
        Default is None.
    arch : Architectures
        MLIP architecture to use for calculations. Default is "mace_mp".
    device : Devices
        Device to run model on. Default is "cpu".
    model_path : Optional[PathLike]
        Path to MLIP model. Default is `None`.
    read_kwargs : ASEReadArgs
        Keyword arguments to pass to ase.io.read. Default is {}.
    sequence_allowed : bool
        Whether a sequence of Atoms objects is allowed. Default is True.
    calc_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to the selected calculator. Default is {}.
    set_calc : Optional[bool]
        Whether to set (new) calculators for structures. Default is None.
    log_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to `config_logger`. Default is {}.
    tracker_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to `config_tracker`. Default is {}.

    Attributes
    ----------
    logger : Optional[logging.Logger]
        Logger if log file has been specified.
    tracker : Optional[OfflineEmissionsTracker]
        Tracker if logging is enabled.
    """

    def __init__(
        self,
        *,
        struct: Optional[MaybeSequence[Atoms]] = None,
        struct_path: Optional[PathLike] = None,
        arch: Architectures = "mace_mp",
        device: Devices = "cpu",
        model_path: Optional[PathLike] = None,
        read_kwargs: Optional[ASEReadArgs] = None,
        sequence_allowed: bool = True,
        calc_kwargs: Optional[dict[str, Any]] = None,
        set_calc: Optional[bool] = None,
        log_kwargs: Optional[dict[str, Any]] = None,
        tracker_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Read the structure being simulated and attach an MLIP calculator.

        Parameters
        ----------
        struct : Optional[MaybeSequence[Atoms]]
            ASE Atoms structure(s) to simulate. Required if `struct_path` is None.
            Default is None.
        struct_path : Optional[PathLike]
            Path of structure to simulate. Required if `struct` is None. Default is
            None.
        arch : Architectures
            MLIP architecture to use for calculations. Default is "mace_mp".
        device : Devices
            Device to run MLIP model on. Default is "cpu".
        model_path : Optional[PathLike]
            Path to MLIP model. Default is `None`.
        read_kwargs : Optional[ASEReadArgs]
            Keyword arguments to pass to ase.io.read. Default is {}.
        sequence_allowed : bool
            Whether a sequence of Atoms objects is allowed. Default is True.
        calc_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to the selected calculator. Default is {}.
        set_calc : Optional[bool]
            Whether to set (new) calculators for structures. Default is None.
        log_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to `config_logger`. Default is {}.
        tracker_kwargs : Optional[dict[str, Any]]
            Keyword arguments to pass to `config_tracker`. Default is {}.
        """
        (read_kwargs, calc_kwargs, log_kwargs, tracker_kwargs) = none_to_dict(
            (read_kwargs, calc_kwargs, log_kwargs, tracker_kwargs)
        )

        self.struct = struct
        self.struct_path = struct_path
        self.arch = arch
        self.device = device
        self.model_path = model_path
        self.read_kwargs = read_kwargs
        self.calc_kwargs = calc_kwargs

        if not self.model_path and "model_path" in self.calc_kwargs:
            raise ValueError("`model_path` must be passed explicitly")

        # Configure logging
        self.logger = config_logger(**log_kwargs)
        self.tracker = config_tracker(self.logger, **tracker_kwargs)

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
            logger=self.logger,
        )