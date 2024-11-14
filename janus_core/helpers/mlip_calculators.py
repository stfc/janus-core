"""
Configure MLIP calculators.

Similar in spirit to matcalc and quacc approaches
- https://github.com/materialsvirtuallab/matcalc
- https://github.com/Quantum-Accelerators/quacc.git
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

from ase.calculators.mixing import SumCalculator

from janus_core.helpers.janus_types import Architectures, Devices, PathLike
from janus_core.helpers.utils import none_to_dict

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    import torch


def _set_model_path(
    model_path: PathLike | None = None,
    kwargs: dict[str, Any] | None = None,
) -> PathLike | torch.nn.Module | None:
    """
    Set `model_path`.

    Parameters
    ----------
    model_path : PathLike | None
        Path to MLIP file.
    kwargs : dict[str, Any] | None
        Dictionary of additional keyword arguments passed to the selected calculator.

    Returns
    -------
    PathLike | torch.nn.Module | None
        Path to MLIP model file, loaded model, or None.
    """
    (kwargs,) = none_to_dict(kwargs)

    # kwargs that may be used for `model_path`` for different MLIPs
    # Note: "model" for chgnet (but not mace_mp or mace_off) and "potential" may refer
    # to loaded PyTorch models
    present = kwargs.keys() & {"model", "model_paths", "potential", "path"}

    # Use model_path if specified, but check not also specified via kwargs
    if model_path and present:
        raise ValueError(
            "`model_path` cannot be used in combination with 'model', "
            "'model_paths', 'potential', or 'path'"
        )
    if len(present) > 1:
        # Check at most one suitable kwarg is specified
        raise ValueError(
            "Only one of 'model', 'model_paths', 'potential', and 'path' can be "
            "specified"
        )
    if present:
        # Set model_path from kwargs if any are specified
        model_path = kwargs.pop(present.pop())

    # Convert to path if file/directory exists
    if isinstance(model_path, (Path, str)) and Path(model_path).expanduser().exists():
        return Path(model_path).expanduser()
    return model_path


def choose_calculator(
    arch: Architectures = "mace",
    device: Devices = "cpu",
    model_path: PathLike | None = None,
    **kwargs,
) -> Calculator:
    """
    Choose MLIP calculator to configure.

    Parameters
    ----------
    arch : Architectures
        MLIP architecture. Default is "mace".
    device : Devices
        Device to run calculator on. Default is "cpu".
    model_path : PathLike | None
        Path to MLIP file.
    **kwargs
        Additional keyword arguments passed to the selected calculator.

    Returns
    -------
    Calculator
        Configured MLIP calculator.

    Raises
    ------
    ModuleNotFoundError
        MLIP module not correctly been installed.
    ValueError
        Invalid architecture specified.
    """
    model_path = _set_model_path(model_path, kwargs)

    if device not in get_args(Devices):
        raise ValueError(f"`device` must be one of: {get_args(Devices)}")

    if arch == "mace":
        from mace import __version__
        from mace.calculators import MACECalculator

        # No default `model_path`
        if model_path is None:
            raise ValueError(
                "Please specify `model_path`, as there is no "
                f"default model for {arch}"
            )
        # Default to float64 precision
        kwargs.setdefault("default_dtype", "float64")

        calculator = MACECalculator(model_paths=model_path, device=device, **kwargs)

    elif arch == "mace_mp":
        from mace import __version__
        from mace.calculators import mace_mp

        # Default to "small" model and float64 precision
        model_path = model_path if model_path else "small"
        kwargs.setdefault("default_dtype", "float64")

        calculator = mace_mp(model=model_path, device=device, **kwargs)

    elif arch == "mace_off":
        from mace import __version__
        from mace.calculators import mace_off

        # Default to "small" model and float64 precision
        model_path = model_path if model_path else "small"
        kwargs.setdefault("default_dtype", "float64")

        calculator = mace_off(model=model_path, device=device, **kwargs)

    elif arch == "m3gnet":
        from matgl import __version__, load_model
        from matgl.apps.pes import Potential
        from matgl.ext.ase import M3GNetCalculator
        import torch

        # Set before loading model to avoid type mismatches
        torch.set_default_dtype(torch.float32)
        kwargs.setdefault("stress_weight", 1.0 / 160.21766208)

        # Use potential (from kwargs) if specified
        # Otherwise, load the model if given a path, else use a default model
        if isinstance(model_path, Potential):
            potential = model_path
            model_path = "loaded_Potential"
        elif isinstance(model_path, Path):
            if model_path.is_file():
                model_path = model_path.parent
            potential = load_model(model_path)
        elif isinstance(model_path, str):
            potential = load_model(model_path)
        else:
            model_path = "M3GNet-MP-2021.2.8-DIRECT-PES"
            potential = load_model(model_path)

        calculator = M3GNetCalculator(potential=potential, **kwargs)

    elif arch == "chgnet":
        from chgnet import __version__
        from chgnet.model.dynamics import CHGNetCalculator
        from chgnet.model.model import CHGNet
        import torch

        # Set before loading to avoid type mismatches
        torch.set_default_dtype(torch.float32)

        # Use loaded model (from kwargs) if specified
        # Otherwise, load the model if given a path, else use a default model
        if isinstance(model_path, CHGNet):
            model = model_path
            model_path = "loaded_CHGNet"
        elif isinstance(model_path, Path):
            model = CHGNet.from_file(model_path)
        elif isinstance(model_path, str):
            model = CHGNet.load(model_name=model_path, use_device=device)
        else:
            model_path = "0.3.0"
            model = None

        calculator = CHGNetCalculator(model=model, use_device=device, **kwargs)

    elif arch == "alignn":
        from alignn import __version__
        from alignn.ff.ff import (
            AlignnAtomwiseCalculator,
            default_path,
            get_figshare_model_ff,
        )

        # Set default path to directory containing config and model location
        if isinstance(model_path, Path):
            if model_path.is_file():
                model_path = model_path.parent
        # If a string, assume referring to model_name e.g. "v5.27.2024"
        elif isinstance(model_path, str):
            model_path = get_figshare_model_ff(model_name=model_path)
        else:
            model_path = default_path()

        calculator = AlignnAtomwiseCalculator(path=model_path, device=device, **kwargs)

    elif arch == "sevennet":
        from sevenn import __version__
        from sevenn.sevennet_calculator import SevenNetCalculator

        if isinstance(model_path, Path):
            model_path = str(model_path)
        elif not isinstance(model_path, str):
            model_path = "SevenNet-0_11July2024"

        kwargs.setdefault("file_type", "checkpoint")
        kwargs.setdefault("sevennet_config", None)
        calculator = SevenNetCalculator(model=model_path, device=device, **kwargs)

    else:
        raise ValueError(
            f"Unrecognized {arch=}. Suported architectures "
            f"are {', '.join(Architectures.__args__)}"
        )

    calculator.parameters["version"] = __version__
    calculator.parameters["arch"] = arch
    calculator.parameters["model_path"] = str(model_path)

    return calculator


def check_calculator(calc: Calculator, attribute: str) -> None:
    """
    Ensure calculator has ability to calculate properties.

    If the calculator is a SumCalculator that inlcudes the TorchDFTD3Calculator, this
    also sets the relevant function so that the MLIP component of the calculator is
    used for properties unrelated to dispersion.

    Parameters
    ----------
    calc : Calculator
        ASE Calculator to check.
    attribute : str
        Attribute to check calculator for.
    """
    # If dispersion added to MLIP calculator, use only MLIP calculator for calculation
    if (
        isinstance(calc, SumCalculator)
        and len(calc.mixer.calcs) == 2
        and calc.mixer.calcs[1].name == "TorchDFTD3Calculator"
        and hasattr(calc.mixer.calcs[0], attribute)
    ):
        setattr(calc, attribute, getattr(calc.mixer.calcs[0], attribute))

    if not hasattr(calc, attribute) or not callable(getattr(calc, attribute)):
        raise NotImplementedError(
            f"The attached calculator does not currently support {attribute}"
        )
