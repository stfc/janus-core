"""
Configure MLIP calculators.

Similar in spirit to matcalc and quacc approaches
- https://github.com/materialsvirtuallab/matcalc
- https://github.com/Quantum-Accelerators/quacc.git
"""

from __future__ import annotations

from os import environ
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

from ase.calculators.mixing import SumCalculator

from janus_core.helpers.janus_types import Architectures, Devices, PathLike
from janus_core.helpers.utils import none_to_dict

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    import torch


def _set_model(
    model: PathLike | None = None,
    kwargs: dict[str, Any] | None = None,
) -> PathLike | torch.nn.Module | None:
    """
    Set `model`.

    Parameters
    ----------
    model
        Path to MLIP file.
    kwargs
        Dictionary of additional keyword arguments passed to the selected calculator.

    Returns
    -------
    PathLike | torch.nn.Module | None
        Name of MLIP model, or path to MLIP model file or loaded model.
    """
    (kwargs,) = none_to_dict(kwargs)

    # kwargs that may be used for `model` for different MLIPs
    # Note: "model" for chgnet (but not mace_mp or mace_off) and "potential" may refer
    # to loaded PyTorch models
    model_kwargs = {"model_path", "model_paths", "potential", "path"}
    present = kwargs.keys() & model_kwargs

    # Use model if specified, but check not also specified via kwargs
    if model and present:
        raise ValueError(
            "`model` cannot be used in combination with 'model_path', "
            "'model_paths', 'potential', or 'path'"
        )
    if len(present) > 1:
        # Check at most one suitable kwarg is specified
        raise ValueError(
            "Only one of 'model_path', 'model_paths', 'potential', and 'path' can be "
            "specified"
        )
    if present:
        # Set model from kwargs if any are specified
        model = kwargs.pop(present.pop())

    # Convert to path if file/directory exists
    if isinstance(model, Path | str) and Path(model).expanduser().exists():
        return Path(model).expanduser()
    return model


def _set_no_weights_only_load():
    """Set environment variable to fix models for torch 2.6."""
    environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def choose_calculator(
    arch: Architectures,
    device: Devices = "cpu",
    model: PathLike | None = None,
    **kwargs,
) -> Calculator:
    """
    Choose MLIP calculator to configure.

    Parameters
    ----------
    arch
        MLIP architecture.
    device
        Device to run calculator on. Default is "cpu".
    model
        MLIP model label, path to model, or loaded model. Default is `None`.
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
    model = _set_model(model, kwargs)

    if device not in get_args(Devices):
        raise ValueError(f"`device` must be one of: {get_args(Devices)}")

    # Fix torch 2.6 (must be before MLIP modules are loaded)
    _set_no_weights_only_load()

    if arch == "mace":
        from mace import __version__
        from mace.calculators import MACECalculator

        # No default `model`
        if model is None:
            raise ValueError(
                f"Please specify `model`, as there is no default model for {arch}"
            )
        # Default to float64 precision
        kwargs.setdefault("default_dtype", "float64")

        calculator = MACECalculator(model_paths=model, device=device, **kwargs)

    elif arch == "mace_mp":
        from mace import __version__
        from mace.calculators import mace_mp

        # Default to "small" model and float64 precision
        model = model if model else "small"
        kwargs.setdefault("default_dtype", "float64")

        calculator = mace_mp(model=model, device=device, **kwargs)

    elif arch == "mace_off":
        from mace import __version__
        from mace.calculators import mace_off

        # Default to "small" model and float64 precision
        model = model if model else "small"
        kwargs.setdefault("default_dtype", "float64")

        calculator = mace_off(model=model, device=device, **kwargs)

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
        if isinstance(model, Potential):
            potential = model
            model = "loaded_Potential"
        elif isinstance(model, Path):
            if model.is_file():
                model = model.parent
            potential = load_model(model)
        elif isinstance(model, str):
            potential = load_model(model)
        else:
            model = "M3GNet-MP-2021.2.8-DIRECT-PES"
            potential = load_model(model)

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
        if isinstance(model, CHGNet):
            loaded_model = model
            model = "loaded_CHGNet"
        elif isinstance(model, Path):
            loaded_model = CHGNet.from_file(model)
        elif isinstance(model, str):
            loaded_model = CHGNet.load(model_name=model, use_device=device)
        else:
            model = __version__
            loaded_model = None

        calculator = CHGNetCalculator(model=loaded_model, use_device=device, **kwargs)

    elif arch == "alignn":
        from alignn import __version__
        from alignn.ff.ff import (
            AlignnAtomwiseCalculator,
            default_path,
            get_figshare_model_ff,
        )

        # Set default path to directory containing config and model location
        if isinstance(model, Path):
            if model.is_file():
                model = model.parent
        # If a string, assume referring to model_name e.g. "v5.27.2024"
        elif isinstance(model, str):
            model = get_figshare_model_ff(model_name=model)
        else:
            model = default_path()

        calculator = AlignnAtomwiseCalculator(path=model, device=device, **kwargs)

    elif arch == "sevennet":
        from sevenn import __version__
        from sevenn.sevennet_calculator import SevenNetCalculator
        import torch

        # Set before loading model to avoid type mismatches
        torch.set_default_dtype(torch.float32)

        if isinstance(model, Path):
            model = str(model)
        elif not isinstance(model, str):
            model = "SevenNet-0_11July2024"

        kwargs.setdefault("file_type", "checkpoint")
        kwargs.setdefault("sevennet_config", None)
        calculator = SevenNetCalculator(model=model, device=device, **kwargs)

    elif arch == "nequip":
        from nequip import __version__
        from nequip.ase import NequIPCalculator

        # No default `model`
        if model is None:
            raise ValueError(
                f"Please specify `model`, as there is no default model for {arch}"
            )

        model = str(model)

        calculator = NequIPCalculator.from_deployed_model(
            model_path=model, device=device, **kwargs
        )

    elif arch == "dpa3":
        from deepmd import __version__
        from deepmd.calculator import DP

        # No default `model`
        if model is None:
            # From https://matbench-discovery.materialsproject.org/models/dpa3-v1-mptrj
            raise ValueError(
                "Please specify `model`, as there is no "
                f"default model for {arch} "
                "e.g. https://bohrium-api.dp.tech/ds-dl/dpa3openlam-74ng-v3.zip"
            )

        model = str(model)

        calculator = DP(model=model, **kwargs)

    elif arch == "orb":
        from orb_models import __version__
        from orb_models.forcefield.calculator import ORBCalculator
        from orb_models.forcefield.direct_regressor import DirectForcefieldRegressor
        import orb_models.forcefield.pretrained as orb_ff

        # Default model
        model = model if model else "orb_v3_conservative_20_omat"

        if isinstance(model, DirectForcefieldRegressor):
            loaded_model = model
            model = "loaded_DirectForcefieldRegressor"
        else:
            try:
                loaded_model = getattr(orb_ff, model.replace("-", "_"))()
            except AttributeError as e:
                raise ValueError(
                    "`model` must be a `DirectForcefieldRegressor`, pre-trained "
                    "model label (e.g. 'orb-v2'), or `None` (uses default, orb-v2)"
                ) from e

        calculator = ORBCalculator(model=loaded_model, device=device, **kwargs)

    elif arch == "mattersim":
        from mattersim import __version__
        from mattersim.forcefield import MatterSimCalculator
        from torch.nn import Module

        # Default model
        model = model if model else "mattersim-v1.0.0-5M"

        if isinstance(model, Module):
            potential = model
            model = "loaded_Module"
        else:
            potential = None

        if isinstance(model, Path):
            model = str(model)

        calculator = MatterSimCalculator(
            potential=potential, load_path=model, device=device, **kwargs
        )

    elif arch == "grace":
        from tensorpotential.calculator import grace_fm

        __version__ = "0.5.1"

        # Default model
        model = model if model else "GRACE-2L-OMAT"

        if isinstance(model, Path):
            model = str(model)

        calculator = grace_fm(model, **kwargs)

    elif arch == "fairchem":
        from fairchem.core import OCPCalculator

        if isinstance(model_path, Path):
            model_path = str(model_path)
        elif not isinstance(model_path, str):
            model_path = "EquiformerV2-31M-S2EF-OC20-All+MD"
        kwargs.setdefault("local_cache", "pretrained_models")
        cpu = False
        if device == "cpu":
            cpu = True

        calculator = OCPCalculator(model_name=model_path, cpu=cpu, **kwargs)

    else:
        raise ValueError(
            f"Unrecognized {arch=}. Suported architectures "
            f"are {', '.join(Architectures.__args__)}"
        )

    if isinstance(model, Path):
        model = model.as_posix()
    calculator.parameters["version"] = __version__
    calculator.parameters["arch"] = arch
    calculator.parameters["model"] = str(model)

    return calculator


def check_calculator(calc: Calculator, attribute: str) -> None:
    """
    Ensure calculator has ability to calculate properties.

    If the calculator is a SumCalculator that inlcudes the TorchDFTD3Calculator, this
    also sets the relevant function so that the MLIP component of the calculator is
    used for properties unrelated to dispersion.

    Parameters
    ----------
    calc
        ASE Calculator to check.
    attribute
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
