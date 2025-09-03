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

from ase import units
from ase.calculators.mixing import SumCalculator
from torch import get_default_dtype

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
    model_kwargs = {
        "model_path",
        "model_paths",
        "potential",
        "path",
        "model_name",
        "checkpoint_path",
        "predict_unit",
    }
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


def add_dispersion(
    calc: Calculator,
    device: Devices = "cpu",
    dtype: torch.dtype | None = None,
    **kwargs,
) -> SumCalculator:
    """
    Add D3 dispersion calculator to existing calculator.

    Parameters
    ----------
    calc
        Calculator to add D3 correction to.
    device
        Device to run calculator on. Default is "cpu".
    dtype
        Calculation precision. Default is current torch dtype.
    **kwargs
        Additional keyword arguments passed to `TorchDFTD3Calculator`.

    Returns
    -------
    SumCalculator
        Configured calculator with D3 dispersion correction added.
    """
    try:
        from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
    except ImportError as err:
        raise ImportError("Please install the d3 extra.") from err

    dtype = dtype if dtype else get_default_dtype()

    d3_calc = TorchDFTD3Calculator(
        device=device,
        dtype=dtype,
        **kwargs,
    )
    sum_calc = SumCalculator([calc, d3_calc])

    # Copy calculator parameters to make more accessible
    sum_calc.parameters = calc.parameters
    if "arch" in sum_calc.parameters:
        sum_calc.parameters["arch"] = sum_calc.parameters["arch"] + "_d3"

    return sum_calc


def choose_calculator(
    arch: Architectures,
    device: Devices = "cpu",
    model: PathLike | None = None,
    dispersion: bool = False,
    dispersion_kwargs: dict[str, Any] | None = None,
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
    dispersion
        Whether to add D3 dispersion.
    dispersion_kwargs
        Additional keyword arguments for `TorchDFTD3Calculator`. Defaults for mace_mp
        are taken from mace_mp's defaults.
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
    dispersion_kwargs = dispersion_kwargs if dispersion_kwargs else {}

    model = _set_model(model, kwargs)

    if device not in get_args(Devices):
        raise ValueError(f"`device` must be one of: {get_args(Devices)}")

    # Fix torch 2.6 (must be before MLIP modules are loaded)
    _set_no_weights_only_load()

    match arch:
        case "mace":
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

        case "mace_mp":
            from mace import __version__
            from mace.calculators import mace_mp

            # Default to "small" model and float64 precision
            model = model if model else "small"
            kwargs.setdefault("default_dtype", "float64")

            # Set mace_mp dispersion defaults
            dispersion_kwargs.setdefault("damping", kwargs.pop("damping", "bj"))
            dispersion_kwargs.setdefault("xc", kwargs.pop("dispersion_xc", "pbe"))
            dispersion_kwargs.setdefault(
                "cutoff", kwargs.pop("dispersion_cutoff", 40.0 * units.Bohr)
            )

            calculator = mace_mp(model=model, device=device, **kwargs)

        case "mace_off":
            from mace import __version__
            from mace.calculators import mace_off

            # Default to "small" model and float64 precision
            model = model if model else "small"
            kwargs.setdefault("default_dtype", "float64")

            calculator = mace_off(model=model, device=device, **kwargs)

        case "mace_omol":
            from mace import __version__
            from mace.calculators import mace_omol

            # Default to "extra_large" model and float64 precision
            model = model if model else "extra_large"
            kwargs.setdefault("default_dtype", "float64")

            calculator = mace_omol(model=model, device=device, **kwargs)

        case "m3gnet":
            from matgl import __version__, load_model
            from matgl.apps.pes import Potential
            from matgl.ext.ase import M3GNetCalculator
            import torch

            # Set before loading model to avoid type mismatches
            torch.set_default_dtype(torch.float32)
            kwargs.setdefault("stress_weight", 1.0 / 160.21766208)

            # Use potential (from kwargs) if specified
            # Otherwise, load the model if given a path, else use a default model
            match model:
                case Potential():
                    potential = model
                    model = "loaded_Potential"
                case Path():
                    if model.is_file():
                        model = model.parent
                    potential = load_model(model)
                case str():
                    potential = load_model(model)
                case _:
                    model = "M3GNet-MP-2021.2.8-DIRECT-PES"
                    potential = load_model(model)

            calculator = M3GNetCalculator(potential=potential, **kwargs)

        case "chgnet":
            from chgnet import __version__
            from chgnet.model.dynamics import CHGNetCalculator
            from chgnet.model.model import CHGNet
            import torch

            # Set before loading to avoid type mismatches
            torch.set_default_dtype(torch.float32)

            # Use loaded model (from kwargs) if specified
            # Otherwise, load the model if given a path, else use a default model
            match model:
                case CHGNet():
                    loaded_model = model
                    model = "loaded_CHGNet"
                case Path():
                    loaded_model = CHGNet.from_file(model)
                case str():
                    loaded_model = CHGNet.load(model_name=model, use_device=device)
                case _:
                    model = __version__
                    loaded_model = None

            calculator = CHGNetCalculator(
                model=loaded_model, use_device=device, **kwargs
            )

        case "alignn":
            from alignn import __version__
            from alignn.ff.ff import (
                AlignnAtomwiseCalculator,
                default_path,
                get_figshare_model_ff,
            )

            # Set default path to directory containing config and model location
            match model:
                case Path():
                    if model.is_file():
                        model = model.parent
                # If a string, assume referring to model_name e.g. "v5.27.2024"
                case str():
                    model = get_figshare_model_ff(model_name=model)
                case _:
                    model = default_path()

            calculator = AlignnAtomwiseCalculator(path=model, device=device, **kwargs)

        case "sevennet":
            from sevenn import __version__
            from sevenn.sevennet_calculator import SevenNetCalculator
            import torch

            # Set before loading model to avoid type mismatches
            torch.set_default_dtype(torch.float32)

            match model:
                case Path() | str():
                    model = str(model)
                case _:
                    model = "SevenNet-0_11July2024"

            kwargs.setdefault("file_type", "checkpoint")
            kwargs.setdefault("sevennet_config", None)
            calculator = SevenNetCalculator(model=model, device=device, **kwargs)

        case "nequip":
            from nequip import __version__
            from nequip.ase import NequIPCalculator

            # No default `model`
            if model is None:
                raise ValueError(
                    f"Please specify `model`, as there is no default model for {arch}"
                )

            model = str(model)

            calculator = NequIPCalculator.from_compiled_model(
                compile_path=model, device=device, **kwargs
            )

        case "dpa3":
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

        case "orb":
            from orb_models import __version__
            from orb_models.forcefield.calculator import ORBCalculator
            from orb_models.forcefield.direct_regressor import DirectForcefieldRegressor
            import orb_models.forcefield.pretrained as orb_ff

            match model:
                case DirectForcefieldRegressor():
                    loaded_model = model
                    model = "loaded_DirectForcefieldRegressor"
                case str() if hasattr(orb_ff, model.replace("-", "_")):
                    loaded_model = getattr(orb_ff, model.replace("-", "_"))()
                case None:
                    # Default model
                    model = "orb_v3_conservative_20_omat"
                    loaded_model = getattr(orb_ff, model)()
                case _:
                    raise ValueError(
                        "`model` must be a `DirectForcefieldRegressor`, pre-trained "
                        "model label (e.g. 'orb-v2'), or `None` (uses default, orb-v2)"
                    )

            calculator = ORBCalculator(model=loaded_model, device=device, **kwargs)

        case "mattersim":
            from mattersim import __version__
            from mattersim.forcefield import MatterSimCalculator
            from torch.nn import Module

            potential = None
            match model:
                case Module():
                    potential = model
                    model = "loaded_Module"
                case Path() | str():
                    model = str(model)
                case None:
                    model = "mattersim-v1.0.0-5M"

            calculator = MatterSimCalculator(
                potential=potential, load_path=model, device=device, **kwargs
            )

        case "grace":
            from tensorpotential.calculator import grace_fm

            __version__ = "0.5.1"

            # Default model
            model = model if model else "GRACE-2L-OMAT"

            if isinstance(model, Path):
                model = str(model)

            calculator = grace_fm(model, **kwargs)

        case "equiformer" | "esen":
            from fairchem.core import OCPCalculator, __version__

            match arch, model:
                case ("equiformer", None):
                    model = "EquiformerV2-31M-S2EF-OC20-All+MD"
                case ("esen", None):
                    model = "eSEN-30M-OMAT24"
                case _:
                    pass

            model_name = None
            checkpoint_path = None

            if isinstance(model, Path) and model.exists():
                checkpoint_path = str(model)
            else:
                model_name = str(model)

            kwargs.setdefault("local_cache", Path("~/.cache/fairchem").expanduser())
            cpu = True if device == "cpu" else False

            calculator = OCPCalculator(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                cpu=cpu,
                **kwargs,
            )

        case "pet_mad":
            from pet_mad import __version__
            from pet_mad._version import LATEST_VERSION
            from pet_mad.calculator import PETMADCalculator

            calculator = PETMADCalculator(
                checkpoint_path=model, device=device, **kwargs
            )

            if model is None:
                model = LATEST_VERSION

        case "uma":
            from fairchem.core import FAIRChemCalculator, __version__, pretrained_mlip
            from fairchem.core.units.mlip_unit import MLIPPredictUnit

            match model:
                case MLIPPredictUnit():
                    predict_unit = model
                    model = "loaded_Module"
                case Path() | str():
                    predict_unit = pretrained_mlip.get_predict_unit(
                        model_name=model, device=device
                    )
                case None:
                    model = "uma-m-1p1"
                    predict_unit = pretrained_mlip.get_predict_unit(
                        model_name=model, device=device
                    )

            kwargs.setdefault("task_name", "omat")

            calculator = FAIRChemCalculator(
                predict_unit=predict_unit,
                **kwargs,
            )

        case _:
            raise ValueError(
                f"Unrecognized {arch=}. Suported architectures "
                f"are {', '.join(Architectures.__args__)}"
            )

    if isinstance(model, Path):
        model = model.as_posix()
    calculator.parameters["version"] = __version__
    calculator.parameters["arch"] = arch
    calculator.parameters["model"] = str(model)

    if dispersion:
        return add_dispersion(calc=calculator, device=device, **dispersion_kwargs)

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
