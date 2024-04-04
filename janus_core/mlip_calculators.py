"""
Configure MLIP calculators.

Similar in spirit to matcalc and quacc approaches
- https://github.com/materialsvirtuallab/matcalc
- https://github.com/Quantum-Accelerators/quacc.git
"""

from ase.calculators.calculator import Calculator

from .janus_types import Architectures, Devices


def choose_calculator(
    architecture: Architectures = "mace",
    device: Devices = "cpu",
    **kwargs,
) -> Calculator:
    """
    Choose MLIP calculator to configure.

    Parameters
    ----------
    architecture : Architectures, optional
        MLIP architecture. Default is "mace".
    device : Devices
        Device to run calculator on. Default is "cpu".
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
    # pylint: disable=import-outside-toplevel, too-many-branches, import-error
    # Optional imports handled via `architecture`. We could catch these,
    # but the error message is clear if imports are missing.
    if "model" in kwargs and "model_paths" in kwargs:
        raise ValueError("Please specify either `model` or `model_paths`")

    if architecture == "mace":
        from mace import __version__
        from mace.calculators import MACECalculator

        # `model_paths` is keyword for path to model, so take from kwargs if specified
        # Otherwise, take `model` if specified, then default to `""`, which will
        # raise a ValueError
        kwargs.setdefault("model_paths", kwargs.pop("model", ""))
        kwargs.setdefault("default_dtype", "float64")
        calculator = MACECalculator(device=device, **kwargs)

    elif architecture == "mace_mp":
        from mace import __version__
        from mace.calculators import mace_mp

        # `model` is keyword for path to model, so take from kwargs if specified
        # Otherwise, take `model_paths` if specified, then default to "small"
        kwargs.setdefault("model", kwargs.pop("model_paths", "small"))
        kwargs.setdefault("default_dtype", "float64")
        calculator = mace_mp(**kwargs)

    elif architecture == "mace_off":
        from mace import __version__
        from mace.calculators import mace_off

        # `model` is keyword for path to model, so take from kwargs if specified
        # Otherwise, take `model_paths` if specified, then default to "small"
        kwargs.setdefault("model", kwargs.pop("model_paths", "small"))
        kwargs.setdefault("default_dtype", "float64")
        calculator = mace_off(**kwargs)

    elif architecture == "m3gnet":
        from matgl import __version__, load_model
        from matgl.ext.ase import M3GNetCalculator

        model = kwargs.setdefault("model", load_model("M3GNet-MP-2021.2.8-DIRECT-PES"))
        kwargs.setdefault("stress_weight", 1.0 / 160.21766208)
        calculator = M3GNetCalculator(potential=model, **kwargs)

    elif architecture == "chgnet":
        from chgnet import __version__
        from chgnet.model.dynamics import CHGNetCalculator

        calculator = CHGNetCalculator(use_device=device, **kwargs)

    else:
        raise ValueError(
            f"Unrecognized {architecture=}. Suported architectures "
            f"are {', '.join(Architectures.__args__)}"
        )

    calculator.parameters["version"] = __version__

    return calculator
