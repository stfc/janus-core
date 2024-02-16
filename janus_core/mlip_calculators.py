"""Configure MLIP calculators.

Similar in spirit with matcalc and quacc approaches
- https://github.com/materialsvirtuallab/matcalc
- https://github.com/Quantum-Accelerators/quacc.git
"""

from __future__ import annotations

from typing import Literal

from ase.calculators.calculator import Calculator

architectures = ["mace", "mace_mp", "mace_off", "m3gnet", "chgnet"]


def choose_calculator(
    architecture: Literal[architectures] = "mace", **kwargs
) -> Calculator:
    """Choose MLIP calculator to configure.

    Parameters
    ----------
    architecture : Literal[architectures], optional
        MLIP architecture. Default is "mace".

    Raises
    ------
    ModuleNotFoundError
        MLIP module not correctly been installed.
    ValueError
        Invalid architecture specified.

    Returns
    -------
    calculator : Calculator
        Configured MLIP calculator.
    """
    # pylint: disable=import-outside-toplevel
    # pylint: disable=too-many-branches
    # pylint: disable=import-error
    # Optional imports handled via `architecture`. We could catch these,
    # but the error message is clear if imports are missing.
    if architecture == "mace":
        from mace import __version__
        from mace.calculators import MACECalculator

        if "default_dtype" not in kwargs:
            kwargs["default_dtype"] = "float64"
        if "device" not in kwargs:
            kwargs["device"] = "cuda"
        calculator = MACECalculator(**kwargs)

    elif architecture == "mace_mp":
        from mace import __version__
        from mace.calculators import mace_mp

        if "default_dtype" not in kwargs:
            kwargs["default_dtype"] = "float64"
        if "device" not in kwargs:
            kwargs["device"] = "cuda"
        if "model" not in kwargs:
            kwargs["model"] = "small"
        calculator = mace_mp(**kwargs)

    elif architecture == "mace_off":
        from mace import __version__
        from mace.calculators import mace_off

        if "default_dtype" not in kwargs:
            kwargs["default_dtype"] = "float64"
        if "device" not in kwargs:
            kwargs["device"] = "cuda"
        if "model" not in kwargs:
            kwargs["model"] = "small"
        calculator = mace_off(**kwargs)

    elif architecture == "m3gnet":
        from matgl import __version__, load_model
        from matgl.ext.ase import M3GNetCalculator

        if "model" not in kwargs:
            model = load_model("M3GNet-MP-2021.2.8-DIRECT-PES")
        if "stress_weight" not in kwargs:
            kwargs.setdefault("stress_weight", 1.0 / 160.21766208)
        calculator = M3GNetCalculator(potential=model, **kwargs)

    elif architecture == "chgnet":
        from chgnet import __version__
        from chgnet.model.dynamics import CHGNetCalculator

        calculator = CHGNetCalculator(**kwargs)

    else:
        raise ValueError(
            f"Unrecognized {architecture=}. Suported architectures are {architectures}"
        )

    calculator.parameters["version"] = __version__

    return calculator
