"""Configure MLIP calculators.

Similar in spirit with matcalc and quacc approaches
- https://github.com/materialsvirtuallab/matcalc
- https://github.com/Quantum-Accelerators/quacc.git
"""

from typing import Literal

from ase.calculators.calculator import Calculator

architectures = ["mace", "mace_mp", "mace_off", "m3gnet", "chgnet"]
devices = ["cpu", "cuda", "mps"]


def choose_calculator(
    architecture: Literal[architectures] = "mace",
    device: Literal[devices] = "cuda",
    **kwargs,
) -> Calculator:
    """
    Choose MLIP calculator to configure.

    Parameters
    ----------
    architecture : Literal[architectures], optional
        MLIP architecture. Default is "mace".
    device : Literal[devices]
        Device to run calculator on. Default is "cuda".
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
    if architecture == "mace":
        from mace import __version__
        from mace.calculators import MACECalculator

        kwargs.setdefault("default_dtype", "float64")
        calculator = MACECalculator(device=device, **kwargs)

    elif architecture == "mace_mp":
        from mace import __version__
        from mace.calculators import mace_mp

        kwargs.setdefault("default_dtype", "float64")
        kwargs.setdefault("model", "small")
        calculator = mace_mp(**kwargs)

    elif architecture == "mace_off":
        from mace import __version__
        from mace.calculators import mace_off

        kwargs.setdefault("default_dtype", "float64")
        kwargs.setdefault("model", "small")
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
            f"Unrecognized {architecture=}. Suported architectures are {architectures}"
        )

    calculator.parameters["version"] = __version__

    return calculator
