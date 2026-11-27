"""Tool functions exposing all janus-core calculation and training modes for MCP."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, get_args

from ase import Atoms
from ase.io import read as ase_read
import numpy as np

from janus_core.helpers.janus_types import (
    Architectures,
    Devices,
    Ensembles,
    EoSNames,
    PhononCalcs,
    Properties,
)

JanusMode = Literal[
    "singlepoint",
    "geomopt",
    "md",
    "phonons",
    "eos",
    "elasticity",
    "neb",
    "descriptors",
    "train",
    "preprocess",
]

JANUS_MODES: dict[JanusMode, str] = {
    "singlepoint": (
        "Calculate static single-point energy, forces, stress, or Hessian."
    ),
    "geomopt": ("Perform geometry optimization and optional unit cell relaxation."),
    "md": (
        "Run molecular dynamics simulations across NVE, NVT, NPH, and NPT ensembles."
    ),
    "phonons": (
        "Calculate phonon force constants, band structures, DOS, PDOS, and thermal "
        "properties."
    ),
    "eos": "Calculate equation of state (E-V fit, bulk modulus, equilibrium volume).",
    "elasticity": "Calculate the 6x6 elasticity stiffness tensor and elastic moduli.",
    "neb": "Run Nudged Elastic Band (NEB / CI-NEB) minimum energy pathway search.",
    "descriptors": "Extract MLIP invariant descriptors for atomic structures.",
    "train": "Train or fine-tune an MLIP from a YAML configuration file.",
    "preprocess": "Preprocess MLIP training datasets to HDF5 format.",
}


def _to_jsonable(value: Any) -> Any:
    """
    Recursively convert Python, NumPy, Path, and ASE objects to JSON-safe values.

    Parameters
    ----------
    value
        Arbitrary Python object returned by a janus-core calculation.

    Returns
    -------
    Any
        JSON-serializable representation of `value`.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Atoms):
        return {
            "formula": value.get_chemical_formula(),
            "n_atoms": len(value),
            "pbc": [bool(flag) for flag in value.pbc],
            "cell_lengths_angstrom": [float(x) for x in value.cell.lengths()],
            "cell_angles_deg": [float(x) for x in value.cell.angles()],
            "volume_angstrom3": float(value.get_volume()) if any(value.pbc) else None,
        }
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_to_jsonable(v) for v in value]
    return str(value)


def _merge_kwargs(
    base: dict[str, Any], extra_kwargs: dict[str, Any] | None
) -> dict[str, Any]:
    """
    Merge optional extra keyword arguments into a base dictionary.

    Parameters
    ----------
    base
        Base keyword arguments for a calculation.
    extra_kwargs
        Optional additional keyword arguments overriding or extending `base`.

    Returns
    -------
    dict[str, Any]
        Combined keyword argument dictionary.
    """
    merged = dict(base)
    if extra_kwargs:
        merged.update(extra_kwargs)
    return merged


def list_modes() -> dict[str, Any]:
    """
    List all supported janus-core calculation and training modes.

    Returns
    -------
    dict[str, Any]
        Dictionary mapping each mode name to its description, plus total count.
    """
    return {
        "modes": dict(JANUS_MODES),
        "count": len(JANUS_MODES),
    }


def list_architectures() -> dict[str, Any]:
    """
    List supported MLIP architectures, compute devices, MD ensembles, and EoS types.

    Returns
    -------
    dict[str, Any]
        Dictionary of supported architectures, devices, ensembles, properties,
        phonon calculations, and equation-of-state fits.
    """
    return {
        "architectures": list(get_args(Architectures)),
        "devices": list(get_args(Devices)),
        "ensembles": list(get_args(Ensembles)),
        "properties": list(get_args(Properties)),
        "phonon_calcs": list(get_args(PhononCalcs)),
        "eos_types": list(get_args(EoSNames)),
    }


def inspect_structure(struct: str, index: int | str = -1) -> dict[str, Any]:
    """
    Inspect an atomic structure file and return formula, cell, and PBC metadata.

    Parameters
    ----------
    struct
        Path to an ASE-readable structure file (e.g., CIF, XYZ, POSCAR, extxyz).
    index
        Frame index or slice string passed to `ase.io.read`. Default is -1.

    Returns
    -------
    dict[str, Any]
        Summary of structure composition, periodicity, lattice parameters, and volume.
    """
    atoms_or_list = ase_read(struct, index=index)
    if isinstance(atoms_or_list, list):
        return {
            "path": str(Path(struct)),
            "n_frames": len(atoms_or_list),
            "frames": [_to_jsonable(frame) for frame in atoms_or_list],
        }
    summary = _to_jsonable(atoms_or_list)
    summary["path"] = str(Path(struct))
    summary["elements"] = sorted(set(atoms_or_list.get_chemical_symbols()))
    return summary


def run_singlepoint(
    struct: str,
    arch: Architectures = "mace_mp",
    device: Devices = "cpu",
    model: str | None = None,
    properties: list[Properties] | None = None,
    write_results: bool = False,
    file_prefix: str | None = None,
    calc_kwargs: dict[str, Any] | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run a single-point calculation for an atomic structure.

    Parameters
    ----------
    struct
        Path to structure file(s) to evaluate.
    arch
        MLIP architecture name. Default is "mace_mp".
    device
        Compute device ("cpu", "cuda", "mps", "xpu"). Default is "cpu".
    model
        Optional model label or path to local model file. Default is None.
    properties
        Properties to calculate (e.g., ["energy", "forces", "stress", "hessian"]).
        Default is None (calculates energy, forces, and stress).
    write_results
        Whether to write results to disk. Default is False.
    file_prefix
        Prefix for output files. Default is None.
    calc_kwargs
        Extra keyword arguments for the MLIP calculator. Default is None.
    extra_kwargs
        Additional keyword arguments passed to `SinglePoint`. Default is None.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary of calculated properties.
    """
    from janus_core.calculations.single_point import SinglePoint

    kwargs = _merge_kwargs(
        {
            "struct": struct,
            "arch": arch,
            "device": device,
            "model": model,
            "properties": tuple(properties) if properties else (),
            "write_results": write_results,
            "file_prefix": file_prefix,
            "calc_kwargs": calc_kwargs,
            "attach_logger": False,
            "track_carbon": False,
            "enable_progress_bar": False,
        },
        extra_kwargs,
    )
    sp = SinglePoint(**kwargs)
    results = sp.run()
    return {
        "mode": "singlepoint",
        "arch": arch,
        "device": device,
        "results": _to_jsonable(results),
    }


def run_geomopt(
    struct: str,
    arch: Architectures = "mace_mp",
    device: Devices = "cpu",
    model: str | None = None,
    fmax: float = 0.1,
    steps: int = 1000,
    filter_class: str | None = "FrechetCellFilter",
    optimizer: str = "LBFGS",
    write_results: bool = True,
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run geometry optimization and optional unit cell relaxation.

    Parameters
    ----------
    struct
        Path to structure file to optimize.
    arch
        MLIP architecture name. Default is "mace_mp".
    device
        Compute device. Default is "cpu".
    model
        Optional model label or path to local model file. Default is None.
    fmax
        Force convergence threshold in eV/Å. Default is 0.1.
    steps
        Maximum optimization steps. Default is 1000.
    filter_class
        ASE cell filter class name (e.g., "FrechetCellFilter") or None for fixed-cell.
        Default is "FrechetCellFilter".
    optimizer
        ASE optimizer name (e.g., "LBFGS", "BFGS", "FIRE"). Default is "LBFGS".
    write_results
        Whether to save the optimized structure to disk. Default is True.
    extra_kwargs
        Additional keyword arguments passed to `GeomOpt`. Default is None.

    Returns
    -------
    dict[str, Any]
        Convergence status, final energy, structure summary, and output metadata.
    """
    from janus_core.calculations.geom_opt import GeomOpt

    kwargs = _merge_kwargs(
        {
            "struct": struct,
            "arch": arch,
            "device": device,
            "model": model,
            "fmax": fmax,
            "steps": steps,
            "filter_class": filter_class,
            "optimizer": optimizer,
            "write_results": write_results,
            "attach_logger": False,
            "track_carbon": False,
        },
        extra_kwargs,
    )
    opt = GeomOpt(**kwargs)
    opt.run()
    energy = float(opt.struct.get_potential_energy())
    return {
        "mode": "geomopt",
        "arch": arch,
        "device": device,
        "converged": bool(opt.converged),
        "energy_ev": energy,
        "structure": _to_jsonable(opt.struct),
    }


def run_md(
    struct: str,
    ensemble: Ensembles = "nvt",
    arch: Architectures = "mace_mp",
    device: Devices = "cpu",
    model: str | None = None,
    steps: int = 100,
    timestep: float = 1.0,
    temp: float = 300.0,
    pressure: float = 0.0,
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run a molecular dynamics simulation in the specified thermodynamic ensemble.

    Parameters
    ----------
    struct
        Path to initial structure file.
    ensemble
        Thermodynamic ensemble ("nve", "nvt", "nvt-nh", "nvt-csvr", "npt", "nph",
        "npt-mtk", "npt-mtk-iso", "npt-mtk-aniso"). Default is "nvt".
    arch
        MLIP architecture name. Default is "mace_mp".
    device
        Compute device. Default is "cpu".
    model
        Optional model label or path to local model file. Default is None.
    steps
        Number of MD integration steps. Default is 100.
    timestep
        Integration timestep in fs. Default is 1.0.
    temp
        Target temperature in K. Default is 300.0.
    pressure
        Target pressure in GPa (used for NPT/NPH ensembles). Default is 0.0.
    extra_kwargs
        Additional keyword arguments passed to the ensemble class. Default is None.

    Returns
    -------
    dict[str, Any]
        Simulation metadata, output files, and final structure summary.
    """
    from janus_core.calculations.md import (
        NPH,
        NPT,
        NPT_MTK,
        NVE,
        NVT,
        NVT_CSVR,
        NVT_NH,
    )

    base_kwargs: dict[str, Any] = {
        "struct": struct,
        "arch": arch,
        "device": device,
        "model": model,
        "steps": steps,
        "timestep": timestep,
        "temp": temp,
        "attach_logger": False,
        "track_carbon": False,
        "enable_progress_bar": False,
    }

    if ensemble in ("npt", "nph", "npt-mtk", "npt-mtk-iso", "npt-mtk-aniso"):
        base_kwargs["pressure"] = pressure

    kwargs = _merge_kwargs(base_kwargs, extra_kwargs)

    ensemble_map: dict[str, type] = {
        "nve": NVE,
        "nvt": NVT,
        "nvt-nh": NVT_NH,
        "nvt-csvr": NVT_CSVR,
        "npt": NPT,
        "nph": NPH,
    }

    if ensemble in ensemble_map:
        dyn = ensemble_map[ensemble](**kwargs)
    elif ensemble in ("npt-mtk", "npt-mtk-iso", "npt-mtk-aniso"):
        dyn = NPT_MTK(ensemble=ensemble, **kwargs)
    else:
        raise ValueError(f"Unsupported ensemble: {ensemble}")

    dyn.run()
    return {
        "mode": "md",
        "ensemble": ensemble,
        "arch": arch,
        "device": device,
        "steps": steps,
        "output_files": _to_jsonable(getattr(dyn, "output_files", {})),
        "structure": _to_jsonable(dyn.struct),
    }


def run_phonons(
    struct: str,
    arch: Architectures = "mace_mp",
    device: Devices = "cpu",
    model: str | None = None,
    supercell: int | list[int] = 2,
    calcs: list[PhononCalcs] | None = None,
    minimize: bool = False,
    write_results: bool = True,
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run phonon force constant, band structure, DOS, and thermal calculations.

    Parameters
    ----------
    struct
        Path to crystal structure file.
    arch
        MLIP architecture name. Default is "mace_mp".
    device
        Compute device. Default is "cpu".
    model
        Optional model label or path to local model file. Default is None.
    supercell
        Supercell size (int, 3-element diagonal, or 9-element matrix). Default is 2.
    calcs
        List of phonon sub-calculations ("bands", "dos", "pdos", "qpoints", "thermal").
        Default is None (force constants only).
    minimize
        Whether to optimize geometry before computing phonons. Default is False.
    write_results
        Whether to write phonon outputs to disk. Default is True.
    extra_kwargs
        Additional keyword arguments passed to `Phonons`. Default is None.

    Returns
    -------
    dict[str, Any]
        Phonon calculation metadata and available result keys.
    """
    from janus_core.calculations.phonons import Phonons

    kwargs = _merge_kwargs(
        {
            "struct": struct,
            "arch": arch,
            "device": device,
            "model": model,
            "supercell": supercell,
            "calcs": tuple(calcs) if calcs else (),
            "minimize": minimize,
            "write_results": write_results,
            "attach_logger": False,
            "track_carbon": False,
            "enable_progress_bar": False,
        },
        extra_kwargs,
    )
    phonons = Phonons(**kwargs)
    phonons.run()
    results_summary: dict[str, Any] = {}
    for key, val in phonons.results.items():
        if key == "phonon":
            results_summary[key] = "Phonopy object"
        elif key == "thermal" and isinstance(val, Mapping):
            results_summary[key] = _to_jsonable(val)
        else:
            results_summary[key] = type(val).__name__
    return {
        "mode": "phonons",
        "arch": arch,
        "device": device,
        "supercell": supercell,
        "calcs": list(calcs) if calcs else [],
        "results": results_summary,
    }


def run_eos(
    struct: str,
    arch: Architectures = "mace_mp",
    device: Devices = "cpu",
    model: str | None = None,
    min_volume: float = 0.95,
    max_volume: float = 1.05,
    n_volumes: int = 7,
    eos_type: EoSNames = "birchmurnaghan",
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Calculate the Equation of State (E-V curve and bulk modulus).

    Parameters
    ----------
    struct
        Path to crystal structure file.
    arch
        MLIP architecture name. Default is "mace_mp".
    device
        Compute device. Default is "cpu".
    model
        Optional model label or path to local model file. Default is None.
    min_volume
        Minimum volume scale factor. Default is 0.95.
    max_volume
        Maximum volume scale factor. Default is 1.05.
    n_volumes
        Number of volume points to sample. Default is 7.
    eos_type
        ASE EquationOfState fit type. Default is "birchmurnaghan".
    extra_kwargs
        Additional keyword arguments passed to `EoS`. Default is None.

    Returns
    -------
    dict[str, Any]
        Fitted bulk modulus (GPa), equilibrium volume (Å^3), energy (eV), and E-V data.
    """
    from janus_core.calculations.eos import EoS

    kwargs = _merge_kwargs(
        {
            "struct": struct,
            "arch": arch,
            "device": device,
            "model": model,
            "min_volume": min_volume,
            "max_volume": max_volume,
            "n_volumes": n_volumes,
            "eos_type": eos_type,
            "attach_logger": False,
            "track_carbon": False,
        },
        extra_kwargs,
    )
    eos = EoS(**kwargs)
    results = eos.run()
    return {
        "mode": "eos",
        "arch": arch,
        "device": device,
        "eos_type": eos_type,
        "bulk_modulus_gpa": _to_jsonable(results.get("bulk_modulus")),
        "v_0_angstrom3": _to_jsonable(results.get("v_0")),
        "e_0_ev": _to_jsonable(results.get("e_0")),
        "volumes": _to_jsonable(eos.volumes),
        "energies": _to_jsonable(eos.energies),
    }


def run_elasticity(
    struct: str,
    arch: Architectures = "mace_mp",
    device: Devices = "cpu",
    model: str | None = None,
    minimize: bool = True,
    n_strains: int = 4,
    write_results: bool = True,
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Calculate the 6x6 elasticity stiffness tensor and derived elastic moduli.

    Parameters
    ----------
    struct
        Path to crystal structure file.
    arch
        MLIP architecture name. Default is "mace_mp".
    device
        Compute device. Default is "cpu".
    model
        Optional model label or path to local model file. Default is None.
    minimize
        Whether to relax the initial structure first. Default is True.
    n_strains
        Number of normal and shear strains to apply. Default is 4.
    write_results
        Whether to write the elasticity tensor to disk. Default is True.
    extra_kwargs
        Additional keyword arguments passed to `Elasticity`. Default is None.

    Returns
    -------
    dict[str, Any]
        Voigt 6x6 elastic tensor (GPa) and Voigt/Reuss/VRH moduli when available.
    """
    from janus_core.calculations.elasticity import Elasticity

    kwargs = _merge_kwargs(
        {
            "struct": struct,
            "arch": arch,
            "device": device,
            "model": model,
            "minimize": minimize,
            "n_strains": n_strains,
            "write_results": write_results,
            "attach_logger": False,
            "track_carbon": False,
        },
        extra_kwargs,
    )
    elasticity = Elasticity(**kwargs)
    elasticity.run()
    tensor = elasticity.elastic_tensor
    return {
        "mode": "elasticity",
        "arch": arch,
        "device": device,
        "voigt_gpa": _to_jsonable(tensor.voigt),
        "k_vrh_gpa": float(tensor.k_vrh),
        "g_vrh_gpa": float(tensor.g_vrh),
        "y_mod_gpa": float(tensor.y_mod),
    }


def run_neb(
    init_struct: str | None = None,
    final_struct: str | None = None,
    neb_structs: str | None = None,
    arch: Architectures = "mace_mp",
    device: Devices = "cpu",
    model: str | None = None,
    n_images: int = 15,
    fmax: float = 0.1,
    steps: int = 100,
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run a Nudged Elastic Band (NEB) calculation to find a minimum energy pathway.

    Parameters
    ----------
    init_struct
        Path to initial endpoint structure. Default is None.
    final_struct
        Path to final endpoint structure. Default is None.
    neb_structs
        Path to pre-interpolated NEB band images. Default is None.
    arch
        MLIP architecture name. Default is "mace_mp".
    device
        Compute device. Default is "cpu".
    model
        Optional model label or path to local model file. Default is None.
    n_images
        Number of intermediate NEB images. Default is 15.
    fmax
        Force convergence threshold in eV/Å. Default is 0.1.
    steps
        Maximum NEB optimization steps. Default is 100.
    extra_kwargs
        Additional keyword arguments passed to `NEB`. Default is None.

    Returns
    -------
    dict[str, Any]
        NEB convergence status and barrier/energy results.
    """
    from janus_core.calculations.neb import NEB

    kwargs = _merge_kwargs(
        {
            "init_struct": init_struct,
            "final_struct": final_struct,
            "neb_structs": neb_structs,
            "arch": arch,
            "device": device,
            "model": model,
            "n_images": n_images,
            "fmax": fmax,
            "steps": steps,
            "attach_logger": False,
            "track_carbon": False,
        },
        extra_kwargs,
    )
    neb = NEB(**kwargs)
    results = neb.run()
    return {
        "mode": "neb",
        "arch": arch,
        "device": device,
        "converged": bool(neb.converged),
        "results": _to_jsonable(results),
    }


def run_descriptors(
    struct: str,
    arch: Architectures = "mace_mp",
    device: Devices = "cpu",
    model: str | None = None,
    invariants_only: bool = True,
    calc_per_element: bool = False,
    calc_per_atom: bool = False,
    write_results: bool = False,
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Calculate MLIP descriptors for atomic structure(s).

    Parameters
    ----------
    struct
        Path to structure file(s).
    arch
        MLIP architecture name. Default is "mace_mp".
    device
        Compute device. Default is "cpu".
    model
        Optional model label or path to local model file. Default is None.
    invariants_only
        Whether to return only invariant descriptors. Default is True.
    calc_per_element
        Whether to compute per-element mean descriptors. Default is False.
    calc_per_atom
        Whether to compute per-atom descriptors. Default is False.
    write_results
        Whether to save structures with descriptors to disk. Default is False.
    extra_kwargs
        Additional keyword arguments passed to `Descriptors`. Default is None.

    Returns
    -------
    dict[str, Any]
        Descriptor calculation metadata and extracted descriptor values.
    """
    from janus_core.calculations.descriptors import Descriptors

    kwargs = _merge_kwargs(
        {
            "struct": struct,
            "arch": arch,
            "device": device,
            "model": model,
            "invariants_only": invariants_only,
            "calc_per_element": calc_per_element,
            "calc_per_atom": calc_per_atom,
            "write_results": write_results,
            "attach_logger": False,
            "track_carbon": False,
            "enable_progress_bar": False,
        },
        extra_kwargs,
    )
    desc = Descriptors(**kwargs)
    desc.run()
    structs = desc.struct if isinstance(desc.struct, Sequence) else [desc.struct]
    descriptor_info = [
        {
            key: _to_jsonable(val)
            for key, val in atoms.info.items()
            if "descriptor" in key
        }
        for atoms in structs
    ]
    return {
        "mode": "descriptors",
        "arch": arch,
        "device": device,
        "n_structures": len(structs),
        "descriptors": descriptor_info,
    }


def run_train(
    arch: Architectures,
    mlip_config: str,
    file_prefix: str = "janus_train",
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Train or fine-tune an MLIP using a YAML configuration file.

    Parameters
    ----------
    arch
        MLIP architecture to train (e.g., "mace", "mace_mp", "nequip", "sevennet").
    mlip_config
        Path to the MLIP training YAML configuration file.
    file_prefix
        Output directory/prefix for training artifacts. Default is "janus_train".
    extra_kwargs
        Additional keyword arguments passed to `janus_core.training.train.train`.
        Default is None.

    Returns
    -------
    dict[str, Any]
        Training completion metadata.
    """
    from janus_core.training.train import train

    kwargs = _merge_kwargs(
        {
            "arch": arch,
            "mlip_config": Path(mlip_config),
            "file_prefix": Path(file_prefix),
            "attach_logger": False,
            "track_carbon": False,
        },
        extra_kwargs,
    )
    train(**kwargs)
    return {
        "mode": "train",
        "arch": arch,
        "mlip_config": str(Path(mlip_config)),
        "file_prefix": str(Path(file_prefix)),
        "status": "completed",
    }


def run_preprocess(
    mlip_config: str,
    req_file_keys: list[str] | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Convert and preprocess MLIP training datasets to HDF5 using a YAML config file.

    Parameters
    ----------
    mlip_config
        Path to the MLIP preprocessing YAML configuration file.
    req_file_keys
        List of required file keys in `mlip_config`. Default is
        ["train_file", "test_file", "valid_file"].
    extra_kwargs
        Additional keyword arguments passed to
        `janus_core.training.preprocess.preprocess`. Default is None.

    Returns
    -------
    dict[str, Any]
        Preprocessing completion metadata.
    """
    from janus_core.training.preprocess import preprocess

    kwargs = _merge_kwargs(
        {
            "mlip_config": Path(mlip_config),
            "req_file_keys": (
                tuple(req_file_keys)
                if req_file_keys is not None
                else ("train_file", "test_file", "valid_file")
            ),
            "attach_logger": False,
            "track_carbon": False,
        },
        extra_kwargs,
    )
    preprocess(**kwargs)
    return {
        "mode": "preprocess",
        "mlip_config": str(Path(mlip_config)),
        "status": "completed",
    }


MODE_HANDLERS: dict[JanusMode, Callable[..., dict[str, Any]]] = {
    "singlepoint": run_singlepoint,
    "geomopt": run_geomopt,
    "md": run_md,
    "phonons": run_phonons,
    "eos": run_eos,
    "elasticity": run_elasticity,
    "neb": run_neb,
    "descriptors": run_descriptors,
    "train": run_train,
    "preprocess": run_preprocess,
}


def run_mode(mode: JanusMode, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Run any janus-core calculation or training mode by name.

    Parameters
    ----------
    mode
        Name of the janus-core mode to execute (one of `JANUS_MODES`).
    params
        Keyword arguments forwarded to the corresponding mode handler. Default is None.

    Returns
    -------
    dict[str, Any]
        JSON-serializable result dictionary returned by the mode handler.
    """
    if mode not in MODE_HANDLERS:
        valid = ", ".join(sorted(MODE_HANDLERS))
        raise ValueError(f"Unsupported janus-core mode '{mode}'. Valid modes: {valid}")
    handler = MODE_HANDLERS[mode]
    return handler(**(params or {}))
