"""Test configuration of MLIP calculators."""

from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError, URLError

from ase import units
from ase.calculators.mixing import SumCalculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from numpy import isfinite
import pytest
import torch

from janus_core.calculations.single_point import SinglePoint
from tests.utils import chdir, read_atoms, skip_extras

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models"

DPA3_PATH = MODEL_PATH / "2025-01-10-dpa3-mptrj.pth"
MACE_PATH = MODEL_PATH / "mace_mp_small.model"
NEQUIP_PATH = MODEL_PATH / "toluene.nequip.pth"
PET_MAD_CHECKPOINT = (
    "https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-mad-s-v1.1.0.ckpt"
)
SEVENNET_PATH = MODEL_PATH / "sevennet_0.pth"
UMA_LABEL = "uma-s-1p1"

test_data = [
    ("benzene.xyz", -76.0605725422795, "energy", "energy", {}, None),
    (
        "benzene.xyz",
        -76.06057739257812,
        ["energy"],
        "energy",
        {"default_dtype": "float32"},
        None,
    ),
    ("benzene.xyz", -0.0360169762840179, ["forces"], "forces", {}, [0, 1]),
    ("NaCl.cif", -0.004783275999053424, ["stress"], "stress", {}, [0]),
]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "struct, expected, properties, prop_key, calc_kwargs, idx", test_data
)
def test_potential_energy(
    struct, expected, properties, prop_key, calc_kwargs, idx, device
):
    """Test single point energy using MACE calculators."""
    skip_extras("mace")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    single_point = SinglePoint(
        struct=DATA_PATH / struct,
        arch="mace",
        model=MACE_PATH,
        calc_kwargs=calc_kwargs,
        properties=properties,
        device=device,
    )
    results = single_point.run()[prop_key]

    # Check correct values returned
    if idx is not None:
        if len(idx) == 1:
            assert results[idx[0]] == pytest.approx(expected)
        elif len(idx) == 2:
            assert results[idx[0], idx[1]] == pytest.approx(expected)
        else:
            raise ValueError(f"Invalid index: {idx}")
    else:
        assert results == pytest.approx(expected)
        results_2 = single_point.struct.info["mace_energy"]
        assert results_2 == pytest.approx(expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "arch, expected_energy, struct, kwargs",
    [
        ("chgnet", -29.331436157226562, "NaCl.cif", {}),
        ("dpa3", -27.053507387638092, "NaCl.cif", {"model": DPA3_PATH}),
        (
            "fairchem",
            -27.10070295,
            "NaCl.cif",
            {"model": UMA_LABEL},
        ),
        ("grace", -27.081155042373453, "NaCl.cif", {}),
        ("mace_off", -2081.1209264240006, "H2O.cif", {}),
        ("mace_omol", -2079.8650795528843, "H2O.cif", {}),
        ("mattersim", -27.06208038330078, "NaCl.cif", {}),
        (
            "nequip",
            -169815.1282456301,
            "toluene.xyz",
            {"model": NEQUIP_PATH},
        ),
        ("orb", -27.08186149597168, "NaCl.cif", {}),
        ("orb", -27.089094161987305, "NaCl.cif", {"model": "orb-v2"}),
        ("upet", -30.168052673339844, "NaCl.cif", {}),
        (
            "upet",
            -27.47624969482422,
            "NaCl.cif",
            {"model": PET_MAD_CHECKPOINT},
        ),
        (
            "sevennet",
            -27.061979293823242,
            "NaCl.cif",
            {"model": SEVENNET_PATH},
        ),
        ("sevennet", -27.061979293823242, "NaCl.cif", {}),
        (
            "sevennet",
            -27.061979293823242,
            "NaCl.cif",
            {"model": "SevenNet-0_11July2024"},
        ),
    ],
)
def test_extras(arch, device, expected_energy, struct, kwargs):
    """Test single point energy using extra MLIP calculators."""
    skip_extras(arch)

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    try:
        single_point = SinglePoint(
            struct=DATA_PATH / struct,
            arch=arch,
            device=device,
            properties="energy",
            **kwargs,
        )
        energy = single_point.run()["energy"]
        assert energy == pytest.approx(expected_energy, rel=1e-3)
    except HTTPError as err:  # Inherits from URLError, so check first
        if "Service Unavailable" in err.msg or "Too Many Requests" in err.msg:
            pytest.skip("Model download failed")
        raise err
    except URLError as err:
        if "Connection timed out" in err.reason:
            pytest.skip("Model download failed")
        raise err
    except RuntimeError as err:
        if "Model download failed" in str(err):
            pytest.skip("Model download failed")
        raise err


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_single_point_none(device):
    """Test single point stress using MACE calculator."""
    skip_extras("mace")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    single_point = SinglePoint(
        struct=DATA_PATH / "NaCl.cif",
        arch="mace",
        model=MACE_PATH,
        device=device,
    )

    results = single_point.run()
    for prop in ["energy", "forces", "stress"]:
        assert prop in results


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_single_point_clean(device):
    """Test single point stress using MACE calculator."""
    skip_extras("mace")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    single_point = SinglePoint(
        struct=DATA_PATH / "H2O.cif",
        arch="mace",
        model=MACE_PATH,
        device=device,
    )

    results = single_point.run()
    for prop in ["energy", "forces"]:
        assert prop in results
    assert "mace_stress" not in results


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_single_point_traj(device):
    """Test single point stress using MACE calculator."""
    skip_extras("mace")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    single_point = SinglePoint(
        struct=DATA_PATH / "benzene-traj.xyz",
        arch="mace",
        model=MACE_PATH,
        properties="energy",
        device=device,
    )

    assert len(single_point.struct) == 2
    results = single_point.run()
    assert results["energy"][0] == pytest.approx(-76.0605725422795)
    assert results["energy"][1] == pytest.approx(-74.80419118083256)
    assert single_point.struct[0].info["mace_energy"] == pytest.approx(
        -76.0605725422795
    )
    assert single_point.struct[1].info["mace_energy"] == pytest.approx(
        -74.80419118083256
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_single_point_write(tmp_path, device):
    """Test writing singlepoint results."""
    skip_extras("mace")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    with chdir(tmp_path):
        data_path = DATA_PATH / "NaCl.cif"
        results_dir = Path("janus_results")
        results_path = results_dir / "NaCl-results.extxyz"

        single_point = SinglePoint(
            struct=data_path,
            arch="mace",
            model=MACE_PATH,
            write_results=True,
            device=device,
        )
        assert "mace_forces" not in single_point.struct.arrays

        single_point.run()
        atoms = read_atoms(results_path)
        assert "mace_forces" in atoms.arrays
        assert atoms.info["mace_energy"] == pytest.approx(-27.035127799332745)
        assert "mace_stress" in atoms.info
        assert atoms.info["mace_stress"] == pytest.approx(
            [
                -0.004783275999053391,
                -0.004783275999053417,
                -0.004783275999053412,
                -2.3858882876234007e-19,
                -5.02032761017409e-19,
                -2.29070171362209e-19,
            ]
        )
        assert atoms.arrays["mace_forces"][0] == pytest.approx(
            [4.11996826e-18, 1.79977561e-17, 1.80139537e-17]
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_single_point_write_kwargs(tmp_path, device):
    """Test passing write_kwargs to singlepoint results."""
    skip_extras("mace")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    data_path = DATA_PATH / "NaCl.cif"
    results_path = tmp_path / "NaCl.extxyz"

    single_point = SinglePoint(
        struct=data_path,
        arch="mace",
        model=MACE_PATH,
        write_results=True,
        write_kwargs={"filename": results_path},
        device=device,
    )
    assert "mace_forces" not in single_point.struct.arrays

    single_point.run()
    atoms = read(results_path)
    assert atoms.info["mace_energy"] is not None
    assert "mace_forces" in atoms.arrays


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_single_point_molecule(tmp_path, device):
    """Test singlepoint results for isolated molecule."""
    skip_extras("mace")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    data_path = DATA_PATH / "H2O.cif"
    results_path = tmp_path / "H2O.extxyz"
    single_point = SinglePoint(
        struct=data_path,
        arch="mace",
        model=MACE_PATH,
        properties="energy",
        device=device,
    )

    assert isfinite(single_point.run()["energy"]).all()

    single_point.write_results = True
    single_point.write_kwargs = {"filename": results_path}
    single_point.properties = None
    single_point.run()

    atoms = read(results_path)
    assert atoms.info["mace_energy"] == pytest.approx(-14.035236305927514)
    assert "mace_forces" in atoms.arrays
    assert "mace_stress" in atoms.info
    assert atoms.info["mace_stress"] == pytest.approx(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-5
    )


def test_invalid_prop():
    """Test invalid property request."""
    skip_extras("mace")

    with pytest.raises(NotImplementedError):
        SinglePoint(
            struct=DATA_PATH / "H2O.cif",
            arch="mace",
            model=MACE_PATH,
            properties="invalid",
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_atoms(device):
    """Test passing ASE Atoms structure."""
    skip_extras("mace")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    struct = read(DATA_PATH / "NaCl.cif")
    single_point = SinglePoint(
        struct=struct,
        arch="mace",
        model=MACE_PATH,
        properties="energy",
        device=device,
    )
    assert single_point.run()["energy"] < 0


def test_no_atoms_or_path():
    """Test passing neither ASE Atoms structure nor structure path."""
    with pytest.raises(TypeError):
        SinglePoint(
            arch="mace",
            model=MACE_PATH,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_invalidate_calc(device):
    """Test setting invalidate_calc via write_kwargs."""
    skip_extras("mace")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    struct = DATA_PATH / "NaCl.cif"

    single_point = SinglePoint(
        struct=struct,
        arch="mace",
        model=MACE_PATH,
        write_kwargs={"invalidate_calc": False},
        device=device,
    )

    single_point.run()
    assert "energy" in single_point.struct.calc.results

    single_point.write_kwargs = {"invalidate_calc": True}
    single_point.run()
    assert "energy" not in single_point.struct.calc.results


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_logging(tmp_path, device):
    """Test attaching logger to SinglePoint and emissions are saved to info."""
    skip_extras("mace")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    log_file = tmp_path / "sp.log"

    single_point = SinglePoint(
        struct=DATA_PATH / "NaCl.cif",
        arch="mace_mp",
        model=MACE_PATH,
        properties="energy",
        log_kwargs={"filename": log_file},
        device=device,
    )

    assert "emissions" not in single_point.struct.info

    single_point.run()

    assert log_file.exists()
    assert "emissions" in single_point.struct.info
    assert single_point.struct.info["emissions"] > 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_hessian(device):
    """Test Hessian."""
    skip_extras("mace")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    sp = SinglePoint(
        model=MACE_PATH,
        struct=DATA_PATH / "NaCl.cif",
        arch="mace_mp",
        properties="hessian",
        device=device,
    )
    results = sp.run()
    assert "hessian" in results
    assert results["hessian"].shape == (24, 8, 3)
    assert "mace_mp_hessian" in sp.struct.info


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_hessian_traj(device):
    """Test calculating Hessian for trajectory."""
    skip_extras("mace")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    sp = SinglePoint(
        model=MACE_PATH,
        struct=DATA_PATH / "benzene-traj.xyz",
        arch="mace_mp",
        properties="hessian",
        device=device,
    )
    results = sp.run()
    assert "hessian" in results
    assert len(results["hessian"]) == 2
    assert results["hessian"][0].shape == (36, 12, 3)
    assert results["hessian"][1].shape == (36, 12, 3)
    assert "mace_mp_hessian" in sp.struct[0].info
    assert "mace_mp_hessian" in sp.struct[1].info


@pytest.mark.parametrize("struct", ["NaCl.cif", "benzene-traj.xyz"])
def test_hessian_not_implemented(struct):
    """Test unimplemented Hessian."""
    skip_extras("chgnet")

    with pytest.raises(NotImplementedError):
        SinglePoint(
            struct=DATA_PATH / struct,
            arch="chgnet",
            properties="hessian",
        )


def test_invalid_model_calc_kwargs():
    """Test error is raised when model is passed via calc_kwargs."""
    skip_extras("mace")

    with pytest.raises(ValueError):
        SinglePoint(
            arch="mace_mp",
            calc_kwargs={"model": MACE_PATH},
            struct=DATA_PATH / "NaCl.cif",
        )


def test_fake_calc_error():
    """Test an error is raised if SinglePointCalculator is set."""
    struct = read(DATA_PATH / "NaCl-results.extxyz")
    assert isinstance(struct.calc, SinglePointCalculator)

    with pytest.raises(ValueError):
        SinglePoint(struct=struct)


@pytest.mark.parametrize(
    "struct", (DATA_PATH / "NaCl.cif", read(DATA_PATH / "NaCl.cif"))
)
def test_missing_arch(struct):
    """Test missing arch."""
    skip_extras("mace")

    with pytest.raises(ValueError, match="A calculator must be attached"):
        SinglePoint(struct=struct)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "arch, kwargs, pred",
    [
        (
            "mace_mp",
            {"damping": "zero", "xc": "pbe", "cutoff": 95 * units.Bohr},
            -0.08281749,
        ),
        ("mace_off", {}, -0.08281747),
        ("mace_omol", {}, -0.08281747),
        ("mattersim", {}, -0.08281749),
        ("sevennet", {}, -0.08281749),
    ],
)
def test_dispersion(arch, kwargs, pred, device):
    """Test dispersion correction."""
    skip_extras(arch)
    pytest.importorskip("torch_dftd")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    try:
        data_path = DATA_PATH / "benzene.xyz"
        sp_no_d3 = SinglePoint(
            struct=data_path,
            arch=arch,
            properties="energy",
            calc_kwargs={"dispersion": False},
            device=device,
        )
        assert not isinstance(sp_no_d3.struct.calc, SumCalculator)
        no_d3_results = sp_no_d3.run()

        sp_d3 = SinglePoint(
            struct=data_path,
            arch=arch,
            properties="energy",
            calc_kwargs={"dispersion": True, "dispersion_kwargs": {**kwargs}},
            device=device,
        )
        assert isinstance(sp_d3.struct.calc, SumCalculator)
        d3_results = sp_d3.run()

        assert (d3_results["energy"] - no_d3_results["energy"]) == pytest.approx(
            pred, rel=1e-5
        )
    except RuntimeError as err:
        if "Model download failed" in str(err):
            pytest.skip("Model download failed")
        raise err


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mace_mp_dispersion(device):
    """Test mace_mp dispersion correction matches default."""
    skip_extras("mace_mp")
    pytest.importorskip("torch_dftd")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from mace.calculators import mace_mp

    data_path = DATA_PATH / "benzene.xyz"

    no_d3_energy = SinglePoint(
        struct=data_path,
        arch="mace_mp",
        properties="energy",
        calc_kwargs={"dispersion": False},
        device=device,
    ).run()["energy"]

    d3_energy = SinglePoint(
        struct=data_path,
        arch="mace_mp",
        properties="energy",
        calc_kwargs={"dispersion": True},
        device=device,
    ).run()["energy"]

    struct = read(data_path)
    struct.calc = mace_mp(model="small", dispersion=True, device=device)

    mace_d3_energy = SinglePoint(
        struct=struct,
        properties="energy",
        calc_kwargs={"dispersion": False},
    ).run()["energy"]

    # Different default to other architectures
    assert d3_energy - no_d3_energy == pytest.approx(-0.29815768)
    assert d3_energy == pytest.approx(mace_d3_energy)
