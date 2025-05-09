"""Test configuration of MLIP calculators."""

from __future__ import annotations

from pathlib import Path
import shutil
from urllib.error import URLError

from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from huggingface_hub.utils._auth import get_token
from numpy import isfinite
import pytest

from janus_core.calculations.single_point import SinglePoint
from tests.utils import read_atoms, skip_extras

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models"

ALIGNN_PATH = MODEL_PATH / "v5.27.2024"
DPA3_PATH = MODEL_PATH / "2025-01-10-dpa3-mptrj.pth"
FAIRCHEM_EQUIFORMER = "EquiformerV2-31M-S2EF-OC20-All+MD"
FAIRCHEM_ESEN = "eSEN-30M-OMAT24"
MACE_PATH = MODEL_PATH / "mace_mp_small.model"
NEQUIP_PATH = MODEL_PATH / "toluene.pth"
SEVENNET_PATH = MODEL_PATH / "sevennet_0.pth"

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


@pytest.mark.parametrize(
    "struct, expected, properties, prop_key, calc_kwargs, idx", test_data
)
def test_potential_energy(struct, expected, properties, prop_key, calc_kwargs, idx):
    """Test single point energy using MACE calculators."""
    skip_extras("mace")

    single_point = SinglePoint(
        struct=DATA_PATH / struct,
        arch="mace",
        model=MACE_PATH,
        calc_kwargs=calc_kwargs,
        properties=properties,
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


@pytest.mark.parametrize(
    "arch, device, expected_energy, struct, kwargs",
    [
        (
            "alignn",
            "cpu",
            -11.148092269897461,
            "NaCl.cif",
            {"model": ALIGNN_PATH / "best_model.pt"},
        ),
        (
            "alignn",
            "cpu",
            -11.148092269897461,
            "NaCl.cif",
            {"model": ALIGNN_PATH / "best_model.pt"},
        ),
        ("chgnet", "cpu", -29.331436157226562, "NaCl.cif", {}),
        ("dpa3", "cpu", -27.053507387638092, "NaCl.cif", {"model": DPA3_PATH}),
        ("dpa3", "cpu", -27.053507387638092, "NaCl.cif", {"model_path": DPA3_PATH}),
        (
            "fairchem",
            "cpu",
            -0.7482733,
            "NaCl.cif",
            {"model": FAIRCHEM_EQUIFORMER},
        ),
        (
            "fairchem",
            "cpu",
            -0.7482733,
            "NaCl.cif",
            {},
        ),
        (
            "fairchem",
            "cpu",
            -27.0977497,
            "NaCl.cif",
            {"model": FAIRCHEM_ESEN},
        ),
        ("grace", "cpu", -27.081155042373453, "NaCl.cif", {}),
        ("mattersim", "cpu", -27.06208038330078, "NaCl.cif", {}),
        ("m3gnet", "cpu", -26.729949951171875, "NaCl.cif", {}),
        (
            "nequip",
            "cpu",
            -169815.1282456301,
            "toluene.xyz",
            {"model": NEQUIP_PATH},
        ),
        ("orb", "cpu", -27.08186149597168, "NaCl.cif", {}),
        ("orb", "cpu", -27.089094161987305, "NaCl.cif", {"model": "orb-v2"}),
        (
            "sevennet",
            "cpu",
            -27.061979293823242,
            "NaCl.cif",
            {"model": SEVENNET_PATH},
        ),
        ("sevennet", "cpu", -27.061979293823242, "NaCl.cif", {}),
        (
            "sevennet",
            "cpu",
            -27.061979293823242,
            "NaCl.cif",
            {"model": "SevenNet-0_11July2024"},
        ),
    ],
)
def test_extras(arch, device, expected_energy, struct, kwargs):
    """Test single point energy using extra MLIP calculators."""
    skip_extras(arch)
    # Skip fairchem eSEN if unable to download
    if (
        arch == "fairchem"
        and kwargs.get("model", None) == FAIRCHEM_ESEN
        and not get_token()
    ):
        pytest.skip("Unable to download model")

    try:
        single_point = SinglePoint(
            struct=DATA_PATH / struct,
            arch=arch,
            device=device,
            properties="energy",
            **kwargs,
        )
        energy = single_point.run()["energy"]
        assert energy == pytest.approx(expected_energy, rel=1e-5)
    except URLError as err:
        if "Connection timed out" in err.reason:
            pytest.skip("Model download failed")
        raise err


def test_single_point_none():
    """Test single point stress using MACE calculator."""
    skip_extras("mace")

    single_point = SinglePoint(
        struct=DATA_PATH / "NaCl.cif",
        arch="mace",
        model=MACE_PATH,
    )

    results = single_point.run()
    for prop in ["energy", "forces", "stress"]:
        assert prop in results


def test_single_point_clean():
    """Test single point stress using MACE calculator."""
    skip_extras("mace")

    single_point = SinglePoint(
        struct=DATA_PATH / "H2O.cif",
        arch="mace",
        model=MACE_PATH,
    )

    results = single_point.run()
    for prop in ["energy", "forces"]:
        assert prop in results
    assert "mace_stress" not in results


def test_single_point_traj():
    """Test single point stress using MACE calculator."""
    skip_extras("mace")

    single_point = SinglePoint(
        struct=DATA_PATH / "benzene-traj.xyz",
        arch="mace",
        model=MACE_PATH,
        properties="energy",
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


def test_single_point_write():
    """Test writing singlepoint results."""
    skip_extras("mace")

    data_path = DATA_PATH / "NaCl.cif"
    results_dir = Path("./janus_results")
    results_path = results_dir / "NaCl-results.extxyz"

    assert not results_dir.exists()

    try:
        single_point = SinglePoint(
            struct=data_path,
            arch="mace",
            model=MACE_PATH,
            write_results=True,
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
    finally:
        shutil.rmtree(results_dir, ignore_errors=True)


def test_single_point_write_kwargs(tmp_path):
    """Test passing write_kwargs to singlepoint results."""
    skip_extras("mace")

    data_path = DATA_PATH / "NaCl.cif"
    results_path = tmp_path / "NaCl.extxyz"

    single_point = SinglePoint(
        struct=data_path,
        arch="mace",
        model=MACE_PATH,
        write_results=True,
        write_kwargs={"filename": results_path},
    )
    assert "mace_forces" not in single_point.struct.arrays

    single_point.run()
    atoms = read(results_path)
    assert atoms.info["mace_energy"] is not None
    assert "mace_forces" in atoms.arrays


def test_single_point_molecule(tmp_path):
    """Test singlepoint results for isolated molecule."""
    skip_extras("mace")

    data_path = DATA_PATH / "H2O.cif"
    results_path = tmp_path / "H2O.extxyz"
    single_point = SinglePoint(
        struct=data_path,
        arch="mace",
        model=MACE_PATH,
        properties="energy",
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


def test_atoms():
    """Test passing ASE Atoms structure."""
    skip_extras("mace")

    struct = read(DATA_PATH / "NaCl.cif")
    single_point = SinglePoint(
        struct=struct,
        arch="mace",
        model=MACE_PATH,
        properties="energy",
    )
    assert single_point.run()["energy"] < 0


def test_no_atoms_or_path():
    """Test passing neither ASE Atoms structure nor structure path."""
    with pytest.raises(TypeError):
        SinglePoint(
            arch="mace",
            model=MACE_PATH,
        )


def test_invalidate_calc():
    """Test setting invalidate_calc via write_kwargs."""
    skip_extras("mace")

    struct = DATA_PATH / "NaCl.cif"

    single_point = SinglePoint(
        struct=struct,
        arch="mace",
        model=MACE_PATH,
        write_kwargs={"invalidate_calc": False},
    )

    single_point.run()
    assert "energy" in single_point.struct.calc.results

    single_point.write_kwargs = {"invalidate_calc": True}
    single_point.run()
    assert "energy" not in single_point.struct.calc.results


def test_logging(tmp_path):
    """Test attaching logger to SinglePoint and emissions are saved to info."""
    skip_extras("mace")

    log_file = tmp_path / "sp.log"

    single_point = SinglePoint(
        struct=DATA_PATH / "NaCl.cif",
        arch="mace_mp",
        model=MACE_PATH,
        properties="energy",
        log_kwargs={"filename": log_file},
    )

    assert "emissions" not in single_point.struct.info

    single_point.run()

    assert log_file.exists()
    assert "emissions" in single_point.struct.info
    assert single_point.struct.info["emissions"] > 0


def test_hessian():
    """Test Hessian."""
    skip_extras("mace")

    sp = SinglePoint(
        model=MACE_PATH,
        struct=DATA_PATH / "NaCl.cif",
        arch="mace_mp",
        properties="hessian",
    )
    results = sp.run()
    assert "hessian" in results
    assert results["hessian"].shape == (24, 8, 3)
    assert "mace_mp_hessian" in sp.struct.info


def test_hessian_traj():
    """Test calculating Hessian for trajectory."""
    skip_extras("mace")

    sp = SinglePoint(
        model=MACE_PATH,
        struct=DATA_PATH / "benzene-traj.xyz",
        arch="mace_mp",
        properties="hessian",
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


def test_invalid_model_model_path():
    """Test error is raised when model and model_path are passed."""
    skip_extras("mace")

    with pytest.raises(ValueError):
        SinglePoint(
            arch="mace_mp",
            model=MACE_PATH,
            model_path=MACE_PATH,
            struct=DATA_PATH / "NaCl.cif",
        )


@pytest.mark.parametrize("keyword", ("model", "model_path"))
def test_invalid_model_calc_kwargs(keyword):
    """Test error is raised when model and model via calc_kwargs are passed."""
    skip_extras("mace")

    with pytest.raises(ValueError):
        SinglePoint(
            arch="mace_mp",
            model=MACE_PATH,
            calc_kwargs={keyword: MACE_PATH},
            struct=DATA_PATH / "NaCl.cif",
        )


def test_invalid_model_path_calc_kwargs():
    """Test error is raised when model_path is passed via calc_kwargs."""
    skip_extras("mace")

    with pytest.raises(ValueError):
        SinglePoint(
            arch="mace_mp",
            calc_kwargs={"model_path": MACE_PATH},
            struct=DATA_PATH / "NaCl.cif",
        )


def test_deprecation_model_path():
    """Test FutureWarning raised for model_path."""
    skip_extras("mace")

    with pytest.warns(FutureWarning, match="`model_path` has been deprecated"):
        sp = SinglePoint(
            arch="mace_mp",
            model_path=MACE_PATH,
            struct=DATA_PATH / "NaCl.cif",
        )

    assert sp.struct.calc.parameters["model"] == str(MACE_PATH.as_posix())


def test_deprecation_model_calc_kwargs():
    """Test FutureWarning raised for model in calc_kwargs."""
    skip_extras("mace")

    with pytest.warns(FutureWarning, match="Please pass `model` explicitly"):
        sp = SinglePoint(
            arch="mace_mp",
            calc_kwargs={"model": MACE_PATH},
            struct=DATA_PATH / "NaCl.cif",
        )

    assert sp.struct.calc.parameters["model"] == str(MACE_PATH.as_posix())


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
