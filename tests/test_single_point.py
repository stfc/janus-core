"""Test configuration of MLIP calculators."""

from pathlib import Path

from ase.io import read
from numpy import isfinite
import pytest

from janus_core.calculations.single_point import SinglePoint
from tests.utils import read_atoms

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"

test_data = [
    (DATA_PATH / "benzene.xyz", -76.0605725422795, "energy", "energy", {}, None),
    (
        DATA_PATH / "benzene.xyz",
        -76.06057739257812,
        ["energy"],
        "energy",
        {"default_dtype": "float32"},
        None,
    ),
    (DATA_PATH / "benzene.xyz", -0.0360169762840179, ["forces"], "forces", {}, [0, 1]),
    (DATA_PATH / "NaCl.cif", -0.004783275999053424, ["stress"], "stress", {}, [0]),
]


@pytest.mark.parametrize(
    "struct_path, expected, properties, prop_key, calc_kwargs, idx", test_data
)
def test_potential_energy(
    struct_path, expected, properties, prop_key, calc_kwargs, idx
):
    """Test single point energy using MACE calculators."""
    calc_kwargs["model"] = MODEL_PATH
    single_point = SinglePoint(
        struct_path=struct_path, architecture="mace", calc_kwargs=calc_kwargs
    )
    results = single_point.run(properties)[prop_key]

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


def test_single_point_none():
    """Test single point stress using MACE calculator."""
    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    results = single_point.run()
    for prop in ["energy", "forces", "stress"]:
        assert prop in results


def test_single_point_clean():
    """Test single point stress using MACE calculator."""
    single_point = SinglePoint(
        struct_path=DATA_PATH / "H2O.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    results = single_point.run()
    for prop in ["energy", "forces"]:
        assert prop in results
    assert "mace_stress" not in results


def test_single_point_traj():
    """Test single point stress using MACE calculator."""
    single_point = SinglePoint(
        struct_path=DATA_PATH / "benzene-traj.xyz",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    assert len(single_point.struct) == 2
    results = single_point.run("energy")
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
    data_path = DATA_PATH / "NaCl.cif"
    results_path = Path("./NaCl-results.xyz").absolute()
    assert not results_path.exists()

    single_point = SinglePoint(
        struct_path=data_path,
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    assert "mace_forces" not in single_point.struct.arrays

    single_point.run(write_results=True)
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


def test_single_point_write_kwargs(tmp_path):
    """Test passing write_kwargs to singlepoint results."""
    data_path = DATA_PATH / "NaCl.cif"
    results_path = tmp_path / "NaCl.xyz"

    single_point = SinglePoint(
        struct_path=data_path,
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    assert "mace_forces" not in single_point.struct.arrays

    single_point.run(write_results=True, write_kwargs={"filename": results_path})
    atoms = read(results_path)
    assert atoms.info["mace_energy"] is not None
    assert "mace_forces" in atoms.arrays


def test_single_point_molecule(tmp_path):
    """Test singlepoint results for isolated molecule."""
    data_path = DATA_PATH / "H2O.cif"
    results_path = tmp_path / "H2O.xyz"
    single_point = SinglePoint(
        struct_path=data_path,
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )

    assert isfinite(single_point.run("energy")["energy"]).all()

    single_point.run(write_results=True, write_kwargs={"filename": results_path})
    atoms = read(results_path)
    assert atoms.info["mace_energy"] == pytest.approx(-14.035236305927514)
    assert "mace_forces" in atoms.arrays
    assert "mace_stress" in atoms.info
    assert atoms.info["mace_stress"] == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_invalid_prop():
    """Test invalid property request."""
    single_point = SinglePoint(
        struct_path=DATA_PATH / "H2O.cif",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    with pytest.raises(NotImplementedError):
        single_point.run("invalid")


def test_atoms():
    """Test passing ASE Atoms structure."""
    struct = read(DATA_PATH / "NaCl.cif")
    single_point = SinglePoint(
        struct=struct,
        struct_name="NaCl",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    assert single_point.struct_name == "NaCl"
    assert single_point.run("energy")["energy"] < 0


def test_default_atoms_name():
    """Test default structure name when passing ASE Atoms structure."""
    struct = read(DATA_PATH / "NaCl.cif")
    single_point = SinglePoint(
        struct=struct,
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    assert single_point.struct_name == "Cl4Na4"


def test_default_path_name():
    """Test default structure name when passing structure path."""
    struct_path = DATA_PATH / "NaCl.cif"
    single_point = SinglePoint(
        struct_path=struct_path,
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    assert single_point.struct_name == "NaCl"


def test_path_specify_name():
    """Test specifying structure name with structure path."""
    struct_path = DATA_PATH / "NaCl.cif"
    single_point = SinglePoint(
        struct_path=struct_path,
        struct_name="example_name",
        architecture="mace",
        calc_kwargs={"model": MODEL_PATH},
    )
    assert single_point.struct_name == "example_name"


def test_atoms_and_path():
    """Test passing ASE Atoms structure and structure path togther."""
    struct = read(DATA_PATH / "NaCl.cif")
    struct_path = DATA_PATH / "NaCl.cif"
    with pytest.raises(ValueError):
        SinglePoint(
            struct=struct,
            struct_path=struct_path,
            architecture="mace",
            calc_kwargs={"model": MODEL_PATH},
        )


def test_no_atoms_or_path():
    """Test passing neither ASE Atoms structure nor structure path."""
    with pytest.raises(ValueError):
        SinglePoint(
            architecture="mace",
            calc_kwargs={"model": MODEL_PATH},
        )


test_mlips_data = [
    ("m3gnet", "cpu", -26.729949951171875),
    ("chgnet", "cpu", -29.331436157226562),
]


@pytest.mark.parametrize("arch, device, expected_energy", test_mlips_data)
def test_mlips(arch, device, expected_energy):
    """Test single point energy using CHGNET and M3GNET calculators."""
    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture=arch,
        device=device,
    )
    energy = single_point.run("energy")["energy"]
    assert energy == pytest.approx(expected_energy)


test_extra_mlips_data = [("alignn", "cpu", -11.148092269897461)]


@pytest.mark.extra_mlips
@pytest.mark.parametrize("arch, device, expected_energy", test_extra_mlips_data)
def test_extra_mlips(arch, device, expected_energy):
    """Test single point energy using ALIGNN-FF calculator."""
    single_point = SinglePoint(
        struct_path=DATA_PATH / "NaCl.cif",
        architecture=arch,
        device=device,
    )
    energy = single_point.run("energy")["energy"]
    assert energy == pytest.approx(expected_energy)
