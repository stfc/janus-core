"""Tests for the janus-core MCP server, mode runners, and CLI."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ase import Atoms
import numpy as np
import pytest
from typer.testing import CliRunner

from janus_core.cli.janus import app as janus_app
from janus_core.mcp.server import (
    _validate_modes,
    create_mcp_server,
    run_mcp_server,
)
from janus_core.mcp.tools import (
    JANUS_MODES,
    _to_jsonable,
    inspect_structure,
    list_architectures,
    list_modes,
    run_descriptors,
    run_elasticity,
    run_eos,
    run_geomopt,
    run_md,
    run_mode,
    run_neb,
    run_phonons,
    run_preprocess,
    run_singlepoint,
    run_train,
)

runner = CliRunner()


def test_list_modes():
    """Test that list_modes returns all 10 janus-core modes."""
    result = list_modes()
    assert result["count"] == 10
    expected = {
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
    }
    assert set(result["modes"].keys()) == expected


def test_list_architectures():
    """Test that list_architectures reports architectures, devices, and ensembles."""
    result = list_architectures()
    assert "mace_mp" in result["architectures"]
    assert "cpu" in result["devices"]
    assert "nvt" in result["ensembles"]
    assert "birchmurnaghan" in result["eos_types"]
    assert "energy" in result["properties"]
    assert "bands" in result["phonon_calcs"]


def test_inspect_structure(tmp_path: Path):
    """Test inspect_structure on a real periodic ASE structure."""
    from ase.build import bulk
    from ase.io import write

    struct_path = tmp_path / "nacl.cif"
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    write(struct_path, atoms)

    info = inspect_structure(str(struct_path))
    assert info["n_atoms"] == len(atoms)
    assert info["elements"] == ["Cl", "Na"]
    assert info["pbc"] == [True, True, True]
    assert info["volume_angstrom3"] > 0


def test_to_jsonable():
    """Test recursive JSON serialization of NumPy, Path, and ASE objects."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    payload = {
        "path": Path("/tmp/test.xyz"),
        "arr": np.array([1.0, 2.0]),
        "scalar": np.float64(3.5),
        "atoms": atoms,
        "tuple": (1, 2),
    }
    converted = _to_jsonable(payload)
    assert converted["path"] == "/tmp/test.xyz"
    assert converted["arr"] == [1.0, 2.0]
    assert converted["scalar"] == 3.5
    assert converted["atoms"]["formula"] == "H2"
    assert converted["atoms"]["volume_angstrom3"] is None
    assert converted["tuple"] == [1, 2]


def test_run_singlepoint_mocked():
    """Test run_singlepoint forwards arguments and serializes results."""
    mock_instance = MagicMock()
    mock_instance.run.return_value = {
        "energy": np.float64(-10.5),
        "forces": np.zeros((2, 3)),
    }
    with patch(
        "janus_core.calculations.single_point.SinglePoint",
        return_value=mock_instance,
    ) as mock_cls:
        res = run_singlepoint(struct="NaCl.cif", arch="mace_mp", properties=["energy"])

    assert res["mode"] == "singlepoint"
    assert res["results"]["energy"] == -10.5
    assert res["results"]["forces"] == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    _, kwargs = mock_cls.call_args
    assert kwargs["track_carbon"] is False
    assert kwargs["attach_logger"] is False


def test_run_geomopt_mocked():
    """Test run_geomopt uses FrechetCellFilter by default and returns energy."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.get_potential_energy = MagicMock(return_value=-6.8)
    mock_opt = MagicMock(converged=True, struct=atoms)
    with patch(
        "janus_core.calculations.geom_opt.GeomOpt", return_value=mock_opt
    ) as mock_cls:
        res = run_geomopt(struct="H2.xyz", fmax=0.05)

    assert res["mode"] == "geomopt"
    assert res["converged"] is True
    assert res["energy_ev"] == -6.8
    _, kwargs = mock_cls.call_args
    assert kwargs["filter_class"] == "FrechetCellFilter"
    assert kwargs["track_carbon"] is False


@pytest.mark.parametrize(
    ("ensemble", "cls_name"),
    [
        ("nve", "NVE"),
        ("nvt", "NVT"),
        ("nvt-nh", "NVT_NH"),
        ("nvt-csvr", "NVT_CSVR"),
        ("npt", "NPT"),
        ("nph", "NPH"),
        ("npt-mtk", "NPT_MTK"),
        ("npt-mtk-iso", "NPT_MTK"),
        ("npt-mtk-aniso", "NPT_MTK"),
    ],
)
def test_run_md_all_ensembles(ensemble: str, cls_name: str):
    """Test run_md dispatches to the right ensemble class."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    mock_dyn = MagicMock(struct=atoms, output_files={"traj": Path("out.extxyz")})
    with patch(
        f"janus_core.calculations.md.{cls_name}", return_value=mock_dyn
    ) as mock_cls:
        res = run_md(struct="H2.xyz", ensemble=ensemble, steps=5)  # type: ignore[arg-type]

    assert res["mode"] == "md"
    assert res["ensemble"] == ensemble
    assert res["output_files"] == {"traj": "out.extxyz"}
    mock_cls.assert_called_once()
    mock_dyn.run.assert_called_once()


def test_run_md_invalid_ensemble():
    """Test run_md raises ValueError on unknown ensemble."""
    with pytest.raises(ValueError, match="Unsupported ensemble"):
        run_md(struct="H2.xyz", ensemble="invalid")  # type: ignore[arg-type]


def test_run_phonons_mocked():
    """Test run_phonons summarizes phonon results cleanly."""
    mock_ph = MagicMock(
        results={
            "phonon": object(),
            "thermal": {"temperatures": np.array([0.0, 300.0])},
        }
    )
    with patch("janus_core.calculations.phonons.Phonons", return_value=mock_ph):
        res = run_phonons(struct="NaCl.cif", supercell=2, calcs=["thermal"])

    assert res["mode"] == "phonons"
    assert res["results"]["phonon"] == "Phonopy object"
    assert res["results"]["thermal"]["temperatures"] == [0.0, 300.0]


def test_run_eos_mocked():
    """Test run_eos returns fitted bulk modulus, v_0, and e_0."""
    mock_eos = MagicMock(volumes=[10.0, 11.0], energies=[-5.0, -5.1])
    mock_eos.run.return_value = {
        "bulk_modulus": np.float64(25.3),
        "v_0": np.float64(10.5),
        "e_0": np.float64(-5.12),
    }
    with patch("janus_core.calculations.eos.EoS", return_value=mock_eos):
        res = run_eos(struct="NaCl.cif")

    assert res["mode"] == "eos"
    assert res["bulk_modulus_gpa"] == 25.3
    assert res["v_0_angstrom3"] == 10.5
    assert res["e_0_ev"] == -5.12


def test_run_elasticity_mocked():
    """Test run_elasticity serializes Voigt matrix and moduli."""
    tensor = SimpleNamespace(
        voigt=np.eye(6),
        k_vrh=30.0,
        g_vrh=15.0,
        y_mod=38.5,
    )
    mock_el = MagicMock(elastic_tensor=tensor)
    with patch("janus_core.calculations.elasticity.Elasticity", return_value=mock_el):
        res = run_elasticity(struct="NaCl.cif")

    assert res["mode"] == "elasticity"
    assert res["k_vrh_gpa"] == 30.0
    assert len(res["voigt_gpa"]) == 6


def test_run_neb_mocked():
    """Test run_neb returns convergence and barrier results."""
    mock_neb = MagicMock(converged=True)
    mock_neb.run.return_value = {"barrier": np.float64(0.45)}
    with patch("janus_core.calculations.neb.NEB", return_value=mock_neb):
        res = run_neb(init_struct="init.xyz", final_struct="final.xyz", n_images=5)

    assert res["mode"] == "neb"
    assert res["converged"] is True
    assert res["results"]["barrier"] == 0.45


def test_run_descriptors_mocked():
    """Test run_descriptors extracts descriptor keys from atoms.info."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.info["mace_mp_descriptor"] = np.array([0.1, 0.2])
    mock_desc = MagicMock(struct=atoms)
    with patch(
        "janus_core.calculations.descriptors.Descriptors", return_value=mock_desc
    ):
        res = run_descriptors(struct="H2.xyz")

    assert res["mode"] == "descriptors"
    assert res["n_structures"] == 1
    assert res["descriptors"][0]["mace_mp_descriptor"] == [0.1, 0.2]


def test_run_train_and_preprocess_mocked():
    """Test run_train and run_preprocess call training helpers."""
    with patch("janus_core.training.train.train") as mock_train:
        res_train = run_train(arch="mace_mp", mlip_config="train.yml")
    assert res_train["mode"] == "train"
    assert res_train["status"] == "completed"
    mock_train.assert_called_once()

    with patch("janus_core.training.preprocess.preprocess") as mock_pre:
        res_pre = run_preprocess(mlip_config="preprocess.yml")
    assert res_pre["mode"] == "preprocess"
    assert res_pre["status"] == "completed"
    mock_pre.assert_called_once()


def test_run_mode_dispatch_and_error():
    """Test unified run_mode dispatcher and invalid mode error."""
    with patch.dict(
        "janus_core.mcp.tools.MODE_HANDLERS",
        {"singlepoint": MagicMock(return_value={"mode": "singlepoint", "ok": True})},
    ):
        res = run_mode("singlepoint", {"struct": "NaCl.cif"})
    assert res == {"mode": "singlepoint", "ok": True}

    with pytest.raises(ValueError, match="Unsupported janus-core mode"):
        run_mode("not_a_mode")  # type: ignore[arg-type]


def test_validate_modes():
    """Test mode validation and deduplication."""
    assert _validate_modes(None) == list(JANUS_MODES.keys())
    assert _validate_modes(["singlepoint", "md", "singlepoint"]) == [
        "singlepoint",
        "md",
    ]
    with pytest.raises(ValueError, match="Unknown janus-core mode"):
        _validate_modes(["singlepoint", "bogus"])


def test_create_mcp_server_all_modes():
    """Test MCP server registers all 10 modes, utilities, and resources."""
    pytest.importorskip("mcp")
    server = create_mcp_server()

    tools = asyncio.run(server.list_tools())
    tool_names = {tool.name for tool in tools}
    assert set(JANUS_MODES.keys()).issubset(tool_names)
    assert {
        "list_modes",
        "list_architectures",
        "inspect_structure",
        "run_mode",
    }.issubset(tool_names)

    resources = asyncio.run(server.list_resources())
    resource_uris = {str(res.uri) for res in resources}
    assert {"janus://version", "janus://modes", "janus://architectures"}.issubset(
        resource_uris
    )


def test_create_mcp_server_filtered_modes():
    """Test MCP server only registers requested subset of modes."""
    pytest.importorskip("mcp")
    server = create_mcp_server(modes=["singlepoint", "geomopt"])

    tools = asyncio.run(server.list_tools())
    tool_names = {tool.name for tool in tools}
    assert "singlepoint" in tool_names
    assert "geomopt" in tool_names
    assert "md" not in tool_names
    assert "run_mode" in tool_names


@pytest.mark.parametrize("transport", ["stdio", "sse", "streamable-http"])
def test_run_mcp_server_transports(transport: str):
    """Test run_mcp_server dispatches all MCP transport modes."""
    mock_server = MagicMock()

    def _fake_run(transport: str = "stdio", **kwargs):
        return None

    mock_server.run = MagicMock(wraps=_fake_run)
    with patch("janus_core.mcp.server.create_mcp_server", return_value=mock_server):
        run_mcp_server(
            transport=transport,  # type: ignore[arg-type]
            host="0.0.0.0",
            port=9001,
            mount_path="/custom",
        )

    mock_server.run.assert_called_once()
    _, kwargs = mock_server.run.call_args
    assert kwargs["transport"] == transport


def test_run_mcp_server_invalid_transport():
    """Test run_mcp_server rejects unsupported transport names."""
    with pytest.raises(ValueError, match="Unsupported transport"):
        run_mcp_server(transport="websocket")  # type: ignore[arg-type]


def test_cli_mcp_help_and_invocation():
    """Test `janus mcp --help` and option forwarding to run_mcp_server."""
    help_res = runner.invoke(janus_app, ["mcp", "--help"])
    assert help_res.exit_code == 0
    assert "--transport" in help_res.stdout
    assert "--mode" in help_res.stdout

    with patch("janus_core.mcp.server.run_mcp_server") as mock_run:
        res = runner.invoke(
            janus_app,
            [
                "mcp",
                "--transport",
                "streamable-http",
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
                "--mode",
                "singlepoint",
                "--mode",
                "md",
            ],
        )
    assert res.exit_code == 0
    mock_run.assert_called_once_with(
        transport="streamable-http",
        host="0.0.0.0",
        port=8080,
        mount_path=None,
        modes=["singlepoint", "md"],
    )
