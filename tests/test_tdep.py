"""Tests for TDEP input generation."""

from __future__ import annotations

from pathlib import Path

from ase.io import write

from janus_core.calculations.md import NVT
from janus_core.calculations.single_point import SinglePoint


def test_nvt_tdep(tmp_path) -> None:
    """Generate TDEP input files through MD NVT post-processing."""
    unit_cell_file = Path(__file__).parent / "data" / "NaCl.cif"
    supercell_file = tmp_path / "supercell.cif"
    output_dir = tmp_path / "tdep-inputs"

    single_point = SinglePoint(
        struct=unit_cell_file,
        arch="mace",
        model=Path(__file__).parent / "models" / "mace_mp_small.model",
    )

    write(supercell_file, single_point.struct, format="cif")

    nvt = NVT(
        struct=single_point.struct,
        temp=300.0,
        steps=10,
        traj_every=1,
        stats_every=1,
        file_prefix=tmp_path / "NaCl-nvt-T300.0",
        post_process_kwargs={
            "tdep_compute": True,
            "tdep_unit_cell_file": unit_cell_file,
            "tdep_supercell_file": supercell_file,
            "tdep_output_dir": output_dir,
        },
    )

    nvt.run()

    expected_files = {
        "infile.ucposcar",
        "infile.ssposcar",
        "infile.positions",
        "infile.forces",
        "infile.stat",
        "infile.meta",
    }

    for name in expected_files:
        assert (output_dir / name).exists()

    meta_lines = (output_dir / "infile.meta").read_text(encoding="utf-8").splitlines()

    assert len(meta_lines) == 4
    assert meta_lines[0].endswith("# N atoms")
    assert meta_lines[1].endswith("# N timesteps")
    assert meta_lines[2] == "0.0    # timestep in fs"
    assert meta_lines[3] == "300.0    # temperature in K"

    positions_lines = (
        (output_dir / "infile.positions").read_text(encoding="utf-8").splitlines()
    )
    forces_lines = (
        (output_dir / "infile.forces").read_text(encoding="utf-8").splitlines()
    )
    stat_lines = (output_dir / "infile.stat").read_text(encoding="utf-8").splitlines()

    assert len(positions_lines) > 0
    assert len(forces_lines) > 0
    assert len(stat_lines) > 0
