"""Test integration of PLUMED with molecular dynamics."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from ase import units
from ase.io import read

from janus_core.calculations.md import NVT, MolecularDynamics
from janus_core.calculations.single_point import SinglePoint

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"

PLUMED_INPUT_CONTENT = """
UNITS LENGTH=A TIME=fs ENERGY=eV
d: DISTANCE ATOMS=1,2
PRINT ARG=d FILE=COLVAR STRIDE=1
"""

@pytest.fixture
def plumed_input_file(tmp_path):
    """Create a simple PLUMED input file for testing."""
    plumed_file = tmp_path / "plumed.dat"
    plumed_file.write_text(PLUMED_INPUT_CONTENT)
    return plumed_file


def test_plumed_import():
    """Test importing PLUMED."""
    try:
        from ase.calculators.plumed import Plumed
        import plumed
        has_plumed = True
    except ImportError:
        has_plumed = False
    
    if not has_plumed:
        pytest.skip("PLUMED not installed")


@pytest.mark.skipif(
    not os.environ.get("JANUS_TEST_PLUMED", False), 
    reason="Set JANUS_TEST_PLUMED=1 to run PLUMED integration tests"
)
def test_nvt_plumed(tmp_path, plumed_input_file):
    """Test NVT with PLUMED."""
    plumed_log = tmp_path / "plumed.log"
    
    orig_dir = Path.cwd()
    colvar_file = tmp_path / "COLVAR"
    
    try:
        os.chdir(tmp_path)
        
        single_point = SinglePoint(
            struct=DATA_PATH / "benzene.xyz",
            arch="mace_mp",
            calc_kwargs={"model": MODEL_PATH},
        )
        
        nvt = NVT(
            struct=single_point.struct,
            steps=5,
            timestep=0.5,
            temp=300.0,
            plumed_input=plumed_input_file,
            plumed_log=plumed_log,
            file_prefix=tmp_path / "benzene",
        )
        
        nvt.run()
        
        assert plumed_log.exists()
        assert colvar_file.exists()
        
        with open(colvar_file, "r") as f:
            lines = f.readlines()
        assert len(lines) > 1
        assert "d" in lines[0]
    
    finally:
        os.chdir(orig_dir)


def test_no_plumed_input():
    """Test MD without PLUMED input."""
    single_point = SinglePoint(
        struct=DATA_PATH / "benzene.xyz",
        arch="mace_mp",
        calc_kwargs={"model": MODEL_PATH},
    )
    
    nvt = NVT(
        struct=single_point.struct,
        steps=5,
        plumed_input=None,
    )
    
    output_files = nvt.output_files
    assert "plumed_log" in output_files
    assert output_files["plumed_log"] is None