"""Test elasticity commandline interface."""

from __future__ import annotations

from pathlib import Path

from ase.io import read
import numpy as np
from pytest import approx
from typer.testing import CliRunner
import yaml

from janus_core.cli.janus import app
from tests.utils import assert_log_contains, chdir, strip_ansi_codes

DATA_PATH = Path(__file__).parent / "data"
MACE_PATH = Path(__file__).parent / "models" / "mace_mp_small.model"

runner = CliRunner()


def test_help():
    """Test calling janus elasticity --help."""
    result = runner.invoke(app, ["elasticity", "--help"])
    assert result.exit_code == 0
    assert "Usage: janus elasticity [OPTIONS]" in strip_ansi_codes(result.stdout)


def test_elasticity(tmp_path):
    """Test calculating the ElasticTensor from the command line."""
    with chdir(tmp_path):
        results_dir = Path("./janus_results")
        elasticity_path = results_dir / "NaCl-elastic_tensor.dat"
        log_path = results_dir / "NaCl-elasticity-log.yml"
        summary_path = results_dir / "NaCl-elasticity-summary.yml"
        generated_path = results_dir / "NaCl-generated.extxyz"

        result = runner.invoke(
            app,
            [
                "elasticity",
                "--struct",
                DATA_PATH / "NaCl.cif",
                "--arch",
                "mace_mp",
                "--n-strains",
                "2",
                "--minimize-all",
                "--write-structures",
            ],
        )

        assert result.exit_code == 0

        assert elasticity_path.exists()
        assert log_path.exists()
        assert summary_path.exists()
        assert generated_path.exists()

        written_elasticity = np.loadtxt(elasticity_path)

        assert written_elasticity[0] == approx(27.368617328271498)
        assert written_elasticity[-1] == approx(2.1775175649257585)

        generated = read(generated_path, index=":")
        assert len(generated) == 13

        assert_log_contains(log_path, includes=["Minimising initial structure"])

        with open(log_path) as io:
            logs = yaml.safe_load(io.read())
            optimization_logs = sum(
                log["message"] == ["Starting geometry optimization"] for log in logs
            )
            assert optimization_logs == 13

        with open(summary_path, encoding="utf8") as file:
            elasticity_summary = yaml.safe_load(file)

        assert "command" in elasticity_summary
        assert "janus elasticity" in elasticity_summary["command"]
        assert "start_time" in elasticity_summary
        assert "config" in elasticity_summary
        assert "info" in elasticity_summary
        assert "end_time" in elasticity_summary

        assert "emissions" in elasticity_summary
        assert elasticity_summary["emissions"] > 0

        assert "n_strains" in elasticity_summary["config"]
        assert elasticity_summary["config"]["n_strains"] == 2
