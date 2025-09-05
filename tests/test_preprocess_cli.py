"""Test preprocess commandline interface."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner
import yaml

from janus_core.cli.janus import app
from tests.utils import (
    assert_log_contains,
    chdir,
    check_output_files,
    clear_log_handlers,
    strip_ansi_codes,
)

DATA_PATH = Path(__file__).parent / "data"

runner = CliRunner()


def write_tmp_config(config_path: Path, tmp_path: Path) -> Path:
    """
    Fix paths in config files and write corrected config to tmp_path.

    Parameters
    ----------
    config_path
        Path to yaml config file to be fixed.
    tmp_path
        Temporary path from pytest in which to write corrected config.

    Returns
    -------
    Path
        Temporary path to corrected config file.
    """
    # Load config from tests/data
    with open(config_path, encoding="utf8") as file:
        config = yaml.safe_load(file)

    # Use DATA_PATH to set paths relative to this test file
    for file in ("train_file", "test_file", "valid_file"):
        if file in config and (DATA_PATH / Path(config[file]).name).exists():
            config[file] = str(DATA_PATH / Path(config[file]).name)

    # Write out temporary config with corrected paths
    tmp_config = tmp_path / "config.yml"
    with open(tmp_config, "w", encoding="utf8") as file:
        yaml.dump(config, file)

    return tmp_config


def test_help():
    """Test calling `janus preprocess --help`."""
    result = runner.invoke(app, ["preprocess", "--help"])
    assert result.exit_code == 0
    assert "Usage: janus preprocess [OPTIONS]" in strip_ansi_codes(result.stdout)


def test_preprocess(tmp_path):
    """Test MLIP preprocessing."""
    with chdir(tmp_path):
        train_path = Path("train")
        val_path = Path("val")
        test_path = Path("test")
        stats_path = Path("statistics.json")
        results_dir = Path("janus_results")
        log_path = results_dir / "preprocess-log.yml"
        summary_path = results_dir / "preprocess-summary.yml"

        config = write_tmp_config(DATA_PATH / "mlip_preprocess.yml", Path())

        result = runner.invoke(
            app,
            [
                "preprocess",
                "--mlip-config",
                config,
            ],
        )
        assert result.exit_code == 0

        assert train_path.is_dir()
        assert val_path.is_dir()
        assert test_path.is_dir()
        assert not stats_path.exists()
        assert log_path.exists()
        assert summary_path.exists()

        assert_log_contains(
            log_path, includes=["Starting preprocessing", "Preprocessing complete"]
        )

        # Read train summary file and check contents
        with open(summary_path, encoding="utf8") as file:
            preprocess_summary = yaml.safe_load(file)

        assert "command" in preprocess_summary
        assert "janus preprocess" in preprocess_summary["command"]
        assert "start_time" in preprocess_summary
        assert "config" in preprocess_summary
        assert "end_time" in preprocess_summary

        assert "emissions" in preprocess_summary
        assert preprocess_summary["emissions"] > 0

        output_files = {
            "log": log_path,
            "summary": summary_path,
        }
        check_output_files(preprocess_summary, output_files)

        clear_log_handlers()


def test_preprocess_stats(tmp_path):
    """Test statistics file."""
    with chdir(tmp_path):
        stats_path = Path("./statistics.json")
        log_path = tmp_path / "test.log"
        summary_path = tmp_path / "summary.yml"

        config = write_tmp_config(DATA_PATH / "mlip_preprocess_stats.yml", Path())

        result = runner.invoke(
            app,
            [
                "preprocess",
                "--mlip-config",
                config,
                "--log",
                log_path,
                "--summary",
                summary_path,
            ],
        )

        assert result.exit_code == 0
        assert stats_path.is_file()

        clear_log_handlers()


def test_no_carbon(tmp_path):
    """Test disabling carbon tracking."""
    with chdir(tmp_path):
        log_path = tmp_path / "test.log"
        summary_path = tmp_path / "summary.yml"

        config = write_tmp_config(DATA_PATH / "mlip_preprocess.yml", Path())

        result = runner.invoke(
            app,
            [
                "preprocess",
                "--mlip-config",
                config,
                "--no-tracker",
                "--log",
                log_path,
                "--summary",
                summary_path,
            ],
        )
        assert result.exit_code == 0

        with open(summary_path, encoding="utf8") as file:
            preprocess_summary = yaml.safe_load(file)
        assert "emissions" not in preprocess_summary

        clear_log_handlers()
