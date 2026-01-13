"""Test train commandline interface."""

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
MODEL_PATH = Path(__file__).parent / "models"

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

    # Use MODEL_PATH to set paths relative to this test file
    for file in ("foundation_model",):
        if file in config and (MODEL_PATH / Path(config[file]).name).exists():
            config[file] = str(MODEL_PATH / Path(config[file]).name)

    # Write out temporary config with corrected paths
    tmp_config = tmp_path / "config.yml"
    with open(tmp_config, "w", encoding="utf8") as file:
        yaml.dump(config, file)

    return tmp_config


def test_help():
    """Test calling `janus train --help`."""
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "Usage: janus train [OPTIONS]" in strip_ansi_codes(result.stdout)


def test_train(tmp_path):
    """Test MLIP training."""
    with chdir(tmp_path):
        model = Path("test.model")
        compiled_model = Path("test_compiled.model")
        results_path = Path("results")
        checkpoints_path = Path("checkpoints")
        logs_path = Path("logs")
        results_dir = Path("./janus_results")
        log_path = results_dir / "train-log.yml"
        summary_path = results_dir / "train-summary.yml"

        config = write_tmp_config(DATA_PATH / "mlip_train.yml", Path())

        result = runner.invoke(
            app,
            [
                "train",
                "mace--mlip-config",
                config,
            ],
        )
        assert result.exit_code == 0

        assert model.exists()
        assert compiled_model.exists()
        assert logs_path.is_dir()
        assert results_path.is_dir()
        assert checkpoints_path.is_dir()
        assert log_path.exists()
        assert summary_path.exists()

        assert_log_contains(
            log_path, includes=["Starting training", "Training complete"]
        )

        # Read train summary file and check contents
        with open(summary_path, encoding="utf8") as file:
            train_summary = yaml.safe_load(file)

        assert "command" in train_summary
        assert "janus train" in train_summary["command"]
        assert "start_time" in train_summary
        assert "config" in train_summary
        assert "end_time" in train_summary

        assert "emissions" in train_summary
        assert train_summary["emissions"] > 0

        output_files = {
            "log": log_path,
            "summary": summary_path,
        }
        check_output_files(train_summary, output_files)

        clear_log_handlers()


def test_train_with_foundation(tmp_path):
    """Test MLIP training raises error with foundation_model in config."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    config = write_tmp_config(DATA_PATH / "mlip_train_invalid.yml", tmp_path)

    result = runner.invoke(
        app,
        [
            "train",
            "mace",
            "--mlip-config",
            config,
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)


def test_fine_tune(tmp_path):
    """Test MLIP fine-tuning."""
    with chdir(tmp_path):
        log_path = tmp_path / "test.log"
        summary_path = tmp_path / "summary.yml"

        model = Path("test-finetuned.model")
        compiled_model = Path("test-finetuned_compiled.model")
        logs_path = Path("logs")
        results_path = Path("results")
        checkpoints_path = Path("checkpoints")

        config = write_tmp_config(DATA_PATH / "mlip_fine_tune.yml", Path())

        result = runner.invoke(
            app,
            [
                "train",
                "mace",
                "--mlip-config",
                config,
                "--fine-tune",
                "--log",
                log_path,
                "--summary",
                summary_path,
            ],
        )
        assert result.exit_code == 0

        assert model.exists()
        assert compiled_model.exists()
        assert logs_path.is_dir()
        assert results_path.is_dir()
        assert checkpoints_path.is_dir()

        clear_log_handlers()


def test_fine_tune_no_foundation(tmp_path):
    """Test MLIP fine-tuning raises errors without foundation_model."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    config = DATA_PATH / "mlip_fine_tune_no_foundation.yml"

    result = runner.invoke(
        app,
        [
            "train",
            "mace",
            "--mlip-config",
            config,
            "--fine-tune",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)


def test_fine_tune_invalid_foundation(tmp_path):
    """Test MLIP fine-tuning raises errors with invalid foundation_model."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"
    config = DATA_PATH / "mlip_fine_tune_invalid_foundation.yml"

    result = runner.invoke(
        app,
        [
            "train",
            "mace",
            "--mlip-config",
            config,
            "--fine-tune",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)


def test_no_carbon(tmp_path):
    """Test disabling carbon tracking."""
    with chdir(tmp_path):
        Path("test.model")
        Path("test_compiled.model")
        Path("results")
        Path("checkpoints")
        Path("logs")
        log_path = tmp_path / "test.log"
        summary_path = tmp_path / "summary.yml"

        config = write_tmp_config(DATA_PATH / "mlip_train.yml", Path())

        result = runner.invoke(
            app,
            [
                "train",
                "mace",
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
            train_summary = yaml.safe_load(file)
        assert "emissions" not in train_summary

        clear_log_handlers()
