"""Test train commandline interface."""

import logging
from pathlib import Path
import shutil

from typer.testing import CliRunner
import yaml

from janus_core.cli.janus import app
from tests.utils import assert_log_contains, clear_log_handlers, strip_ansi_codes

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models"

runner = CliRunner()


def write_tmp_config(config_path: Path, tmp_path: Path) -> Path:
    """
    Fix paths in config files and write corrected config to tmp_path.

    Parameters
    ----------
    config_path : Path
        Path to yaml config file to be fixed.
    tmp_path : Path
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
    summary_path = tmp_path / "summary.yml"
    model = "test.model"
    compiled_model = "test_compiled.model"
    results_path = "results"
    checkpoints_path = "checkpoints"
    logs_path = "logs"
    log_path = Path("./train-log.yml").absolute()
    summary_path = Path("./train-summary.yml").absolute()

    assert not Path(model).exists()
    assert not Path(compiled_model).exists()
    assert not Path(logs_path).exists()
    assert not Path(results_path).exists()
    assert not Path(checkpoints_path).exists()
    assert not log_path.exists()
    assert not summary_path.exists()

    config = write_tmp_config(DATA_PATH / "mlip_train.yml", tmp_path)

    result = runner.invoke(
        app,
        [
            "train",
            "--mlip-config",
            config,
        ],
    )
    try:
        assert result.exit_code == 0

        assert Path(model).exists()
        assert Path(compiled_model).exists()
        assert Path(logs_path).is_dir()
        assert Path(results_path).is_dir()
        assert Path(checkpoints_path).is_dir()

        assert log_path.exists()
        assert summary_path.exists()

        assert_log_contains(
            log_path, includes=["Starting training", "Training complete"]
        )

        # Read train summary file and check contents
        assert summary_path.exists()
        with open(summary_path, encoding="utf8") as file:
            train_summary = yaml.safe_load(file)

        assert "command" in train_summary
        assert "janus train" in train_summary["command"]
        assert "start_time" in train_summary
        assert "inputs" in train_summary
        assert "end_time" in train_summary

        assert "emissions" in train_summary
        assert train_summary["emissions"] > 0

    finally:
        # Tidy up models
        Path(model).unlink(missing_ok=True)
        Path(compiled_model).unlink(missing_ok=True)

        # Tidy up directories
        shutil.rmtree(logs_path, ignore_errors=True)
        shutil.rmtree(results_path, ignore_errors=True)
        shutil.rmtree(checkpoints_path, ignore_errors=True)

        log_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)

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
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    model = "test-finetuned.model"
    compiled_model = "test-finetuned_compiled.model"
    logs_path = "logs"
    results_path = "results"
    checkpoints_path = "checkpoints"

    assert not Path(model).exists()
    assert not Path(compiled_model).exists()
    assert not Path(logs_path).exists()
    assert not Path(results_path).exists()
    assert not Path(checkpoints_path).exists()

    config = write_tmp_config(DATA_PATH / "mlip_fine_tune.yml", tmp_path)

    result = runner.invoke(
        app,
        [
            "train",
            "--mlip-config",
            config,
            "--fine-tune",
            "--log",
            log_path,
            "--summary",
            summary_path,
        ],
    )
    try:
        assert Path(model).exists()
        assert Path(compiled_model).exists()
        assert Path(logs_path).is_dir()
        assert Path(results_path).is_dir()
        assert Path(checkpoints_path).is_dir()
    finally:
        # Tidy up models
        Path(model).unlink(missing_ok=True)
        Path(compiled_model).unlink(missing_ok=True)

        # Tidy up directories
        shutil.rmtree(logs_path, ignore_errors=True)
        shutil.rmtree(results_path, ignore_errors=True)
        shutil.rmtree(checkpoints_path, ignore_errors=True)

        # Clean up logger
        logger = logging.getLogger()
        logger.handlers = [
            h for h in logger.handlers if not isinstance(h, logging.FileHandler)
        ]

        assert result.exit_code == 0


def test_fine_tune_no_foundation(tmp_path):
    """Test MLIP fine-tuning raises errors without foundation_model."""
    log_path = tmp_path / "test.log"
    summary_path = tmp_path / "summary.yml"

    config = DATA_PATH / "mlip_fine_tune_no_foundation.yml"

    result = runner.invoke(
        app,
        [
            "train",
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
