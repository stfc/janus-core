"""Test train commandline interface."""

from pathlib import Path
import shutil

from mace.cli.run_train import run as run_train  # pylint: disable=unused-import
from typer.testing import CliRunner

from janus_core.cli.janus import app

DATA_PATH = Path(__file__).parent / "data"

runner = CliRunner()


def test_help():
    """Test calling `janus train --help`."""
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "Usage: janus train [OPTIONS]" in result.stdout


def test_train():
    """Test MLIP training."""
    model = "test.model"
    compiled_model = "test_compiled.model"
    logs_path = "logs"
    results_path = "results"
    checkpoints_path = "checkpoints"

    assert not Path(model).exists()
    assert not Path(compiled_model).exists()
    assert not Path(logs_path).exists()
    assert not Path(results_path).exists()
    assert not Path(checkpoints_path).exists()

    config = DATA_PATH / "mlip_train.yml"

    result = runner.invoke(
        app,
        [
            "train",
            "--mlip-config",
            config,
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

        assert result.exit_code == 0


def test_train_with_foundation():
    """Test MLIP training raises error with foundation_model in config."""
    config = DATA_PATH / "mlip_train_invalid.yml"

    result = runner.invoke(
        app,
        [
            "train",
            "--mlip-config",
            config,
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)


def test_fine_tune():
    """Test MLIP fine-tuning."""
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

    config = DATA_PATH / "mlip_fine_tune.yml"

    result = runner.invoke(
        app,
        ["train", "--mlip-config", config, "--fine-tune"],
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

        assert result.exit_code == 0


def test_fine_tune_no_foundation():
    """Test MLIP fine-tuning raises errors without foundation_model."""
    config = DATA_PATH / "mlip_fine_tune_no_foundation.yml"

    result = runner.invoke(
        app,
        ["train", "--mlip-config", config, "--fine-tune"],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)


def test_fine_tune_invalid_foundation():
    """Test MLIP fine-tuning raises errors with invalid foundation_model."""
    config = DATA_PATH / "mlip_fine_tune_invalid_foundation.yml"

    result = runner.invoke(
        app,
        ["train", "--mlip-config", config, "--fine-tune"],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
