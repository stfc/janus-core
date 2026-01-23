"""Test train commandline interface."""

from __future__ import annotations

from pathlib import Path

from pytest import skip
from typer.testing import CliRunner
import yaml

from janus_core.cli.janus import app
from tests.utils import (
    assert_log_contains,
    chdir,
    check_output_files,
    clear_log_handlers,
    skip_extras,
    strip_ansi_codes,
)

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models"
NEQUIP_EXTRA_MODEL_PATH = (
    Path(__file__).parent / "models" / "extra" / "NequIP-MP-L-0.1.nequip.zip"
)

runner = CliRunner()


def write_tmp_config_mace(config_path: Path, tmp_path: Path) -> Path:
    """
    Fix paths in config files and write corrected config to tmp_path for mace.

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


def write_tmp_config_nequip(
    config_path: Path,
    tmp_path: Path,
    fine_tune: bool = False,
    model_type: str = "package",
) -> Path:
    """
    Fix paths in config files and write corrected config to tmp_path for nequip.

    Parameters
    ----------
    config_path
        Path to yaml config file to be fixed.
    tmp_path
        Temporary path from pytest in which to write corrected config.
    model_path
        Path to a saved model.

    Returns
    -------
    Path
        Temporary path to corrected config file.
    """
    # Load config from tests/data
    with open(config_path, encoding="utf8") as file:
        config = yaml.safe_load(file)

    # Use DATA_PATH to set paths relative to this test file
    for file in ("train_file_path", "test_file_path", "val_file_path"):
        if (
            file in config["data"]
            and (DATA_PATH / Path(config["data"][file]).name).exists()
        ):
            config["data"][file] = str(DATA_PATH / Path(config["data"][file]).name)

    if fine_tune:
        model = Path(config["training_module"]["model"][model_type + "_path"]).name
        if (MODEL_PATH / model).exists():
            config["training_module"]["model"][model_type + "_path"] = str(
                MODEL_PATH / model
            )
        elif (MODEL_PATH / "extra" / model).exists():
            config["training_module"]["model"][model_type + "_path"] = str(
                MODEL_PATH / "extra" / model
            )

    # Write out temporary config with corrected paths
    tmp_config = tmp_path / "config.yaml"
    with open(tmp_config, "w", encoding="utf8") as file:
        yaml.dump(config, file, sort_keys=False)

    return tmp_config


def test_help():
    """Test calling `janus train --help`."""
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "Usage: janus train [OPTIONS]" in strip_ansi_codes(result.stdout)


def test_train(tmp_path):
    """Test MLIP training."""
    skip_extras("mace")

    with chdir(tmp_path):
        results_dir = Path("./janus_results")

        model = results_dir / "test.model"
        compiled_model = results_dir / "test_compiled.model"
        checkpoints_path = results_dir / "checkpoints"
        logs_path = results_dir / "logs"
        log_path = results_dir / "train-log.yml"
        summary_path = results_dir / "train-summary.yml"

        config = write_tmp_config_mace(DATA_PATH / "mlip_train.yml", Path())

        result = runner.invoke(
            app,
            [
                "train",
                "mace",
                "--mlip-config",
                config,
            ],
        )
        assert result.exit_code == 0

        assert results_dir.is_dir()
        assert model.exists()
        assert compiled_model.exists()
        assert logs_path.is_dir()
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
    skip_extras("mace")

    results_dir = tmp_path / "janus_results"

    log_path = results_dir / "test.log"
    summary_path = results_dir / "summary.yml"
    config = write_tmp_config_mace(DATA_PATH / "mlip_train_invalid.yml", tmp_path)

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
    skip_extras("mace")

    with chdir(tmp_path):
        results_dir = Path("janus_results")

        model = results_dir / "test-finetuned.model"
        compiled_model = results_dir / "test-finetuned_compiled.model"
        checkpoints_path = results_dir / "checkpoints"

        log_path = tmp_path / "test.log"
        summary_path = tmp_path / "summary.yml"
        logs_path = results_dir / Path("logs")

        config = write_tmp_config_mace(DATA_PATH / "mlip_fine_tune.yml", Path())

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

        assert results_dir.is_dir()
        assert model.exists()
        assert compiled_model.exists()
        assert logs_path.is_dir()
        assert checkpoints_path.is_dir()

        clear_log_handlers()


def test_fine_tune_no_foundation(tmp_path):
    """Test MLIP fine-tuning raises errors without foundation_model."""
    skip_extras("mace")

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
    skip_extras("mace")

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
    skip_extras("mace")

    with chdir(tmp_path):
        results_dir = Path("./janus_results_no_carbon")

        model_path = results_dir / "test.model"
        compiled_path = results_dir / "test_compiled.model"
        checkpoints_path = results_dir / "checkpoints"
        logs_path = results_dir / "logs"

        log_path = tmp_path / "test.log"
        summary_path = tmp_path / "summary.yml"

        config = write_tmp_config_mace(DATA_PATH / "mlip_train.yml", Path())

        result = runner.invoke(
            app,
            [
                "train",
                "mace",
                "--mlip-config",
                config,
                "--file-prefix",
                results_dir,
                "--no-tracker",
                "--log",
                log_path,
                "--summary",
                summary_path,
            ],
        )
        assert result.exit_code == 0

        assert results_dir.exists()
        assert model_path.exists()
        assert checkpoints_path.exists()
        assert compiled_path.exists()
        assert logs_path.exists()

        with open(summary_path, encoding="utf8") as file:
            train_summary = yaml.safe_load(file)
        assert "emissions" not in train_summary

        clear_log_handlers()


def test_nequip_train(tmp_path):
    """Test training with nequip."""
    skip_extras("nequip")

    with chdir(tmp_path):
        log_path = tmp_path / "test.log"
        summary_path = tmp_path / "summary.yml"

        results_dir = Path("janus_results")

        config_path = DATA_PATH / "nequip_train.yaml"

        last_ckpt_path = results_dir / "last.ckpt"
        best_ckpt_path = results_dir / "best.ckpt"
        train_log_path = results_dir / "train_log"
        metrics_path = results_dir / "train_log/version_0/metrics.csv"

        config_path = write_tmp_config_nequip(config_path, tmp_path)

        result = runner.invoke(
            app,
            [
                "train",
                "nequip",
                "--mlip-config",
                config_path,
                "--log",
                log_path,
                "--summary",
                summary_path,
            ],
        )
        assert result.exit_code == 0

        assert results_dir.exists()
        assert log_path.exists()
        assert summary_path.exists()
        assert best_ckpt_path.exists()
        assert last_ckpt_path.exists()
        assert train_log_path.exists()
        assert metrics_path.exists()

        with open(metrics_path) as metrics:
            header = metrics.readline().split(",")

        assert header[:3] == ["epoch", "lr-Adam", "step"]


def test_nequip_train_invalid_config_suffix(tmp_path):
    """Test training with nequip."""
    skip_extras("nequip")

    with chdir(tmp_path):
        config_path = DATA_PATH / "mlip_train.yml"

        config_path = write_tmp_config_mace(config_path, tmp_path)

        result = runner.invoke(
            app,
            ["train", "nequip", "--mlip-config", config_path],
        )
        assert result.exit_code == 1
        assert isinstance(result.exception, ValueError)


def test_nequip_fine_tune_foundation(tmp_path):
    """Test fine-tuning with a nequip foundation model."""
    skip_extras("nequip")

    if not NEQUIP_EXTRA_MODEL_PATH.exists():
        skip(f"Extra model: {NEQUIP_EXTRA_MODEL_PATH} not downloaded.")

    with chdir(tmp_path):
        results_dir = Path("janus_results")

        log_path = tmp_path / "ft_test.log"
        summary_path = tmp_path / "ft_summary.yml"

        best_ckpt_path = results_dir / "best.ckpt"
        train_log_path = results_dir / "train_log"
        metrics_path = results_dir / "train_log/version_0/metrics.csv"

        config_path = write_tmp_config_nequip(
            DATA_PATH / "nequip_fine_tune.yaml", tmp_path, True
        )

        result = runner.invoke(
            app,
            [
                "train",
                "nequip",
                "--mlip-config",
                config_path,
                "--fine-tune",
                "--log",
                log_path,
                "--summary",
                summary_path,
            ],
        )
        assert result.exit_code == 0

        assert results_dir.exists()
        assert log_path.exists()
        assert summary_path.exists()
        assert best_ckpt_path.exists()
        assert train_log_path.exists()
        assert metrics_path.exists()

        with open(metrics_path) as metrics:
            header = metrics.readline().split(",")

        assert header[:3] == ["epoch", "lr-Adam", "step"]
