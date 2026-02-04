"""Test train commandline interface."""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import pytest
from pytest import skip
from typer.testing import CliRunner
import yaml

from janus_core.cli.janus import app
from tests.utils import (
    assert_log_contains,
    chdir,
    check_output_files,
    clear_log_handlers,
    rename_atoms_attributes,
    skip_extras,
    strip_ansi_codes,
)

DATA_PATH = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models"
EXTRA_MODEL_PATH = MODEL_PATH / "extra"
NEQUIP_EXTRA_MODEL_PATH = EXTRA_MODEL_PATH / "NequIP-MP-L-0.1.nequip.zip"
SEVENNET_EXTRA_MODEL_PATH = EXTRA_MODEL_PATH / "SevenNet_l3i5.pth"

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
    for file in config["data"].keys() & {
        "train_file_path",
        "test_file_path",
        "val_file_path",
    }:
        pth = DATA_PATH / Path(config["data"][file]).name
        if pth.is_file():
            config["data"][file] = str(pth)

    if fine_tune:
        model_dict = config["training_module"]["model"]
        model = Path(model_dict[f"{model_type}_path"]).name
        for pth in (model, f"extra/{model}"):
            if (MODEL_PATH / pth).is_file():
                model_dict[f"{model_type}_path"] = str(MODEL_PATH / pth)

    # Write out temporary config with corrected paths
    tmp_config = tmp_path / "config.yaml"
    with open(tmp_config, "w", encoding="utf8") as file:
        yaml.dump(config, file, sort_keys=False)

    return tmp_config


def write_tmp_data_sevennet(
    config_path: Path, tmp_path: Path, fine_tune: bool = False
) -> Path:
    """
    Fix paths and data columns, write config and data to tmp_path.

    Parameters
    ----------
    config_path
        Path to yaml config file to be fixed.
    tmp_path
        Temporary path from pytest in which to write corrected config.

    Returns
    -------
    Path.
        Temporary path to corrected config file.
    """
    # Load config from tests/data
    with open(config_path, encoding="utf8") as file:
        config = yaml.safe_load(file)

    # Use DATA_PATH to set paths relative to this test file
    for dataset in config["data"].keys() & {"load_dataset_path", "load_validset_path"}:
        files = config["data"][dataset]
        for i, file in enumerate(files):
            name = Path(file).name
            path = DATA_PATH / name
            if path.exists():
                frames = read(path, index=":")
                # There is currenlty no option to rename these.
                rename_info = {"dft_energy": "energy", "dft_stress": "stress"}
                rename_arrays = {"dft_forces": "forces"}
                for frame in frames:
                    rename_atoms_attributes(frame, rename_info, rename_arrays)
                write(tmp_path / name, frames)
                files[i] = str(tmp_path / name)

    if fine_tune:
        model = Path(config["train"]["continue"]["checkpoint"]).name
        if (MODEL_PATH / "extra" / model).exists():
            config["train"]["continue"]["checkpoint"] = str(
                MODEL_PATH / "extra" / model
            )

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
    skip_extras("mace")

    with chdir(tmp_path):
        results_dir = Path.cwd() / "janus_results"

        model = results_dir / "test.model"
        compiled_model = results_dir / "test_compiled.model"
        checkpoints_path = results_dir / "checkpoints"
        logs_path = results_dir / "logs"
        log_path = results_dir / "train-log.yml"
        summary_path = results_dir / "train-summary.yml"

        config = write_tmp_config_mace(DATA_PATH / "mlip_train.yml", Path.cwd())

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
        logs_path = results_dir / "logs"

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
        results_dir = Path.cwd() / "janus_results_no_carbon"

        model_path = results_dir / "test.model"
        compiled_path = results_dir / "test_compiled.model"
        checkpoints_path = results_dir / "checkpoints"
        logs_path = results_dir / "logs"

        log_path = tmp_path / "test.log"
        summary_path = tmp_path / "summary.yml"

        config = write_tmp_config_mace(DATA_PATH / "mlip_train.yml", Path.cwd())

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

        with open(metrics_path, encoding="utf-8") as metrics:
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


@pytest.mark.skipif(
    not NEQUIP_EXTRA_MODEL_PATH.exists(),
    reason=f"Extra model: {NEQUIP_EXTRA_MODEL_PATH} not downloaded.",
)
def test_nequip_fine_tune_foundation(tmp_path):
    """Test fine-tuning with a nequip foundation model."""
    skip_extras("nequip")

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

        with open(metrics_path, encoding="utf-8") as metrics:
            header = metrics.readline().split(",")

        assert header[:3] == ["epoch", "lr-Adam", "step"]


def test_sevennet_train(tmp_path):
    """Test training with sevennet."""
    skip_extras("sevennet")

    with chdir(tmp_path):
        log_path = tmp_path / "test.log"
        summary_path = tmp_path / "summary.yml"

        results_dir = Path("janus_results")

        checkpoints_paths = [
            results_dir / f"checkpoint_{ver}.pth" for ver in ("0", "1", "best")
        ]
        sevennet_log_path = results_dir / "log.sevenn"
        sevenn_data_path = results_dir / "sevenn_data"
        metrics_path = results_dir / "lc.csv"

        config_path = DATA_PATH / "sevennet_train.yml"
        config_path = write_tmp_data_sevennet(config_path, tmp_path)

        result = runner.invoke(
            app,
            [
                "train",
                "sevennet",
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
        assert sevennet_log_path.exists()
        assert sevenn_data_path.is_dir()
        assert metrics_path.exists()

        for checkpoint in checkpoints_paths:
            assert checkpoint.exists()

        with open(metrics_path) as metrics:
            lines = metrics.readlines()
            assert len(lines) == 3
            assert lines[0].split(",")[0] == "epoch"


def test_sevennet_fine_tune_foundation(tmp_path):
    """Test training with sevennet."""
    skip_extras("sevennet")

    if not SEVENNET_EXTRA_MODEL_PATH.exists():
        skip(f"Extra model: {SEVENNET_EXTRA_MODEL_PATH} not downloaded.")

    with chdir(tmp_path):
        log_path = tmp_path / "test.log"
        summary_path = tmp_path / "summary.yml"

        results_dir = Path("janus_results")

        checkpoints_paths = [
            results_dir / f"checkpoint_{ver}.pth" for ver in ("0", "1", "best")
        ]
        sevennet_log_path = results_dir / "log.sevenn"
        sevenn_data_path = results_dir / "sevenn_data"
        metrics_path = results_dir / "lc.csv"

        config_path = DATA_PATH / "sevennet_fine_tune.yml"
        config_path = write_tmp_data_sevennet(config_path, tmp_path, True)

        result = runner.invoke(
            app,
            [
                "train",
                "sevennet",
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
        assert sevennet_log_path.exists()
        assert sevenn_data_path.is_dir()
        assert metrics_path.exists()

        for checkpoint in checkpoints_paths:
            assert checkpoint.exists()

        with open(metrics_path) as metrics:
            lines = metrics.readlines()
            assert len(lines) == 2
            assert lines[0].split(",")[0] == "epoch"
