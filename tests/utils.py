"""Utility functions for tests."""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
import re
from typing import Any
from urllib.request import urlopen, Request

from ase import Atoms
from ase.io import read
import pytest
import yaml

from janus_core.helpers.janus_types import MaybeSequence, PathLike

def read_atoms(path: Path) -> Atoms | None:
    """
    Read Atoms structure file, and delete file regardless of success.

    Parameters
    ----------
    path
        Path to file containing Atoms structure to be read.

    Returns
    -------
    Atoms | None
        Atoms structure read from file, or None is any Exception is thrown.
    """
    assert path.exists()
    try:
        atoms = read(path)
        assert isinstance(atoms, Atoms)
    finally:
        # Ensure file is still deleted if read fails
        path.unlink()

    return atoms if atoms else None


def assert_log_contains(
    log_path: PathLike,
    *,
    includes: MaybeSequence[str] = (),
    excludes: MaybeSequence[str] = (),
) -> None:
    """
    Check messages are present or not within a yaml-formatted log file.

    Parameters
    ----------
    log_path
        Path to log file to check messsages of.
    includes
        Messages that must appear in the log file. Default is None.
    excludes
        Messages that must not appear in the log file. Default is None.
    """
    # Convert single strings to iterable
    includes = [includes] if isinstance(includes, str) else includes
    excludes = [excludes] if isinstance(excludes, str) else excludes

    # Read log file
    with open(log_path, encoding="utf8") as log_file:
        logs = yaml.safe_load(log_file)
    # Nested join as log["message"] may be a list
    messages = "".join(
        "".join(log["message"]) if log["message"] else "" for log in logs
    )

    assert all(inc in messages for inc in includes)
    assert all(exc not in messages for exc in excludes)


def strip_ansi_codes(output: str) -> str:
    """
    Remove any ANSI sequences from output string.

    Based on:
    https://stackoverflow.com/questions/14693701/how-can-i-remove-the-ansi-escape-sequences-from-a-string-in-python/14693789#14693789

    Parameters
    ----------
    output
        Output that may contain ANSI sequences to be removed.

    Returns
    -------
    str
        Output with ANSI sequences removed.
    """
    # 7-bit C1 ANSI sequences
    ansi_escape = re.compile(
        r"""
        \x1B  # ESC
        (?:   # 7-bit C1 Fe (except CSI)
            [@-Z\\-_]
        |     # or [ for CSI, followed by a control sequence
            \[
            [0-?]*  # Parameter bytes
            [ -/]*  # Intermediate bytes
            [@-~]   # Final byte
        )
    """,
        re.VERBOSE,
    )
    return ansi_escape.sub("", output)


def clear_log_handlers():
    """Clear all log handlers."""
    logger = logging.getLogger()
    logger.handlers = [
        h for h in logger.handlers if not isinstance(h, logging.FileHandler)
    ]


def skip_extras(arch: str):
    """
    Skip tests if unable to import module for architecture.

    Parameters
    ----------
    arch
        Model architecture.
    """
    match arch:
        case "alignn":
            # alignn caches downloaded model in site-packages
            pytest.importorskip("alignn.ff.ff")
        case "alphanet":
            pytest.importorskip("alphanet")
        case "chgnet":
            pytest.importorskip("chgnet")
        case "dpa3":
            pytest.importorskip("deepmd")
        case "equiformer":
            pytest.importorskip("fairchem.core.common.relaxation")
        case "esen":
            pytest.importorskip("fairchem.core.common.relaxation")
            from huggingface_hub.utils._auth import get_token

            if not get_token():
                pytest.skip("Unable to download model")
        case "grace":
            pytest.importorskip("tensorpotential")
        case "mace" | "mace_mp" | "mace_off" | "mace_omol":
            pytest.importorskip("mace")
        case "mattersim":
            pytest.importorskip("mattersim")
        case "m3gnet":
            pytest.importorskip("matgl")
        case "nequip":
            pytest.importorskip("nequip")
        case "orb":
            pytest.importorskip("orb_models")
        case "pet_mad":
            pytest.importorskip("pet_mad")
        case "sevennet":
            pytest.importorskip("sevenn")
        case "uma":
            pytest.importorskip("fairchem.core.calculate")
            from huggingface_hub.utils._auth import get_token

            if not get_token():
                pytest.skip("Unable to download model")


def check_output_files(summary: dict[str, Any], output_files: dict[str, Path]) -> None:
    """
    Check output files in summary match expected and created files.

    Parameters
    ----------
    summary
        Summary file dictionary.
    output_files
        Expected output files to compare with summary.
    """
    assert "output_files" in summary

    for key, value in output_files.items():
        output_file = summary["output_files"][key]
        if isinstance(value, list):
            output_file = [
                Path(file).absolute().as_posix() if file else None
                for file in output_file
            ]
            for file in value:
                assert file.exists(), f"{file} missing"

                assert str(file.absolute().as_posix()) in output_file, (
                    f"{file} is inconsistent with {output_file}"
                )
        elif value:
            assert value.exists(), f"{value} missing"

            output_file = Path(output_file).absolute().as_posix()
            assert output_file == str(value.absolute().as_posix()), (
                f"{value} is inconsistent with {output_file}"
            )


@contextlib.contextmanager
def chdir(path):
    """Change working directory and return to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def download_alphanet_model(model_name: str = "MATPES") -> tuple[Path, Path]:
    """
    Download AlphaNet MATPES model files from GitHub.

    Parameters
    ----------
    model_name
        Name of pretrained model. Default is "MATPES".

    Returns
    -------
    tuple[Path, Path]
        Paths to (checkpoint_file, config_file).
    """
    # Directory for cached test data
    TEST_CACHE_DIR = Path.home() / ".cache" / "janus-core" / "test-data"
    cache_dir = TEST_CACHE_DIR / "alphanet" / model_name
    ckpt_path = cache_dir / "r2scan_1021.ckpt"
    json_path = cache_dir / "matpes.json"
    
    # Return if already cached
    if ckpt_path.exists() and json_path.exists():
        return ckpt_path, json_path
    
    # Download if not cached
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"https://media.githubusercontent.com/media/zmyybc/AlphaNet/main/pretrained/{model_name}"
    
    print(f"Downloading AlphaNet {model_name} checkpoint...")
    request = Request(f"{base_url}/r2scan_1021.ckpt", headers={"User-Agent": "janus-core"})
    with urlopen(request, timeout=300) as response:
        ckpt_path.write_bytes(response.read())
    
    print(f"Downloading AlphaNet {model_name} config...")
    request = Request(f"{base_url}/matpes.json", headers={"User-Agent": "janus-core"})
    with urlopen(request, timeout=30) as response:
        json_path.write_bytes(response.read())
    
    return ckpt_path, json_path
