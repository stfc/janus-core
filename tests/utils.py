"""Utility functions for tests."""

from __future__ import annotations

import logging
from pathlib import Path
import re

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
        case "chgnet":
            pytest.importorskip("chgnet")
        case "dpa3":
            pytest.importorskip("deepmd")
        case "mace" | "mace_mp" | "mace_off":
            pytest.importorskip("mace")
        case "mattersim":
            pytest.importorskip("mattersim")
        case "nequip":
            pytest.importorskip("nequip")
        case "orb":
            pytest.importorskip("orb_models")
        case "sevennet":
            pytest.importorskip("sevenn")
        case "alignn":
            # alignn caches downloaded model in site-packages
            pytest.importorskip("alignn.ff.ff")
        case "m3gnet":
            pytest.importorskip("matgl")


def check_output_files(summary: Path, output_files: dict[str, Path]) -> None:
    """
    Check output files in summary match expected and created files.

    Parameters
    ----------
    summary
        Path to summary file with outputs from calculation class.
    output_files
        Expected output files to compare with summary.
    """
    assert "output_files" in summary

    for key, value in output_files.items():
        output_file = summary["output_files"][key]
        if isinstance(value, list):
            for file in value:
                assert file.exists(), f"{file} missing"

                if str(file.absolute()) not in output_file:
                    raise ValueError(f"{file} is inconsistent with {output_file}")

        else:
            if value and not value.exists():
                raise ValueError(f"{value} missing")

            if output_file != (str(value.absolute()) if value else None):
                raise ValueError(f"{value} is inconsistent with {output_file}")
