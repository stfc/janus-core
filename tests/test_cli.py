"""Test commandline interface with no sub-commands."""

from __future__ import annotations

from typer.testing import CliRunner

from janus_core.cli.janus import app
from tests.utils import strip_ansi_codes

runner = CliRunner()


def test_no_args():
    """Test calling `janus` with no arguments/options."""
    result = runner.invoke(app)
    assert result.exit_code == 0
    assert "Usage: janus [OPTIONS] COMMAND [ARGS]..." in strip_ansi_codes(result.stdout)


def test_help():
    """Test calling `janus --help`."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage: janus [OPTIONS] COMMAND [ARGS]..." in strip_ansi_codes(result.stdout)


def test_version():
    """Test calling `janus --version`."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "janus-core version:" in strip_ansi_codes(result.stdout)
