"""Test commandline interface with no sub-commands."""

from typer.testing import CliRunner

from janus_core.cli.janus import app
from tests.utils import decode_ansi

runner = CliRunner()


def test_no_args():
    """Test calling `janus` with no arguments/options."""
    result = runner.invoke(app)
    assert result.exit_code == 0
    assert "Usage: janus [OPTIONS] COMMAND [ARGS]..." in decode_ansi(result.stdout)


def test_help():
    """Test calling `janus --help`."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage: janus [OPTIONS] COMMAND [ARGS]..." in decode_ansi(result.stdout)


def test_version():
    """Test calling `janus --version`."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "janus-core version:" in decode_ansi(result.stdout)
