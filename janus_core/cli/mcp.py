"""Set up MCP server commandline interface."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from typer import Option, Typer

from janus_core.mcp.tools import JANUS_MODES


class TransportEnum(str, Enum):
    """Supported MCP server transport modes."""

    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable-http"


ModeEnum = Enum("ModeEnum", {mode.upper(): mode for mode in JANUS_MODES}, type=str)

app = Typer()


@app.command()
def mcp(
    transport: Annotated[
        TransportEnum,
        Option(
            "--transport",
            "-t",
            help="MCP transport mode: stdio, sse, or streamable-http.",
            rich_help_panel="Server",
        ),
    ] = TransportEnum.STDIO,
    host: Annotated[
        str,
        Option(
            help="Host interface to bind for sse or streamable-http transports.",
            rich_help_panel="Server",
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        Option(
            help="Port to bind for sse or streamable-http transports.",
            rich_help_panel="Server",
        ),
    ] = 8000,
    mount_path: Annotated[
        str | None,
        Option(
            help="Optional mount path for SSE transport.",
            rich_help_panel="Server",
        ),
    ] = None,
    mode: Annotated[
        list[ModeEnum] | None,
        Option(
            "--mode",
            "-m",
            help=(
                "Restrict exposed janus-core modes (repeatable). "
                "Defaults to all 10 modes."
            ),
            rich_help_panel="Server",
        ),
    ] = None,
) -> None:
    """
    Start the janus-core Model Context Protocol (MCP) server.

    Parameters
    ----------
    transport
        MCP transport mode ("stdio", "sse", or "streamable-http"). Default is "stdio".
    host
        Bind host for HTTP/SSE transports. Default is "127.0.0.1".
    port
        Bind port for HTTP/SSE transports. Default is 8000.
    mount_path
        Optional mount path for SSE transport. Default is None.
    mode
        Optional list of janus-core modes to expose. Default is None (all modes).
    """
    from janus_core.mcp.server import run_mcp_server

    selected_modes = [m.value for m in mode] if mode else None
    run_mcp_server(
        transport=transport.value,  # type: ignore[arg-type]
        host=host,
        port=port,
        mount_path=mount_path,
        modes=selected_modes,
    )


def main() -> None:
    """Entrypoint for the standalone `janus-mcp` console script."""
    app()
