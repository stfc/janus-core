"""MCP server assembly and transport runner for janus-core."""

from __future__ import annotations

from collections.abc import Sequence
import inspect
import json
from typing import Any, Literal

from janus_core import __version__
from janus_core.mcp.tools import (
    JANUS_MODES,
    MODE_HANDLERS,
    JanusMode,
    inspect_structure,
    list_architectures,
    list_modes,
    run_mode,
)

MCPTransport = Literal["stdio", "sse", "streamable-http"]


def _import_mcp_server_class() -> tuple[type[Any], bool]:
    """
    Lazily import the MCP server class from the optional `mcp` SDK.

    Supports both `mcp>=2.0` (`mcp.server.mcpserver.MCPServer`) and `mcp<2.0`
    (`mcp.server.fastmcp.FastMCP`).

    Returns
    -------
    tuple[type[Any], bool]
        Tuple of `(server_class, is_v2)` where `is_v2` is True for `mcp>=2.0`.
    """
    try:
        from mcp.server.mcpserver import MCPServer

        return MCPServer, True
    except ImportError:
        pass

    try:
        from mcp.server.fastmcp import FastMCP

        return FastMCP, False
    except ImportError as exc:
        raise ImportError(
            "The optional 'mcp' dependency is required to run the janus-core MCP "
            "server. Install it with: uv pip install 'janus-core[mcp]'"
        ) from exc


def _validate_modes(modes: Sequence[str] | None) -> list[JanusMode]:
    """
    Validate and normalize requested janus-core modes.

    Parameters
    ----------
    modes
        Optional sequence of mode names to expose. None enables all modes.

    Returns
    -------
    list[JanusMode]
        Validated list of mode names preserving order without duplicates.
    """
    if not modes:
        return list(JANUS_MODES.keys())

    selected: list[JanusMode] = []
    unknown: list[str] = []
    for mode in modes:
        if mode not in JANUS_MODES:
            unknown.append(mode)
        elif mode not in selected:
            selected.append(mode)  # type: ignore[arg-type]

    if unknown:
        valid = ", ".join(JANUS_MODES.keys())
        bad = ", ".join(unknown)
        raise ValueError(f"Unknown janus-core mode(s): {bad}. Valid modes are: {valid}")
    return selected


def create_mcp_server(
    name: str = "janus-core",
    host: str = "127.0.0.1",
    port: int = 8000,
    modes: Sequence[str] | None = None,
) -> Any:
    """
    Create and configure the janus-core MCP server.

    Registers discovery tools (`list_modes`, `list_architectures`,
    `inspect_structure`), the unified `run_mode` dispatcher, dedicated tools for
    every enabled janus-core mode, and read-only MCP resources.

    Parameters
    ----------
    name
        Display name of the MCP server. Default is "janus-core".
    host
        Bind host used by HTTP/SSE transports (used at init on mcp v1).
        Default is "127.0.0.1".
    port
        Bind port used by HTTP/SSE transports (used at init on mcp v1).
        Default is 8000.
    modes
        Optional subset of janus-core modes to expose. Default is None (all modes).

    Returns
    -------
    Any
        Configured `MCPServer` (mcp v2) or `FastMCP` (mcp v1) instance ready to run.
    """
    server_cls, is_v2 = _import_mcp_server_class()
    enabled_modes = _validate_modes(modes)

    instructions = (
        "STFC janus-core MCP server for machine-learnt interatomic potential "
        "(MLIP) atomistic simulations. Supports singlepoint, geomopt, md, "
        "phonons, eos, elasticity, neb, descriptors, train, and preprocess modes."
    )

    init_kwargs: dict[str, Any] = {
        "name": name,
        "instructions": instructions,
    }
    if is_v2:
        init_kwargs["version"] = __version__
    else:
        init_kwargs["host"] = host
        init_kwargs["port"] = port

    server = server_cls(**init_kwargs)

    # Discovery & utility tools
    server.tool()(list_modes)
    server.tool()(list_architectures)
    server.tool()(inspect_structure)

    # Dedicated tool per enabled janus-core mode
    for mode in enabled_modes:
        handler = MODE_HANDLERS[mode]
        server.tool(name=mode, description=JANUS_MODES[mode])(handler)

    # Unified mode runner respecting enabled_modes
    def run_janus_mode(
        mode: JanusMode, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Execute any enabled janus-core calculation or training mode by name.

        Parameters
        ----------
        mode
            Name of the janus-core mode to run.
        params
            Dictionary of parameters for the chosen mode. Default is None.

        Returns
        -------
        dict[str, Any]
            JSON-serializable result dictionary.
        """
        if mode not in enabled_modes:
            allowed = ", ".join(enabled_modes)
            raise ValueError(
                f"Mode '{mode}' is not enabled on this server. Enabled modes: {allowed}"
            )
        return run_mode(mode=mode, params=params)

    server.tool(
        name="run_mode",
        description="Run any enabled janus-core calculation or training mode by name.",
    )(run_janus_mode)

    # Read-only MCP resources
    @server.resource("janus://version")
    def resource_version() -> str:
        """
        Return the installed janus-core version string.

        Returns
        -------
        str
            Installed janus-core version string.
        """
        return __version__

    @server.resource("janus://modes")
    def resource_modes() -> str:
        """
        Return JSON description of enabled janus-core modes.

        Returns
        -------
        str
            JSON string mapping enabled mode names to descriptions.
        """
        payload = {mode: JANUS_MODES[mode] for mode in enabled_modes}
        return json.dumps(payload, indent=2)

    @server.resource("janus://architectures")
    def resource_architectures() -> str:
        """
        Return JSON list of supported MLIP architectures, devices, and ensembles.

        Returns
        -------
        str
            JSON string describing supported architectures, devices, and ensembles.
        """
        return json.dumps(list_architectures(), indent=2)

    return server


def run_mcp_server(
    transport: MCPTransport = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    mount_path: str | None = None,
    modes: Sequence[str] | None = None,
) -> None:
    """
    Create and start the janus-core MCP server using the selected transport.

    Parameters
    ----------
    transport
        MCP transport mode: "stdio", "sse", or "streamable-http". Default is "stdio".
    host
        Bind host for HTTP/SSE transports. Default is "127.0.0.1".
    port
        Bind port for HTTP/SSE transports. Default is 8000.
    mount_path
        Optional mount path (v1) or SSE path (v2) for SSE transport. Default is None.
    modes
        Optional subset of janus-core modes to expose. Default is None (all modes).
    """
    valid_transports = ("stdio", "sse", "streamable-http")
    if transport not in valid_transports:
        raise ValueError(
            f"Unsupported transport '{transport}'. "
            f"Valid transports: {', '.join(valid_transports)}"
        )

    server = create_mcp_server(host=host, port=port, modes=modes)

    if transport == "stdio":
        server.run(transport="stdio")
        return

    # Check if run() accepts host/port kwargs (mcp v2) vs configured at init (mcp v1)
    run_sig = inspect.signature(server.run)
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in run_sig.parameters.values()
    )

    if accepts_var_kwargs:
        run_kwargs: dict[str, Any] = {"host": host, "port": port}
        if transport == "sse" and mount_path is not None:
            run_kwargs["sse_path"] = mount_path
        server.run(transport=transport, **run_kwargs)
    elif transport == "sse":
        server.run(transport="sse", mount_path=mount_path)
    else:
        server.run(transport=transport)
