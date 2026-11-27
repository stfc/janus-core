"""Model Context Protocol (MCP) server for janus-core."""

from __future__ import annotations

from janus_core.mcp.server import (
    MCPTransport,
    create_mcp_server,
    run_mcp_server,
)
from janus_core.mcp.tools import (
    JANUS_MODES,
    inspect_structure,
    list_architectures,
    list_modes,
    run_descriptors,
    run_elasticity,
    run_eos,
    run_geomopt,
    run_md,
    run_mode,
    run_neb,
    run_phonons,
    run_preprocess,
    run_singlepoint,
    run_train,
)

__all__ = [
    "JANUS_MODES",
    "MCPTransport",
    "create_mcp_server",
    "inspect_structure",
    "list_architectures",
    "list_modes",
    "run_descriptors",
    "run_elasticity",
    "run_eos",
    "run_geomopt",
    "run_mcp_server",
    "run_md",
    "run_mode",
    "run_neb",
    "run_phonons",
    "run_preprocess",
    "run_singlepoint",
    "run_train",
]
