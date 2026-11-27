==========
MCP server
==========

``janus-core`` provides an optional `Model Context Protocol (MCP) <https://modelcontextprotocol.io/>`_ server
that exposes all ``janus-core`` calculation and training modes as structured tools and read-only resources
for AI coding assistants, scientific agents, and custom MCP clients.

For a step-by-step walkthrough covering local usage, programmatic MCP clients, and production deployment,
see the :doc:`MCP Server Tutorial </tutorials/mcp>`.

Installation
============

Install ``janus-core`` with the ``mcp`` optional extra alongside your preferred MLIP extra (for example, ``mace``):

.. code-block:: bash

    uv pip install "janus-core[mcp,mace]"

Or with ``pip``:

.. code-block:: bash

    python3 -m pip install "janus-core[mcp,mace]"

.. note::

    The MCP server supports both ``mcp >= 2.0`` (``MCPServer``) and ``mcp < 2.0`` (``FastMCP``) SDK releases.
    Importing ``janus_core`` or using other ``janus`` CLI subcommands does not require ``mcp`` to be installed.

Available tools and resources
=============================

Discovery and utility tools
---------------------------

- ``list_modes`` — Return all 10 supported ``janus-core`` calculation and training modes with descriptions.
- ``list_architectures`` — List supported MLIP architectures, compute devices, MD ensembles, properties, phonon calculations, and equation-of-state fits.
- ``inspect_structure`` — Inspect an ASE-readable structure file (CIF, XYZ, POSCAR, extxyz) and return chemical formula, atom count, elements, PBC flags, cell parameters, and volume.

Calculation and training modes
------------------------------

Every ``janus-core`` mode is exposed both as its own dedicated MCP tool and through the unified ``run_mode(mode, params)`` dispatcher:

.. list-table:: Exposed ``janus-core`` MCP mode tools
   :header-rows: 1
   :widths: 18 82

   * - Mode / Tool
     - Description
   * - ``singlepoint``
     - Static single-point energy, forces, stress tensor, or Hessian.
   * - ``geomopt``
     - Geometry optimization and optional unit-cell relaxation (defaults to ``FrechetCellFilter``).
   * - ``md``
     - Molecular dynamics across ``nve``, ``nvt``, ``nvt-nh``, ``nvt-csvr``, ``npt``, ``nph``, ``npt-mtk``, ``npt-mtk-iso``, and ``npt-mtk-aniso``.
   * - ``phonons``
     - Phonon force constants, band structures, DOS, PDOS, q-points, and thermal properties via Phonopy.
   * - ``eos``
     - Equation of state energy–volume fitting, bulk modulus, and equilibrium volume.
   * - ``elasticity``
     - Full 6x6 Voigt elasticity stiffness tensor and Voigt–Reuss–Hill elastic moduli.
   * - ``neb``
     - Nudged Elastic Band (NEB / CI-NEB) minimum energy pathway search.
   * - ``descriptors``
     - Invariant MLIP descriptor extraction (per-structure, per-element, or per-atom).
   * - ``train``
     - Train or fine-tune an MLIP from a YAML configuration file.
   * - ``preprocess``
     - Preprocess MLIP training datasets to HDF5 format.

All MCP calculation tools default to ``track_carbon=False`` and ``attach_logger=False`` for low-overhead,
headless execution and return JSON-serializable dictionaries.

Read-only MCP resources
-----------------------

- ``janus://version`` — Installed ``janus-core`` version string.
- ``janus://modes`` — JSON map of enabled modes and descriptions.
- ``janus://architectures`` — JSON catalog of supported MLIP architectures, devices, ensembles, and EoS fits.

Usage
=====

Command line interface
----------------------

Start the MCP server via ``janus mcp`` or the standalone ``janus-mcp`` console script:

.. code-block:: bash

    # Default stdio transport (all 10 modes enabled)
    janus mcp

    # Equivalent standalone entrypoint
    janus-mcp --transport stdio

    # Streamable HTTP transport on a custom host and port
    janus mcp --transport streamable-http --host 127.0.0.1 --port 8000

    # Server-Sent Events (SSE) transport with a custom mount/SSE path
    janus mcp --transport sse --host 127.0.0.1 --port 8000 --mount-path /janus/sse

    # Restrict the server to a subset of modes
    janus mcp --transport stdio --mode singlepoint --mode geomopt --mode md

CLI options:

- ``--transport``, ``-t`` — MCP transport protocol: ``stdio`` (default), ``sse``, or ``streamable-http``.
- ``--host`` — Bind address for ``sse`` and ``streamable-http`` transports (default: ``127.0.0.1``).
- ``--port`` — Bind port for ``sse`` and ``streamable-http`` transports (default: ``8000``).
- ``--mount-path`` — Optional mount path (``mcp`` v1) or SSE endpoint path (``mcp`` v2) for ``sse`` transport.
- ``--mode``, ``-m`` — Repeatable option restricting which ``janus-core`` modes are registered. When omitted, all 10 modes are enabled.

Python API
----------

You can also create and launch the server programmatically:

.. code-block:: python

    from janus_core.mcp import create_mcp_server, run_mcp_server

    # Create a configured server object (e.g. for testing or custom ASGI mounting)
    server = create_mcp_server(
        name="janus-core",
        modes=["singlepoint", "geomopt", "md"],
    )

    # Or start serving directly
    run_mcp_server(
        transport="streamable-http",
        host="127.0.0.1",
        port=8000,
    )

Connecting from MCP clients
---------------------------

For local clients (such as Antigravity, Claude Desktop, Cursor, or VS Code) using ``stdio`` transport,
add an entry to your client's MCP configuration file:

.. code-block:: json

    {
      "mcpServers": {
        "janus-core": {
          "command": "uv",
          "args": [
            "run",
            "--no-project",
            "--python",
            "/path/to/env/bin/python",
            "-m",
            "janus_core.cli.janus",
            "mcp",
            "--transport",
            "stdio"
          ]
        }
      }
    }

If ``janus-mcp`` is already on ``PATH`` in your active environment, this simplifies to:

.. code-block:: json

    {
      "mcpServers": {
        "janus-core": {
          "command": "janus-mcp",
          "args": ["--transport", "stdio"]
        }
      }
    }

For remote clients connecting over Streamable HTTP:

.. code-block:: json

    {
      "mcpServers": {
        "janus-core": {
          "url": "http://127.0.0.1:8000/mcp"
        }
      }
    }

Deployment
==========

Choosing a transport
--------------------

.. list-table:: Recommended MCP transport by deployment scenario
   :header-rows: 1
   :widths: 22 30 48

   * - Transport
     - Best for
     - Notes
   * - ``stdio``
     - Local IDE / desktop agents
     - Zero open network ports; client spawns and manages the server process lifecycle.
   * - ``streamable-http``
     - Remote workstations, containers, shared lab servers
     - Modern HTTP transport (endpoint ``/mcp``); works cleanly behind reverse proxies.
   * - ``sse``
     - Legacy SSE-based MCP clients
     - Uses Server-Sent Events (endpoint ``/sse`` by default, configurable via ``--mount-path``).

Remote GPU workstation via SSH tunnel
-------------------------------------

To run MLIP calculations on a remote GPU machine without exposing the MCP port to the network:

1. Start the server bound to loopback on the remote host:

   .. code-block:: bash

       janus mcp --transport streamable-http --host 127.0.0.1 --port 8000

2. Forward the port from your local machine over SSH:

   .. code-block:: bash

       ssh -N -L 8000:127.0.0.1:8000 user@gpu-workstation

3. Point your local MCP client at ``http://127.0.0.1:8000/mcp``.

Container deployment (Docker / Podman)
--------------------------------------

You can containerize the MCP server for reproducible deployments (including optional GPU access):

.. code-block:: dockerfile

    FROM python:3.12-slim

    RUN pip install --no-cache-dir uv && \
        uv pip install --system "janus-core[mcp,mace]"

    WORKDIR /workspace
    EXPOSE 8000

    ENTRYPOINT ["janus-mcp", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8000"]

Run the container with a mounted data volume (and ``--gpus all`` for CUDA acceleration):

.. code-block:: bash

    docker build -t janus-core-mcp .
    docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/data:/workspace/data" janus-core-mcp

systemd service
---------------

For a persistent lab server managed by ``systemd``, create ``/etc/systemd/system/janus-mcp.service``:

.. code-block:: ini

    [Unit]
    Description=janus-core MCP Server
    After=network.target

    [Service]
    Type=simple
    User=janus
    WorkingDirectory=/srv/janus-workspace
    ExecStart=/opt/micromamba/envs/janus/bin/janus-mcp --transport streamable-http --host 127.0.0.1 --port 8000
    Restart=on-failure
    RestartSec=5

    [Install]
    WantedBy=multi-user.target

Enable and start the service:

.. code-block:: bash

    sudo systemctl daemon-reload
    sudo systemctl enable --now janus-mcp
    sudo systemctl status janus-mcp

Security and operational recommendations
----------------------------------------

.. warning::

    MCP tools read and write structure/trajectory files on the server filesystem and execute compute-intensive
    simulations. Never bind ``--host 0.0.0.0`` on an untrusted network without placing the server behind an
    authenticating TLS reverse proxy (such as Nginx or Caddy) and isolating the working directory.

- **Bind to loopback by default**: Keep ``--host 127.0.0.1`` and use SSH tunneling or a TLS reverse proxy with authentication for remote access.
- **Least-privilege mode filtering**: Use ``--mode`` to disable heavy or write-intensive modes such as ``train`` and ``preprocess`` on shared inference servers (for example, ``janus mcp --mode singlepoint --mode geomopt --mode eos``).
- **Resource limits**: When deploying via Docker or ``systemd``, configure memory, CPU/GPU, and working-directory quotas to protect shared hosts from oversized MD or phonon jobs.
